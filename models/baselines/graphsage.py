"""GraphSAGE baseline (2-layer), per Augur build spec (baselines section
unchanged v1/v2 -> v3).

Fairness decision (resolved explicitly, not decided silently -- see report):
GraphSAGE and GAT are standard single-relation-graph layers, but this
project's real graph has three relations (tdt, tbt, tft). CARE-GNN's own RL
selector converged to finding tdt/tbt nearly uninformative relative to tft
(Level 6 final state: p_tdt~0.05-0.10, p_tbt~0.10, p_tft~0.5 at both layers).
If these baselines only saw tdt, the ablation comparison would be unfair to
them: CARE-GNN would look artificially stronger because it simply had access
to more of the graph, not because of its architecture (label-aware
similarity, RL filtering, relation-aware weighting). To isolate architecture
as the variable, both baselines train on the UNION of all three relations'
edges, deduplicated into one homogeneous edge_index: 1,979,669 unique
directed edges from 1,984,852 raw (5,183 exact-duplicate directed pairs
removed -- overlap between relations, e.g. a tdt edge that also happens to
satisfy tft's similarity threshold).

Reuses the exact cached graph artifacts CARE-GNN trained on (data/processed/
via graph.loader.load_processed / graph.relations.load_relations) -- nothing
is regenerated or reprocessed here.
"""

import os
import resource
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
import mlflow

from config import CARE_GNN_CONFIG, FRAUD_CLASS_WEIGHT
from graph.loader import PROCESSED_DIR, load_processed
from graph.relations import load_relations
from training.evaluator import evaluate
from training.loss import WeightedFocalLoss

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CHECKPOINT_PATH = PROJECT_ROOT / "models" / "checkpoints" / "graphsage_best.pt"
MLRUNS_PATH = PROJECT_ROOT / "mlruns"
EVAL_EVERY_N_EPOCHS = 5


def build_merged_edge_index(adj_tdt: np.ndarray, adj_tbt: np.ndarray, adj_tft: np.ndarray) -> np.ndarray:
    """Union of all three relations' directed edges into one homogeneous
    edge_index, exact-duplicate directed pairs removed. See module
    docstring for why this matters for baseline fairness."""
    merged = np.concatenate([adj_tdt, adj_tbt, adj_tft], axis=1)
    unique_pairs = np.unique(merged.T, axis=0)
    return unique_pairs.T.astype(np.int64)


class GraphSAGEBaseline(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int, num_classes: int, dropout: float):
        super().__init__()
        self.conv1 = SAGEConv(feature_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, num_classes)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.conv2(h, edge_index)
        return F.log_softmax(h, dim=-1)


def _load_merged_graph(processed_dir: Path = PROCESSED_DIR):
    node_index_map, features, time_steps, labels = load_processed(processed_dir)
    adj_tdt, adj_tbt, adj_tft = load_relations(processed_dir)
    merged_edges = build_merged_edge_index(adj_tdt, adj_tbt, adj_tft)

    x_full = torch.tensor(features, dtype=torch.float32)
    labels_full = torch.tensor(labels, dtype=torch.long)
    edge_index = torch.tensor(merged_edges, dtype=torch.long)

    train_steps = set(CARE_GNN_CONFIG["train_time_steps"])
    test_steps = set(CARE_GNN_CONFIG["test_time_steps"])
    train_mask = torch.tensor([int(s) in train_steps for s in time_steps], dtype=torch.bool)
    test_mask = torch.tensor([int(s) in test_steps for s in time_steps], dtype=torch.bool)

    return x_full, labels_full, edge_index, train_mask, test_mask, merged_edges.shape[1]


def train(
    config: dict = CARE_GNN_CONFIG,
    benchmark_only: bool = False,
    model_name: str = "graphsage",
    model_cls=GraphSAGEBaseline,
    checkpoint_path: Path = CHECKPOINT_PATH,
    model_kwargs: dict | None = None,
    balanced_undersampling: bool = False,
    fraud_chunk_size: int = 1024,
    focal_class_weights: tuple | None = None,
    ablation_variant: str | None = None,
    run_name: str | None = None,
    eval_every_n_epochs: int = EVAL_EVERY_N_EPOCHS,
):
    """Shared full-graph training loop for GraphSAGE and GAT (models.baselines.gat
    imports and calls this with model_cls=GATBaseline) -- the loop itself
    (loss, eval protocol, mlflow logging, checkpointing) is identical between
    the two architectures per the fairness requirement that only the
    conv-layer type differs, not the training procedure.

    Level 9.1 correction -- balanced under-sampling (paper Section 4.1.4),
    matching training/trainer.py's CARE-GNN protocol: when True, each
    optimizer step's classification loss is computed over an equal number of
    fraud (y=1) and licit (y=0) TRAIN-labelled nodes, sampled fresh each
    batch, instead of the whole train split at once. Message passing is
    UNCHANGED -- model(x_full, edge_index) still computes embeddings for
    every node every batch (GraphSAGE/GAT need full-graph structure for
    correct aggregation regardless, and it's cheap here, ~3-5s), only the
    loss is restricted to the sampled balanced subset via indexing. One
    epoch = one full pass through all fraud-labelled train nodes.
    focal_class_weights defaults to (1.0, 1.0) when balanced (see
    training/trainer.py's identical reasoning: FRAUD_CLASS_WEIGHT would
    over-correct an imbalance that no longer exists at the batch level).
    """
    torch.manual_seed(config["seed"])

    x_full, labels_full, edge_index, train_mask, test_mask, merged_edge_count = _load_merged_graph()

    model = model_cls(
        feature_dim=config["feature_dim"],
        hidden_dim=config["hidden_dim"],
        num_classes=config["num_classes"],
        dropout=config["dropout"],
        **(model_kwargs or {}),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    if focal_class_weights is None:
        focal_class_weights = (1.0, 1.0) if balanced_undersampling else (1.0, FRAUD_CLASS_WEIGHT)
    focal_loss_fn = WeightedFocalLoss(class_weights=torch.tensor(list(focal_class_weights)))

    fraud_train_idx = torch.nonzero(train_mask & (labels_full == 1), as_tuple=False).squeeze(-1)
    licit_train_idx = torch.nonzero(train_mask & (labels_full == 0), as_tuple=False).squeeze(-1)
    train_idx = torch.nonzero(train_mask, as_tuple=False).squeeze(-1)

    test_labelled_idx = torch.nonzero(test_mask & (labels_full != -1), as_tuple=False).squeeze(-1)
    test_y_true = labels_full[test_labelled_idx].numpy()

    mlflow.set_tracking_uri(MLRUNS_PATH.as_uri())
    mlflow.set_experiment("Augur-Elliptic")
    if run_name is None:
        run_name = f"{model_name}-benchmark" if benchmark_only else f"{model_name}-baseline"

    epochs = 1 if benchmark_only else config["epochs"]
    best_f1_illicit = -1.0
    best_epoch = None
    epoch_times = []
    final_total_loss = None

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            **{k: v for k, v in config.items() if not isinstance(v, range)},
            "fraud_class_weight": FRAUD_CLASS_WEIGHT,
            "eval_every_n_epochs": eval_every_n_epochs,
            "benchmark_only": benchmark_only,
            "merged_edge_count": merged_edge_count,
            "graph_mode": "merged_tdt_tbt_tft",
            "balanced_undersampling": balanced_undersampling,
            "focal_class_weights": str(focal_class_weights),
            "epochs_trained": epochs,
        })
        mlflow.set_tags({
            "model_type": model_name,
            "dataset": "elliptic",
            "relations_used": "tdt+tbt+tft (merged, deduplicated)",
            "ablation_variant": ablation_variant or f"{model_name}_baseline",
            "protocol": "corrected" if balanced_undersampling else "original",
            "epochs_trained": str(epochs),
        })

        for epoch in range(1, epochs + 1):
            epoch_start = time.perf_counter()
            model.train()

            g = torch.Generator().manual_seed(config["seed"] * 1_000_003 + epoch)

            if balanced_undersampling:
                fraud_perm = fraud_train_idx[torch.randperm(fraud_train_idx.shape[0], generator=g)]
                batch_losses = []
                for i in range(0, fraud_perm.shape[0], fraud_chunk_size):
                    fraud_chunk = fraud_perm[i : i + fraud_chunk_size]
                    n = fraud_chunk.shape[0]
                    licit_pick = licit_train_idx[torch.randperm(licit_train_idx.shape[0], generator=g)[:n]]
                    batch_centers = torch.cat([fraud_chunk, licit_pick])

                    optimizer.zero_grad()
                    out = model(x_full, edge_index)
                    loss = focal_loss_fn(out[batch_centers], labels_full[batch_centers])
                    loss.backward()
                    optimizer.step()
                    batch_losses.append(loss.item())
                loss_value = sum(batch_losses) / len(batch_losses)
            else:
                optimizer.zero_grad()
                out = model(x_full, edge_index)
                gnn_targets_train = torch.where(train_mask, labels_full, torch.full_like(labels_full, -1))
                loss = focal_loss_fn(out, gnn_targets_train)
                loss.backward()
                optimizer.step()
                loss_value = loss.item()

            epoch_elapsed = time.perf_counter() - epoch_start
            epoch_times.append(epoch_elapsed)
            final_total_loss = loss_value

            peak_rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9
            mlflow.log_metrics(
                {"train/loss": loss_value, "epoch_seconds": epoch_elapsed, "peak_rss_gb": peak_rss_gb},
                step=epoch,
            )
            print(f"[{model_name}] epoch {epoch:3d} | loss={loss_value:.4f} | {epoch_elapsed:.2f}s | peak_rss={peak_rss_gb:.2f}GB")

            if epoch % eval_every_n_epochs == 0 or epoch == epochs:
                model.eval()
                with torch.no_grad():
                    out_eval = model(x_full, edge_index)
                test_y_pred_proba = out_eval[test_labelled_idx, 1].exp().numpy()
                metrics = evaluate(test_y_true, test_y_pred_proba)
                mlflow.log_metrics({f"test/{k}": v for k, v in metrics.items()}, step=epoch)
                print(
                    f"  [{model_name} eval @ epoch {epoch}] f1_illicit={metrics['f1_illicit']:.4f} "
                    f"auc_roc={metrics['auc_roc']:.4f} precision={metrics['precision_illicit']:.4f} "
                    f"recall={metrics['recall_illicit']:.4f}"
                )
                if metrics["f1_illicit"] > best_f1_illicit:
                    best_f1_illicit = metrics["f1_illicit"]
                    best_epoch = epoch
                    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(
                        {"epoch": epoch, "model_state_dict": model.state_dict(), "f1_illicit": best_f1_illicit, "config": config},
                        checkpoint_path,
                    )

    return {
        "epoch_times": epoch_times,
        "final_total_loss": final_total_loss,
        "best_f1_illicit": best_f1_illicit,
        "best_epoch": best_epoch,
        "merged_edge_count": merged_edge_count,
    }


if __name__ == "__main__":
    result = train()
    print(result)
