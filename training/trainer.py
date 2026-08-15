"""Mini-batched training loop for CARE-GNN on the Elliptic graph.

Revised from an earlier full-batch (whole-graph-per-epoch) design after that
approach OOM'd: at max_neighbors=64 (the architecture's real receptive
field -- NOT reduced here, per instruction), CAREGNN.forward()'s dense
[N, max_neighbors, feature_dim] neighbor-feature gather needs an estimated
~77GB for a single forward pass over all 203,769 nodes, against this
machine's 8GB of RAM. Confirmed by (a) an actual OOM kill at
max_neighbors=64 and (b) linear extrapolation from a measured 9.59GB peak
at max_neighbors=8 (8x fewer neighbors -> 8x less memory, matching the
9.59 x 8 ~= 77GB estimate). See the benchmark report for the full numbers.

The fix is to bound the size of the LOCAL INDUCED SUBGRAPH each forward
call materializes, not to shrink max_neighbors (that would change the
model's actual receptive field, a real architectural compromise). Since
CAREGNN.forward()'s local subgraph size is driven by len(center_nodes)
(it discovers center_nodes' neighbours, unioned with center_nodes itself,
then re-samples within that induced set), chunking center_nodes into
batches bounds memory per forward call without touching max_neighbors or
any of the frozen model files.

RL reward accumulation (Algorithm 1's structure): the paper's RL update is
an EPOCH-level step, run once per relation per layer after all of the
epoch's (mini-batch, in this revision) forward/backward work is done --
not once per batch. Each batch's fraud-labelled centers (if any) contribute
a running (distance-weighted-sum, pair-count) to an epoch-level
accumulator; label_aware_distance's own per-call ratio is not used for
this accumulation, since averaging per-batch ratios directly would bias
the result toward whichever batches happened to have fewer fraud-center/
neighbour pairs. rl_step_relation is called exactly once per relation per
layer, after the full batch loop, using the accumulated epoch-wide ratio.
"""

import os
import resource
import time
from pathlib import Path

# MLflow 3.x puts the plain filesystem tracking store into "maintenance
# mode" and raises on it unless this is set -- must happen BEFORE `import
# mlflow` touches tracking-store resolution. This is almost certainly why
# the environment's unconfigured default silently fell back to a SQLite
# store in the first place (see MLRUNS_PATH/set_tracking_uri below).
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import mlflow
import numpy as np
import torch
import torch.nn.functional as F

from config import CARE_GNN_CONFIG, FRAUD_CLASS_WEIGHT
from graph.dataset import AugurDataset
from models.care_gnn.care_gnn import CAREGNN, _sample_neighbors, build_relation_indices
from models.care_gnn.similarity import pairwise_similarity
from training.evaluator import evaluate
from training.loss import SimilarityAuxLoss, WeightedFocalLoss, care_gnn_total_loss

RELATION_NAMES = ["tdt", "tbt", "tft"]
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINTS_DIR = PROJECT_ROOT / "models" / "checkpoints"
MLRUNS_PATH = PROJECT_ROOT / "mlruns"
EVAL_EVERY_N_EPOCHS = 5  # see module docstring / final report for the reasoning
BATCH_SIZE = 2048  # see final report for the benchmark that landed on this value


def _chunk(t: torch.Tensor, batch_size: int) -> list:
    return [t[i : i + batch_size] for i in range(0, t.shape[0], batch_size)]


@torch.no_grad()
def _local_diagnostics_forward(model: CAREGNN, x_full: torch.Tensor, adj_indices: list, center_nodes: torch.Tensor):
    """Read-only replica of CAREGNN.forward()'s two-step local-subgraph
    construction (discover center_nodes' neighbours -> resample restricted
    to that induced set, per layer), generalized to an ARBITRARY, small
    center_nodes set rather than the whole graph. This is what makes RL
    diagnostics mini-batch-safe: called with center_nodes = just the
    fraud-labelled centers inside one training batch, the induced subgraph
    is bounded by that handful of centers plus their neighbours, not by
    the full 203,769-node graph.

    similarity.py/selector.py/aggregator.py/care_gnn.py are frozen --
    correct and tested as of the v3 corrective pass, not modified here.
    CAREGNN.forward() does not expose selected_neighbors/neighbor_idx
    externally, so this function mirrors its internals (using the model's
    own public submodules and care_gnn.py's importable helpers) purely to
    surface what the RL step needs.

    Returns (all LOCAL-index-space, aligned with the returned local_ids --
    NOT global node ids):
        layer0_pred_local: [L] similarities[0]'s tanh output
        selected_per_layer: [num_layers][num_relations] -> [L, max_k] bool
        neighbor_idx_local_per_layer: [num_layers][num_relations] -> [L, max_k] long
        local_ids: [L] sorted global node ids this subgraph covers
    """
    center_nodes_np = center_nodes.detach().cpu().numpy()

    discovered = set(center_nodes_np.tolist())
    for r in range(model.num_relations):
        indptr, indices = adj_indices[r]
        idx, mask = _sample_neighbors(center_nodes_np, indptr, indices, model.max_neighbors, model.seed, r)
        discovered.update(idx[mask].tolist())

    local_ids = np.array(sorted(discovered), dtype=np.int64)
    local_set = set(local_ids.tolist())

    neighbor_idx_local, neighbor_mask_local = [], []
    for r in range(model.num_relations):
        indptr, indices = adj_indices[r]
        idx, mask = _sample_neighbors(
            local_ids, indptr, indices, model.max_neighbors, model.seed, r, restrict_to=local_set
        )
        idx_local = np.searchsorted(local_ids, idx)
        idx_local = np.where(mask, idx_local, 0)
        neighbor_idx_local.append(torch.from_numpy(idx_local).long())
        neighbor_mask_local.append(torch.from_numpy(mask))

    local_ids_t = torch.from_numpy(local_ids).long()
    h = x_full[local_ids_t]

    layer0_pred_local = None
    selected_per_layer, neighbor_idx_local_per_layer = [], []

    for layer_i in range(model.num_layers):
        node_pred = model.similarities[layer_i](h)
        if layer_i == 0:
            layer0_pred_local = node_pred

        selected_per_rel, features_per_rel = [], []
        for r in range(model.num_relations):
            n_idx = neighbor_idx_local[r]
            n_mask = neighbor_mask_local[r]
            K = n_idx.shape[1]

            center_pred_expanded = node_pred.unsqueeze(1).expand(-1, K)
            neighbor_pred = node_pred[n_idx]
            sim_scores = pairwise_similarity(center_pred_expanded, neighbor_pred)

            selected = model.selectors[layer_i](sim_scores, r, n_mask)
            selected_per_rel.append(selected)
            features_per_rel.append(h[n_idx])

        selected_per_layer.append(selected_per_rel)
        neighbor_idx_local_per_layer.append(list(neighbor_idx_local))

        thresholds = model.selectors[layer_i].p
        h = model.aggregators[layer_i](h, selected_per_rel, features_per_rel, thresholds)

    return layer0_pred_local, selected_per_layer, neighbor_idx_local_per_layer, local_ids


def _distance_sum_and_count(layer_pred_local, fraud_local_pos, selected_neighbors, neighbor_idx):
    """Same Eq. 2/5 arithmetic as models/care_gnn/selector.py's
    label_aware_distance, but returns the raw (weighted_sum, count)
    accumulator pair instead of a pre-divided ratio -- needed so an
    epoch's per-batch contributions can be combined into a single correct
    epoch-level average (sum-of-sums / sum-of-counts), rather than
    averaging several already-divided per-batch ratios, which would
    silently bias the result toward batches with fewer fraud-center/
    neighbour pairs. Does not modify or call into selector.py; it
    duplicates its (tiny, two-line) formula rather than repurposing a
    function whose contract is "return the final ratio."
    """
    if fraud_local_pos.numel() == 0:
        return 0.0, 0
    center_pred = layer_pred_local[fraud_local_pos].unsqueeze(1)
    neighbor_pred = layer_pred_local[neighbor_idx]
    diff = (center_pred - neighbor_pred).abs()
    mask = selected_neighbors
    return (diff * mask).sum().item(), int(mask.sum().item())


def train(
    config: dict = CARE_GNN_CONFIG,
    *,
    checkpoint_path: Path,
    benchmark_only: bool = False,
    max_neighbors: int = 64,
    batch_size: int = BATCH_SIZE,
    relation_names: list | None = None,
    enable_rl: bool = True,
    ablation_variant: str = "full",
    run_name: str | None = None,
    balanced_undersampling: bool = False,
    fraud_chunk_size: int = 1024,
    focal_class_weights: tuple | None = None,
    eval_every_n_epochs: int = EVAL_EVERY_N_EPOCHS,
):
    """Ablation hooks (Level 8), all defaulting to the original full-model
    behaviour so the main run is unaffected:
      - relation_names: subset of ["tdt","tbt","tft"] to use; None = all
        three. num_relations is derived from len(relation_names), not read
        from config, so e.g. ["tdt"] alone gives a genuine 1-relation model
        (aggregator.py/care_gnn.py don't hardcode num_relations=3 anywhere,
        verified by their own tests).
      - enable_rl: if False, the RL step (Algorithm 1's epoch-level update)
        is never called at all -- selector.p stays at config's
        selector_init_p (0.5) for the entire run, exactly the "w/o RL
        selector, fixed p=0.5" ablation variant. The per-batch diagnostics
        accumulation is also skipped when disabled, since it would be
        wasted, unused compute otherwise.
      - lambda1=0 (the "w/o label-aware similarity" variant, per v3's
        redefinition -- see ablation/run_ablation.py) needs no dedicated
        parameter here: it's already a plain config value, and
        care_gnn_total_loss's `lambda1 * l_simi` term makes the aux loss's
        gradient contribution exactly zero when lambda1=0, which is exactly
        "the layer-1 MLP never receives gradient."

    Level 9 correction -- balanced under-sampling (paper Section 4.1.4):
      - balanced_undersampling: if True, replaces the natural-imbalance
        mini-batch construction with the paper's actual protocol -- each
        batch's classification centers are an equal number of fraud (y=1)
        and licit (y=0) TRAIN-labelled nodes, sampled fresh each batch. Graph
        structure/neighbor sampling is UNCHANGED: every node, labelled or
        not, remains eligible as a neighbor during aggregation -- only which
        nodes are chosen as classification centers is constrained. One epoch
        = one full pass through all fraud-labelled train nodes (each used
        exactly once, chunked into `fraud_chunk_size` pieces), paired each
        time with a freshly-drawn random licit sample of the same size
        (without replacement within a batch, but licit nodes CAN repeat
        across different batches/epochs since the licit pool (26,432) is
        resampled from fresh each time, not partitioned).
      - focal_class_weights: with batches balanced by construction,
        FRAUD_CLASS_WEIGHT's original purpose (correcting for the dataset's
        natural ~9:1 imbalance) no longer applies within a batch -- keeping
        it would OVER-correct an imbalance that no longer exists at the
        batch level. Defaults to (1.0, 1.0) when balanced_undersampling=True
        and this is left None; explicit values always override.
    """
    torch.manual_seed(config["seed"])

    dataset = AugurDataset()
    data = dataset.data
    x_full = data["transaction"].x
    labels_full = data["transaction"].y  # {1, 0, -1}
    train_mask = dataset.train_mask
    test_mask = dataset.test_mask
    num_nodes = x_full.shape[0]

    relation_names = relation_names or list(RELATION_NAMES)
    all_adj = {
        "tdt": data["transaction", "tdt", "transaction"].edge_index.numpy(),
        "tbt": data["transaction", "tbt", "transaction"].edge_index.numpy(),
        "tft": data["transaction", "tft", "transaction"].edge_index.numpy(),
    }
    selected_adj = [all_adj[name] for name in relation_names]
    adj_indices = build_relation_indices(selected_adj, num_nodes)
    num_relations = len(relation_names)

    # No implicit default (Level 10 fix): checkpoint_path is a required,
    # keyword-only argument -- the previous silent fallback to a shared
    # "care_gnn_best.pt" is exactly what let the no_rl_selector /
    # no_label_similarity / single_relation_tdt / full ablation variants
    # overwrite each other's checkpoints across Level 8/9's runs. Every
    # caller must now name its own file; there is no path left where a run
    # can accidentally collide with another's by omission.

    model = CAREGNN(
        feature_dim=config["feature_dim"],
        hidden_dim=config["hidden_dim"],
        num_classes=config["num_classes"],
        num_relations=num_relations,
        num_layers=config["num_layers"],
        max_neighbors=max_neighbors,
        config=config,
    )
    # model.selectors[*].p is a registered buffer, not an nn.Parameter (by
    # design, see selector.py) -- model.parameters() already excludes it,
    # so Adam cannot touch it; no manual filtering needed here.
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])

    if focal_class_weights is None:
        focal_class_weights = (1.0, 1.0) if balanced_undersampling else (1.0, FRAUD_CLASS_WEIGHT)
    focal_loss_fn = WeightedFocalLoss(class_weights=torch.tensor(list(focal_class_weights)))
    aux_loss_fn = SimilarityAuxLoss()
    lambda1 = config["lambda1"]

    train_idx = torch.nonzero(train_mask, as_tuple=False).squeeze(-1)

    # Fraud-only RL center set: TRAIN-SPLIT AND y == 1 (fraud-labelled).
    # Deliberately NOT the "all labelled" mask used for the main loss --
    # licit-labelled (y=0) and unlabelled (-1) train nodes must be excluded
    # here even though they contribute to the classification loss above.
    fraud_center_idx = torch.nonzero(train_mask & (labels_full == 1), as_tuple=False).squeeze(-1)
    fraud_center_set = set(fraud_center_idx.tolist())
    licit_train_idx = torch.nonzero(train_mask & (labels_full == 0), as_tuple=False).squeeze(-1)
    print(
        f"RL fraud-only center set: {fraud_center_idx.shape[0]} nodes "
        f"(train-split AND y==1; distinct from the "
        f"{int((train_mask & (labels_full != -1)).sum())}-node all-labelled-train mask used for the main loss)"
    )
    if balanced_undersampling:
        print(
            f"Balanced under-sampling ON: {fraud_center_idx.shape[0]} fraud / "
            f"{licit_train_idx.shape[0]} licit available in train split; "
            f"focal_class_weights={focal_class_weights} (FRAUD_CLASS_WEIGHT not applied -- "
            f"batches are balanced by construction)"
        )

    # Test-split eval targets, fixed once (labelled nodes only).
    test_labelled_idx = torch.nonzero(test_mask & (labels_full != -1), as_tuple=False).squeeze(-1)
    test_y_true = labels_full[test_labelled_idx].numpy()

    # Explicit local file-based tracking store, not left to whatever default
    # this environment resolves. Root-caused mid Level 6: mlflow 3.x raises
    # on the plain file store unless MLFLOW_ALLOW_FILE_STORE=true (set above,
    # before mlflow's own import), so the unconfigured default was silently
    # falling back to a SQLite store -- whose own URI construction, given
    # this project directory's literal space in its name, produced a bogus
    # percent-encoded sibling path ("Augur-CAREGNN-Fraud%20System") that a
    # naive f-string never would have hit. That SQLite fallback is the
    # leading suspect for the 10-epoch canary's first attempt taking 106+
    # CPU-minutes without completing even one epoch, versus ~915s measured
    # for an identical single epoch moments earlier: a plain, code-identical
    # replay of the same batch loop WITHOUT any mlflow calls ran at the
    # expected ~7s/batch. Path.as_uri() (not an f-string) used here since it
    # percent-encodes correctly and is what exposed the mismatch above.
    mlflow.set_tracking_uri(MLRUNS_PATH.as_uri())
    mlflow.set_experiment("Augur-Elliptic")
    if run_name is None:
        if ablation_variant == "full":
            # Unchanged from the original full-run naming, preserved exactly
            # so any earlier tooling/queries keyed on this literal string
            # keep working.
            run_name = "CARE-GNN-v3-single-epoch-benchmark" if benchmark_only else "CARE-GNN-full-v3-minibatch"
        else:
            run_name = f"CARE-GNN-{ablation_variant}-benchmark" if benchmark_only else f"CARE-GNN-{ablation_variant}-v3"

    terminated_logged = {(l, r): False for l in range(model.num_layers) for r in range(model.num_relations)}
    best_f1_illicit = -1.0
    best_epoch = None

    epochs = 1 if benchmark_only else config["epochs"]

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            **{k: v for k, v in config.items() if not isinstance(v, range)},
            "train_time_steps": "1-34",
            "test_time_steps": "35-49",
            "fraud_class_weight": FRAUD_CLASS_WEIGHT,
            "eval_every_n_epochs": eval_every_n_epochs,
            "benchmark_only": benchmark_only,
            "batch_size": batch_size,
            "max_neighbors": max_neighbors,
            "epochs_trained": epochs,
        })
        mlflow.set_tags({
            "model_type": "care_gnn",
            "dataset": "elliptic",
            "relations_used": "+".join(relation_names),
            "ablation_variant": ablation_variant,
            "enable_rl": str(enable_rl),
            "lambda1": str(lambda1),
            "epochs_trained": str(epochs),
        })

        epoch_times = []
        final_total_loss = None
        global_batch_step = 0

        for epoch in range(1, epochs + 1):
            epoch_start = time.perf_counter()

            g = torch.Generator().manual_seed(config["seed"] * 1_000_003 + epoch)

            if balanced_undersampling:
                # One epoch = one full pass through ALL fraud-labelled train
                # nodes (each used exactly once), chunked into
                # fraud_chunk_size pieces; each chunk paired with a FRESH
                # random licit sample of the same size (without replacement
                # within the batch, but the licit pool itself is resampled
                # from scratch every batch -- licit nodes can and will repeat
                # across batches/epochs, unlike fraud). Neighbor sampling and
                # graph structure are unaffected -- this only constrains
                # which nodes are chosen as classification CENTERS.
                fraud_perm = fraud_center_idx[torch.randperm(fraud_center_idx.shape[0], generator=g)]
                batches = []
                for i in range(0, fraud_perm.shape[0], fraud_chunk_size):
                    fraud_chunk = fraud_perm[i : i + fraud_chunk_size]
                    n = fraud_chunk.shape[0]
                    licit_pick = licit_train_idx[torch.randperm(licit_train_idx.shape[0], generator=g)[:n]]
                    batches.append(torch.cat([fraud_chunk, licit_pick]))
            else:
                # Shuffle train_idx once per epoch, seeded off (config seed,
                # epoch) for determinism -- standard mini-batch practice, not
                # required by the spec but costs nothing and avoids a fixed
                # batch composition every epoch.
                perm = torch.randperm(train_idx.shape[0], generator=g)
                epoch_train_idx = train_idx[perm]
                batches = _chunk(epoch_train_idx, batch_size)

            # Epoch-level RL accumulators: (weighted_sum, count) per (layer, relation).
            dist_sum = [[0.0] * model.num_relations for _ in range(model.num_layers)]
            dist_count = [[0] * model.num_relations for _ in range(model.num_layers)]

            model.train()
            batch_losses, batch_l_gnn, batch_l_simi = [], [], []

            for batch_centers in batches:
                # --- 1-3: forward, loss, backward, optimizer step (this batch) ---
                optimizer.zero_grad()
                out, layer1_pred = model(x_full, adj_indices, batch_centers)

                gnn_targets = labels_full[batch_centers]  # batch_centers is train-split only, by construction
                labelled_mask = gnn_targets != -1
                y_signed = torch.where(gnn_targets == 1, torch.tensor(1.0), torch.tensor(-1.0))

                loss, breakdown = care_gnn_total_loss(
                    out, gnn_targets, layer1_pred, y_signed, labelled_mask,
                    focal_loss_fn, aux_loss_fn, lambda1,
                )
                loss.backward()
                optimizer.step()

                batch_losses.append(loss.item())
                batch_l_gnn.append(breakdown["l_gnn"])
                batch_l_simi.append(breakdown["l_simi"])

                # Logged every 10th batch, not every batch: profiling (Level 6
                # revision) showed synchronous per-batch mlflow.log_metrics
                # calls costing ~0.5s on average with spikes to ~1.4s (disk
                # flush on the tracking store), a real ~10% overhead on top
                # of the ~4.7s/batch forward+backward cost. Batch-level loss
                # is a diagnostic nicety, not the primary logged record
                # (epoch-level metrics below are) -- coarser granularity here
                # loses little.
                if global_batch_step % 10 == 0:
                    mlflow.log_metrics({"train/batch_loss": loss.item()}, step=global_batch_step)
                global_batch_step += 1

                # --- 4: RL diagnostics, AFTER this batch's backward + optimizer.step() ---
                # Skipped entirely when enable_rl=False ("w/o RL selector"
                # ablation variant): this accumulation only ever feeds the
                # RL step below, so computing it would be pure waste when
                # that step never runs.
                if not enable_rl:
                    continue

                fraud_in_batch = batch_centers[
                    torch.tensor([int(c) in fraud_center_set for c in batch_centers.tolist()], dtype=torch.bool)
                ]
                if fraud_in_batch.numel() == 0:
                    continue  # nothing to accumulate from this batch

                layer0_pred_local, selected_per_layer, neighbor_idx_local_per_layer, local_ids = (
                    _local_diagnostics_forward(model, x_full, adj_indices, fraud_in_batch)
                )
                local_pos = {int(nid): i for i, nid in enumerate(local_ids)}
                fraud_local_pos = torch.tensor(
                    [local_pos[int(c)] for c in fraud_in_batch.tolist()], dtype=torch.long
                )

                for layer_i in range(model.num_layers):
                    for r in range(model.num_relations):
                        s, c = _distance_sum_and_count(
                            layer0_pred_local,
                            fraud_local_pos,
                            selected_per_layer[layer_i][r][fraud_local_pos],
                            neighbor_idx_local_per_layer[layer_i][r][fraud_local_pos],
                        )
                        dist_sum[layer_i][r] += s
                        dist_count[layer_i][r] += c

            # --- RL step: ONCE per relation per layer, after the full batch loop ---
            # Never called at all when enable_rl=False -- selector.p is left
            # exactly at config["selector_init_p"] (0.5) for the entire run,
            # which IS the "w/o RL selector, fixed p=0.5" ablation variant.
            if enable_rl:
                for layer_i in range(model.num_layers):
                    for r in range(model.num_relations):
                        epoch_distance = dist_sum[layer_i][r] / dist_count[layer_i][r] if dist_count[layer_i][r] > 0 else 0.0
                        model.selectors[layer_i].rl_step_relation(r, epoch_distance, epoch)

            epoch_elapsed = time.perf_counter() - epoch_start
            epoch_times.append(epoch_elapsed)

            mean_loss = sum(batch_losses) / len(batch_losses)
            mean_l_gnn = sum(batch_l_gnn) / len(batch_l_gnn)
            mean_l_simi = sum(batch_l_simi) / len(batch_l_simi)
            final_total_loss = mean_loss

            # macOS reports ru_maxrss in bytes (Linux: KB) -- this process
            # only runs on macOS per the environment this was built in.
            # ru_maxrss is a running peak for the whole process, not a
            # per-epoch snapshot -- tracked epoch to epoch specifically to
            # catch a leak (steady climb well beyond one epoch's own
            # working set) before it turns a long run into an OOM crash.
            peak_rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9

            # --- 5: MLflow logging (epoch-level, primary record) ---
            log_dict = {
                "train/total_loss": mean_loss,
                "train/l_gnn": mean_l_gnn,
                "train/l_simi": mean_l_simi,
                "epoch_seconds": epoch_elapsed,
                "num_batches": len(batches),
                "peak_rss_gb": peak_rss_gb,
            }
            for layer_i in range(model.num_layers):
                for r in range(model.num_relations):
                    rel_name = relation_names[r]
                    p_val = float(model.selectors[layer_i].p[r].item())
                    state = model.selectors[layer_i].get_state(r)
                    log_dict[f"selector/p_layer{layer_i}_{rel_name}"] = p_val
                    log_dict[f"selector/is_terminated_layer{layer_i}_{rel_name}"] = int(state.terminated)
                    if state.terminated and not terminated_logged[(layer_i, r)]:
                        log_dict[f"selector/terminated_epoch_layer{layer_i}_{rel_name}"] = epoch
                        terminated_logged[(layer_i, r)] = True
            mlflow.log_metrics(log_dict, step=epoch)

            p_snapshot = " ".join(
                f"L{layer_i}_{relation_names[r]}={float(model.selectors[layer_i].p[r].item()):.3f}"
                for layer_i in range(model.num_layers)
                for r in range(model.num_relations)
            )
            print(
                f"epoch {epoch:3d} | total_loss={mean_loss:.4f} "
                f"l_gnn={mean_l_gnn:.4f} l_simi={mean_l_simi:.4f} "
                f"| {len(batches)} batches | {epoch_elapsed:.2f}s | peak_rss={peak_rss_gb:.2f}GB"
            )
            print(f"         p: {p_snapshot}")

            # --- 6: periodic test-split eval, mini-batched over test_labelled_idx ---
            if epoch % eval_every_n_epochs == 0 or epoch == epochs:
                model.eval()
                pred_proba_chunks = []
                with torch.no_grad():
                    for eval_batch in _chunk(test_labelled_idx, batch_size):
                        out_eval, _ = model(x_full, adj_indices, eval_batch)
                        pred_proba_chunks.append(out_eval[:, 1].exp())
                test_y_pred_proba = torch.cat(pred_proba_chunks).numpy()

                metrics = evaluate(test_y_true, test_y_pred_proba)
                mlflow.log_metrics({f"test/{k}": v for k, v in metrics.items()}, step=epoch)
                print(
                    f"  [eval @ epoch {epoch}] f1_illicit={metrics['f1_illicit']:.4f} "
                    f"auc_roc={metrics['auc_roc']:.4f} precision={metrics['precision_illicit']:.4f} "
                    f"recall={metrics['recall_illicit']:.4f}"
                )

                if metrics["f1_illicit"] > best_f1_illicit:
                    best_f1_illicit = metrics["f1_illicit"]
                    best_epoch = epoch
                    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": model.state_dict(),
                            "f1_illicit": best_f1_illicit,
                            "config": config,
                        },
                        checkpoint_path,
                    )

        selector_final_state = {
            f"layer{layer_i}_{relation_names[r]}": {
                "p": float(model.selectors[layer_i].p[r].item()),
                "terminated": model.selectors[layer_i].get_state(r).terminated,
            }
            for layer_i in range(model.num_layers)
            for r in range(model.num_relations)
        }

    return {
        "epoch_times": epoch_times,
        "final_total_loss": final_total_loss,
        "best_f1_illicit": best_f1_illicit,
        "best_epoch": best_epoch,
        "selector_final_state": selector_final_state,
        "num_fraud_centers": int(fraud_center_idx.shape[0]),
    }


if __name__ == "__main__":
    result = train(checkpoint_path=CHECKPOINTS_DIR / "care_gnn_full_best.pt")
    print(result)
