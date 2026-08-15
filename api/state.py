"""Shared, in-process model/data state, loaded once at API startup.

GraphSAGE always needs a full-graph message-passing pass to score ANY node
(there's no way to score just one node cheaply), so its P(fraud) vector is
precomputed ONCE at startup for every node and served from memory --
subsequent /predict calls are instant lookups, not recomputation.
CARE-GNN's forward() naturally supports a small, explicit center_nodes set
via its local-induced-subgraph mechanism, so it's scored lazily per request
instead (fast for a single node, no reason to precompute 203,769 of them
upfront).
"""

import os
from pathlib import Path

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import numpy as np
import torch

from config import CARE_GNN_CONFIG
from graph.loader import PROCESSED_DIR, load_processed
from graph.relations import load_relations
from models.baselines.graphsage import GraphSAGEBaseline, build_merged_edge_index
from models.care_gnn.care_gnn import CAREGNN, build_relation_indices

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINTS_DIR = PROJECT_ROOT / "models" / "checkpoints"
GRAPHSAGE_CHECKPOINT = CHECKPOINTS_DIR / "graphsage_best.pt"
CARE_GNN_CHECKPOINT = CHECKPOINTS_DIR / "care_gnn_full_best.pt"  # NOT care_gnn_best.pt -- see Level 10 report
ABLATION_RESULTS_CSV = PROJECT_ROOT / "ablation" / "results" / "ablation_results.csv"


class AugurState:
    def __init__(self):
        self.loaded = False

    def load(self):
        if self.loaded:
            return

        node_index_map, features, time_steps, labels = load_processed(PROCESSED_DIR)
        adj_tdt, adj_tbt, adj_tft = load_relations(PROCESSED_DIR)

        self.node_index_map = node_index_map
        self.features = features
        self.labels = labels
        self.time_steps = time_steps
        self.adj_tdt = adj_tdt
        self.num_nodes = features.shape[0]

        self.x_full = torch.tensor(features, dtype=torch.float32)
        merged_edges = build_merged_edge_index(adj_tdt, adj_tbt, adj_tft)
        self.merged_edge_index = torch.tensor(merged_edges, dtype=torch.long)
        self.care_gnn_adj_indices = build_relation_indices([adj_tdt, adj_tbt, adj_tft], self.num_nodes)

        # --- GraphSAGE: load + precompute full P(fraud) vector ---
        gs_ckpt = torch.load(GRAPHSAGE_CHECKPOINT, map_location="cpu", weights_only=False)
        gs_config = gs_ckpt["config"]
        self.graphsage_model = GraphSAGEBaseline(
            feature_dim=gs_config["feature_dim"], hidden_dim=gs_config["hidden_dim"],
            num_classes=gs_config["num_classes"], dropout=gs_config["dropout"],
        )
        self.graphsage_model.load_state_dict(gs_ckpt["model_state_dict"])
        self.graphsage_model.eval()
        with torch.no_grad():
            out = self.graphsage_model(self.x_full, self.merged_edge_index)
        self.graphsage_proba = out[:, 1].exp().numpy()
        self.graphsage_epoch = gs_ckpt["epoch"]
        self.graphsage_f1 = gs_ckpt["f1_illicit"]

        # --- CARE-GNN: load only, scored lazily per request ---
        cg_ckpt = torch.load(CARE_GNN_CHECKPOINT, map_location="cpu", weights_only=False)
        cg_config = cg_ckpt["config"]
        self.care_gnn_model = CAREGNN(
            feature_dim=cg_config["feature_dim"], hidden_dim=cg_config["hidden_dim"],
            num_classes=cg_config["num_classes"], num_relations=cg_config["num_relations"],
            num_layers=cg_config["num_layers"], max_neighbors=64, config=cg_config,
        )
        self.care_gnn_model.load_state_dict(cg_ckpt["model_state_dict"])
        self.care_gnn_model.eval()
        self.care_gnn_epoch = cg_ckpt["epoch"]
        self.care_gnn_f1 = cg_ckpt["f1_illicit"]

        # --- ablation metrics (for the health endpoint) ---
        self.ablation_metrics = self._load_ablation_metrics()

        self.loaded = True

    def _load_ablation_metrics(self) -> dict:
        import csv
        metrics = {}
        if not ABLATION_RESULTS_CSV.exists():
            return metrics
        with open(ABLATION_RESULTS_CSV) as f:
            for row in csv.DictReader(f):
                metrics[row["variant"]] = row
        return metrics

    def graphsage_predict(self, node_id: int) -> float:
        if not (0 <= node_id < self.num_nodes):
            raise ValueError(f"node_id {node_id} out of range [0, {self.num_nodes})")
        return float(self.graphsage_proba[node_id])

    def care_gnn_predict(self, node_id: int) -> float:
        if not (0 <= node_id < self.num_nodes):
            raise ValueError(f"node_id {node_id} out of range [0, {self.num_nodes})")
        center = torch.tensor([node_id], dtype=torch.long)
        with torch.no_grad():
            out, _ = self.care_gnn_model(self.x_full, self.care_gnn_adj_indices, center)
        return float(out[0, 1].exp().item())


STATE = AugurState()
