"""GAT baseline (2-layer), per Augur build spec (baselines section unchanged
v1/v2 -> v3).

Same fairness decision as graphsage.py (merged tdt+tbt+tft graph, 1,979,669
deduplicated directed edges) and the same training loop, loss
(WeightedFocalLoss + FRAUD_CLASS_WEIGHT), evaluation protocol
(training/evaluator.py's evaluate(), F1_illicit primary), and MLflow
experiment/logging convention -- see graphsage.py's module docstring for the
full reasoning, not repeated here. train() is imported and reused as-is;
only the conv-layer architecture differs.

Attention heads: not specified by the spec ("GAT (2-layer)" only). 4 heads
on the hidden layer (standard GAT-paper pattern: multi-head concat in the
hidden layer, single head no-concat at the output layer) is a defensible,
simple choice within that gap -- documented here as a choice, not something
the spec pins down, same spirit as aggregator.py's AGG_all choice.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

from config import CARE_GNN_CONFIG
from models.baselines.graphsage import CHECKPOINT_PATH as _GRAPHSAGE_CHECKPOINT_PATH
from models.baselines.graphsage import train as _shared_train

CHECKPOINT_PATH = _GRAPHSAGE_CHECKPOINT_PATH.parent / "gat_best.pt"

GAT_HEADS = 4


class GATBaseline(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int, num_classes: int, dropout: float, heads: int = GAT_HEADS):
        super().__init__()
        assert hidden_dim % heads == 0, "hidden_dim must be divisible by heads for concat output to equal hidden_dim"
        self.conv1 = GATConv(feature_dim, hidden_dim // heads, heads=heads, concat=True, dropout=dropout)
        self.conv2 = GATConv(hidden_dim, num_classes, heads=1, concat=False, dropout=dropout)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = F.dropout(x, p=self.dropout, training=self.training)
        h = self.conv1(h, edge_index)
        h = F.elu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.conv2(h, edge_index)
        return F.log_softmax(h, dim=-1)


def train(config: dict = CARE_GNN_CONFIG, benchmark_only: bool = False):
    return _shared_train(
        config=config,
        benchmark_only=benchmark_only,
        model_name="gat",
        model_cls=GATBaseline,
        checkpoint_path=CHECKPOINT_PATH,
        model_kwargs={"heads": GAT_HEADS},
    )


if __name__ == "__main__":
    result = train()
    print(result)
