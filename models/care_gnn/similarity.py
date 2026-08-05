"""Module 1 (v3): Label-Aware Similarity Measure (Dou et al. CIKM 2020,
Section 3.2, Eq. 2-4).

Per-layer, per-NODE scalar predictor -- NOT a pairwise cosine similarity of
jointly-projected features (that was v1/v2's error, built from a paraphrase
of the paper rather than its actual equations). Pairwise similarity between
two nodes is computed separately, as the L1 distance between their two
independently-predicted scalars.

Critical, paper-stated asymmetry -- do not "fix" this in a later pass: only
layer 1's LabelAwareSimilarity instance ever receives gradient, via the
auxiliary loss (training/loss.py's SimilarityAuxLoss, Eq. 4). Every
subsequent layer's MLP still runs forward (top-p filtering happens at every
layer) but is never trained -- it stays at random initialization for the
entire life of the model. This is explicit in the paper: "we only use the
similarity measure loss at the first layer... since the neighbor filtering
process at the first layer is critical to both GNN and similarity measures
in the following layers." models/care_gnn/care_gnn.py wires this by only
feeding similarities[0]'s output into the auxiliary loss.
"""

import torch
import torch.nn as nn


class LabelAwareSimilarity(nn.Module):
    """
    Per-layer node-label predictor (paper Eq. 2-3).

    forward() takes a single node's embedding and predicts a scalar in
    (-1, 1) via tanh. Similarity between two nodes is NOT computed inside
    this module -- see `pairwise_distance`/`pairwise_similarity` below,
    which operate on this module's already-computed outputs for a center
    and a neighbour. This means similarity never requires knowing either
    node's true label at inference time -- only at training time, for the
    auxiliary loss.
    """

    def __init__(self, feature_dim: int):
        super().__init__()
        self.mlp = nn.Linear(feature_dim, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: [N, D] node embeddings (raw features at layer 1, previous-layer
        # embeddings at layer l>1)
        return torch.tanh(self.mlp(h)).squeeze(-1)  # [N], each value in (-1, 1)


def pairwise_distance(pred_center: torch.Tensor, pred_neighbor: torch.Tensor) -> torch.Tensor:
    """Eq. 2: D(v, v') = |sigma(MLP(h_v)) - sigma(MLP(h_v'))|"""
    return (pred_center - pred_neighbor).abs()


def pairwise_similarity(pred_center: torch.Tensor, pred_neighbor: torch.Tensor) -> torch.Tensor:
    """Eq. 3: S(v, v') = 1 - D(v, v')"""
    return 1.0 - pairwise_distance(pred_center, pred_neighbor)
