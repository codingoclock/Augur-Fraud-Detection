"""Module 3 (v3): Relation-Aware Neighbor Aggregator (Dou et al. CIKM 2020,
Section 3.4, Eq. 8-9).

Eq. 8 (intra-relation): h_v,r^(l) = ReLU(mean({h_v'^(l-1) : selected neighbours}))
  - parameterless mean aggregator per relation, ReLU applied HERE, per
    relation, not deferred to the end. No learnable weight per relation --
    this differs from v1/v2, which incorrectly gave each relation its own
    Linear layer.

Eq. 9 (inter-relation): h_v^(l) = ReLU(AGG_all(h_v^(l-1) (+) {p_r^(l) * h_v,r^(l)}_r))
  - self term h_v^(l-1) is summed in RAW (no W_self projection before the
    sum -- the projection, if any, lives in AGG_all, applied once, after
    combining)
  - p_r^(l) is used DIRECTLY as the weight, not softmax-normalized. This is
    the one unambiguous correction from v1/v2: "the optimal filtering
    threshold p_r^(l) learned by the RL process [is directly applied] as
    the inter-relation aggregation weights" (paper, Section 3.4).
  - AGG_all is explicitly left flexible by the paper ("can be any type of
    aggregator, and we test them in Section 4.3"). This implementation uses
    a single Linear layer as AGG_all, a defensible, simple choice within
    that stated flexibility -- documented here as a choice, not something
    the paper itself specifies.

Note on the removed W_self/W_per_relation design from v1/v2: those extra
per-relation and self-connection weight matrices are not forbidden by the
paper (AGG_all is explicitly left open), but the specific v1/v2 design --
separate learned projections per relation applied *before* summing, with
softmax-normalized weights -- contradicts the paper's explicit "p_r directly
as the weight" statement and its Eq. 8/9 structure (parameterless
intra-relation mean, single post-combination transform). If ablation results
later suggest the richer v1/v2 parameterization performs better on Elliptic,
that is a legitimate empirical finding -- to be evaluated as a named,
deliberate deviation in the ablation table, not silently substituted back in
as "the" implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RelationAwareAggregator(nn.Module):
    def __init__(self, feature_dim: int, num_relations: int, output_dim: int):
        super().__init__()
        self.num_relations = num_relations
        # AGG_all: the ONLY learnable transform in this module, applied once
        # after combining self + weighted relation terms. Maps feature_dim
        # (dim of the combined, pre-transform vector) to output_dim.
        self.agg_all = nn.Linear(feature_dim, output_dim)

    def forward(
        self,
        x: torch.Tensor,               # [N, feature_dim] center node embeddings, h_v^(l-1)
        selected_neighbors: list,      # list of [N, max_k] bool, per relation
        neighbor_features: list,       # list of [N, max_k, feature_dim], per relation, h_v'^(l-1)
        thresholds: torch.Tensor,      # [num_relations] RAW p_r^(l) values, NOT softmax'd
    ) -> torch.Tensor:
        combined = x  # h_v^(l-1), raw, no projection (the (+) in Eq. 9 starts here)

        for r in range(self.num_relations):
            sel = selected_neighbors[r]           # [N, max_k]
            nf = neighbor_features[r]              # [N, max_k, feature_dim]

            sel_f = nf * sel.unsqueeze(-1).float()
            count = sel.float().sum(dim=-1, keepdim=True).clamp(min=1)
            mean_agg = sel_f.sum(dim=1) / count    # [N, feature_dim]
            h_v_r = F.relu(mean_agg)               # Eq. 8 -- ReLU applied HERE, per relation

            combined = combined + thresholds[r] * h_v_r  # Eq. 9's (+) sum, raw p_r weight

        return F.relu(self.agg_all(combined))       # Eq. 9's outer AGG_all + ReLU
