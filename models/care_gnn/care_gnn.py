"""Full CARE-GNN model (Dou et al. CIKM 2020) assembling LabelAwareSimilarity,
SimilarityAwareSelector, and RelationAwareAggregator (build spec v3).

Neighbour sampling: hand-rolled CSR gather, not PyG's NeighborLoader/
NeighborSampler. This machinery is UNCHANGED from Level 5's v2-era
implementation -- v3's corrections were to what gets computed from sampled
neighbours (similarity's per-node-not-pairwise signature, the aggregator's
internals), not how neighbours are fetched. See _build_relation_index /
_sample_neighbors below for the original Level 5 reasoning, still valid.

Why hand-rolled: NeighborLoader/NeighborSampler are built to hand a
MessagePassing-style GNN layer a re-indexed, variably-sized k-hop subgraph
per batch -- they don't naturally produce the fixed-width, dense
[N, max_k] index/mask pair that similarity.py, selector.py, and
aggregator.py all expect as their contract. A CSR (indptr/indices) lookup
built once per relation from the cached, seeded `adj_*.pkl` edge_index
arrays gives O(1) neighbour-list slicing per node and total control over
that padding/masking contract.

Sampling policy for nodes with > max_neighbors candidates under a relation:
a per-node, per-relation seeded subsample without replacement, seed derived
from (config.seed, relation_idx, node_id) -- independent of batch order or
composition. Nodes with <= max_neighbors candidates are kept in full and
padded with a dummy index (0) plus mask=False -- no fabricated neighbours.

v3 simplification: LabelAwareSimilarity is now a per-NODE predictor (not a
pairwise function of a center+neighbour pair), so this file no longer needs
node LABELS at all inside forward() -- top-p selection only needs
similarity SCORES, and the aggregator only needs raw features. Labels are
only needed OUTSIDE this class, in the training loop (Level 6), to build
{-1,+1}-signed targets for SimilarityAuxLoss and to filter "fraud-labelled,
train-split" center nodes for the RL distance signal (selector.py's
label_aware_distance). forward()'s signature has dropped `labels_full`
accordingly, matching the v3 spec's own `forward(self, x, adj_lists)`.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import CARE_GNN_CONFIG
from models.care_gnn.aggregator import RelationAwareAggregator
from models.care_gnn.selector import SimilarityAwareSelector
from models.care_gnn.similarity import LabelAwareSimilarity, pairwise_similarity


def _build_relation_index(edge_index: np.ndarray, num_nodes: int):
    """CSR-style (indptr, indices) neighbour lookup over a relation's full
    edge_index [2, E]. indices[indptr[u]:indptr[u+1]] are node u's
    neighbours. edge_index[0] is treated as centre/source, edge_index[1] as
    neighbour/destination -- for tdt this means a node's "neighbours" are
    only the transactions it SENT to (tdt stores directed src->dst pairs and
    Level 2 kept it "as-is", un-reversed); for tbt/tft both directions were
    already added at construction time, so this convention is symmetric for
    those two relations regardless.
    """
    if edge_index.shape[1] == 0:
        return np.zeros(num_nodes + 1, dtype=np.int64), np.empty(0, dtype=np.int64)
    src, dst = edge_index[0], edge_index[1]
    order = np.argsort(src, kind="stable")
    indices = dst[order]
    counts = np.bincount(src[order], minlength=num_nodes)
    indptr = np.zeros(num_nodes + 1, dtype=np.int64)
    np.cumsum(counts, out=indptr[1:])
    return indptr, indices


def build_relation_indices(adj_list, num_nodes: int):
    """Convert a list of raw [2, E] numpy edge_index arrays (as produced by
    graph/relations.py / loaded via graph/relations.load_relations) into CSR
    (indptr, indices) neighbour-lookup tuples, one per relation. Call once
    per graph load, not once per forward() call or per batch."""
    return [_build_relation_index(adj, num_nodes) for adj in adj_list]


def _sample_neighbors(node_ids, indptr, indices, max_k, seed, relation_idx, restrict_to=None):
    """For each id in node_ids, return up to max_k neighbour ids (+ mask).

    restrict_to: optional set of allowed neighbour ids, used to build an
    induced subgraph over a fixed local node set (see CAREGNN.forward()).
    """
    n = len(node_ids)
    out_idx = np.zeros((n, max_k), dtype=np.int64)
    out_mask = np.zeros((n, max_k), dtype=bool)
    for i, node_id in enumerate(node_ids):
        candidates = indices[indptr[node_id]:indptr[node_id + 1]]
        if restrict_to is not None:
            candidates = np.array(
                [c for c in candidates if c in restrict_to], dtype=np.int64
            )
        if len(candidates) > max_k:
            node_seed = (int(seed) * 1_000_003 + relation_idx * 97 + int(node_id)) % (2**31 - 1)
            rng = np.random.RandomState(node_seed)
            candidates = rng.choice(candidates, size=max_k, replace=False)
        k = len(candidates)
        out_idx[i, :k] = candidates
        out_mask[i, :k] = True
    return out_idx, out_mask


class CAREGNN(nn.Module):
    """
    Full model: per layer, one LabelAwareSimilarity, one
    SimilarityAwareSelector, one RelationAwareAggregator (v3: all three are
    now per-layer instances -- selectors being per-layer is new in v3 and is
    literally what the spec now says, not a deviation the way per-layer
    similarities was flagged as one against v2's pseudocode). Only layer 0's
    LabelAwareSimilarity ever receives gradient (via the auxiliary loss,
    computed OUTSIDE this class in training/loss.py) -- every other layer's
    MLP runs forward for filtering but is never trained, per
    similarity.py's module docstring.

    Judgment call (flagged): the v3 spec's forward() pseudocode captures
    `layer1_similarity_pred = center_pred` inside the per-relation loop,
    without pinning down exactly which node index set "center_pred" covers
    at that point (the pseudocode's neighbour-sampling internals are
    themselves left as "..."). This class computes layer 0's per-node
    prediction ONCE for the entire local induced-subgraph node set, then --
    for the value actually returned to the caller -- slices it down to the
    SAME `center_nodes` the caller asked for classification output on (not
    the full local set, which also includes sampled neighbours). This
    matches what Eq. 11's total loss actually needs: layer-1 predictions
    for specific batch nodes, which the training loop further filters to
    fraud-labelled, train-split examples before computing the auxiliary
    loss.

    Judgment call (flagged, carried over from Level 5, still applies): this
    builds a *1-hop* induced local subgraph (center_nodes plus their sampled
    neighbours across all relations, edges outside that set dropped) and
    reuses that same fixed topology at every layer, rather than re-expanding
    the neighbourhood by one extra hop per layer. Level 6's trainer can call
    forward() with center_nodes = the entire node set (transductive
    full-batch, matching how the paper and config.py's own
    "full-graph-per-epoch" deviation note both intend), which has no
    truncation since the local subgraph then equals the whole graph.
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
        num_classes: int,
        num_relations: int,
        num_layers: int = 2,
        max_neighbors: int = 64,
        config: dict = CARE_GNN_CONFIG,
    ):
        super().__init__()
        self.num_relations = num_relations
        self.num_layers = num_layers
        self.max_neighbors = max_neighbors
        self.seed = config["seed"]

        dims = [feature_dim] + [hidden_dim] * (num_layers - 1) + [num_classes]
        self.similarities = nn.ModuleList(
            [LabelAwareSimilarity(dims[i]) for i in range(num_layers)]
        )
        self.selectors = nn.ModuleList([
            SimilarityAwareSelector(
                num_relations,
                init_p=config["selector_init_p"],
                tau=config["tau"],
                terminal_window=config["rl_terminal_window"],
                terminal_threshold=config["rl_terminal_reward_threshold"],
                seed=config["seed"],
            )
            for _ in range(num_layers)
        ])
        self.aggregators = nn.ModuleList([
            RelationAwareAggregator(dims[i], num_relations, dims[i + 1]) for i in range(num_layers)
        ])

    def forward(
        self,
        x_full: torch.Tensor,          # [N_total, feature_dim]
        adj_indices: list,             # list of (indptr, indices) CSR tuples, one per relation
        center_nodes: torch.Tensor,    # [B] node indices to produce predictions for
    ):
        """Returns (log_softmax_output [B, num_classes], layer1_similarity_pred [B])."""
        center_nodes_np = center_nodes.detach().cpu().numpy()

        # Step 1: sample each centre node's neighbours from the FULL graph,
        # purely to discover which extra nodes need to enter the local
        # subgraph.
        discovered = set(center_nodes_np.tolist())
        for r in range(self.num_relations):
            indptr, indices = adj_indices[r]
            idx, mask = _sample_neighbors(
                center_nodes_np, indptr, indices, self.max_neighbors, self.seed, r
            )
            discovered.update(idx[mask].tolist())

        local_ids = np.array(sorted(discovered), dtype=np.int64)
        local_set = set(local_ids.tolist())

        # Step 2: resample neighbours for EVERY node in the local set,
        # restricted to the local set, so every layer's aggregator has a
        # self-consistent induced-subgraph neighbourhood.
        neighbor_idx_local, neighbor_mask_local = [], []
        for r in range(self.num_relations):
            indptr, indices = adj_indices[r]
            idx, mask = _sample_neighbors(
                local_ids, indptr, indices, self.max_neighbors, self.seed, r,
                restrict_to=local_set,
            )
            idx_local = np.searchsorted(local_ids, idx)
            idx_local = np.where(mask, idx_local, 0)
            neighbor_idx_local.append(torch.from_numpy(idx_local).long())
            neighbor_mask_local.append(torch.from_numpy(mask))

        local_ids_t = torch.from_numpy(local_ids).long()
        x_local = x_full[local_ids_t]

        h = x_local
        layer0_pred_local = None
        for layer_i in range(self.num_layers):
            node_pred = self.similarities[layer_i](h)  # [L] -- one scalar per local node
            if layer_i == 0:
                layer0_pred_local = node_pred

            selected_per_rel, features_per_rel = [], []
            for r in range(self.num_relations):
                n_idx = neighbor_idx_local[r]     # [L, K]
                n_mask = neighbor_mask_local[r]    # [L, K]
                K = n_idx.shape[1]

                center_pred_expanded = node_pred.unsqueeze(1).expand(-1, K)  # [L, K]
                neighbor_pred = node_pred[n_idx]                              # [L, K]
                sim_scores = pairwise_similarity(center_pred_expanded, neighbor_pred)  # [L, K]

                selected = self.selectors[layer_i](sim_scores, r, n_mask)
                selected_per_rel.append(selected)
                features_per_rel.append(h[n_idx])

            thresholds = self.selectors[layer_i].p  # raw p_r^(l), not softmax'd (v3)
            h = self.aggregators[layer_i](h, selected_per_rel, features_per_rel, thresholds)

        local_pos = {int(nid): i for i, nid in enumerate(local_ids)}
        center_local_pos = torch.tensor(
            [local_pos[int(c)] for c in center_nodes_np], dtype=torch.long
        )
        out = h[center_local_pos]
        layer1_similarity_pred = layer0_pred_local[center_local_pos]

        return F.log_softmax(out, dim=-1), layer1_similarity_pred
