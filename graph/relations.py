"""Build tdt, tbt, tft adjacency (edge_index) arrays over the Elliptic node set."""

import pickle
import random
from itertools import combinations
from pathlib import Path

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

from config import CARE_GNN_CONFIG

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

SEED = CARE_GNN_CONFIG["seed"]
TBT_CAP_PER_BLOCK = 500
TFT_TOP_K = 10
TFT_SIM_THRESHOLD = 0.85


def build_tdt_edges(edgelist_df, node_index_map: dict) -> np.ndarray:
    """Native directed edges from elliptic_txs_edgelist.csv, remapped through node_index_map."""
    src = edgelist_df["txId1"].to_numpy()
    dst = edgelist_df["txId2"].to_numpy()
    edges = []
    dropped = 0
    for u, v in zip(src, dst):
        u_idx = node_index_map.get(int(u))
        v_idx = node_index_map.get(int(v))
        if u_idx is None or v_idx is None:
            dropped += 1
            continue
        edges.append((u_idx, v_idx))
    if dropped:
        print(f"tdt: dropped {dropped} edges referencing txIds absent from the node index map")
    if not edges:
        return np.empty((2, 0), dtype=np.int64)
    return np.array(edges, dtype=np.int64).T  # [2, E]


def _sample_pairs_for_block(node_ids: np.ndarray, cap: int, rng: random.Random) -> list:
    """Return up to `cap` distinct unordered pairs of node indices from node_ids.

    Uses rejection sampling instead of materialising itertools.combinations for
    large blocks -- some Elliptic time steps have >7000 nodes, meaning tens of
    millions of combinations, which is the actual reason a cap is needed at all.
    """
    n = len(node_ids)
    total_pairs = n * (n - 1) // 2
    if total_pairs <= cap:
        return list(combinations(node_ids.tolist(), 2))

    seen = set()
    pairs = []
    while len(pairs) < cap:
        i, j = rng.sample(range(n), 2)
        key = (i, j) if i < j else (j, i)
        if key in seen:
            continue
        seen.add(key)
        pairs.append((int(node_ids[key[0]]), int(node_ids[key[1]])))
    return pairs


def build_tbt_edges(time_steps: np.ndarray, seed: int = SEED, cap: int = TBT_CAP_PER_BLOCK) -> np.ndarray:
    """Transaction-Block-Transaction: connect node pairs sharing a time_step.

    Capped at `cap` pairs per block (sampled, not truncated) to avoid the
    quadratic blow-up on large blocks. Sampling is seeded off config.py's
    seed so the resulting graph -- and every ablation run trained on it --
    is reproducible run to run. This determinism matters for ablation
    comparability: if tbt's edge set silently changed between runs,
    differences between CARE-GNN vs GraphSAGE vs GAT results could be an
    artifact of graph noise rather than a genuine model difference.
    """
    unique_steps = np.unique(time_steps)
    edges = []
    for step in unique_steps:
        node_ids = np.nonzero(time_steps == step)[0]
        if len(node_ids) < 2:
            continue
        # Per-block RNG seeded off (global seed + step) so re-running with the
        # same seed reproduces the same sample for every block, independent
        # of dict/set iteration order elsewhere in the pipeline.
        rng = random.Random(seed + int(step))
        pairs = _sample_pairs_for_block(node_ids, cap, rng)
        for u, v in pairs:
            edges.append((u, v))
            edges.append((v, u))  # undirected
    if not edges:
        return np.empty((2, 0), dtype=np.int64)
    return np.array(edges, dtype=np.int64).T


def build_tft_edges(features: np.ndarray, k: int = TFT_TOP_K, threshold: float = TFT_SIM_THRESHOLD) -> np.ndarray:
    """Transaction-Feature-Transaction: connect nodes whose L2-normalised
    feature vectors have cosine similarity > threshold, capped at top-k
    neighbours per node.

    NearestNeighbors takes no random_state for either 'ball_tree' or
    'brute' -- both are deterministic, so there is nothing for config.py's
    seed to control here. Seeding is a structural no-op for this relation,
    not an oversight.

    Deviation from the spec's literal algorithm='ball_tree': sklearn's
    ball_tree does not support metric='cosine' at all (VALID_METRICS
    rejects it), and an empirical trial using ball_tree with a converted
    euclidean metric on the full 203,769-node feature matrix was killed
    after running 41 minutes at 97% CPU without finishing -- ball_tree's
    pruning collapses toward brute-force behaviour at ~165 dimensions
    (the curse of dimensionality), so it offers none of its usual
    speed advantage here while adding tree-construction overhead on top.
    algorithm='brute' with metric='cosine' is used instead: it is
    functionally identical (same exact top-k, same threshold semantics)
    and lets sklearn compute cosine distances via chunked, BLAS-backed
    matrix multiplication, which is what actually scales at this size.
    """
    normed = normalize(features)
    nbrs = NearestNeighbors(n_neighbors=k + 1, metric="cosine", algorithm="brute")
    nbrs.fit(normed)
    distances, indices = nbrs.kneighbors(normed)

    n = normed.shape[0]
    cosine_sim = 1.0 - distances
    self_mask = indices == np.arange(n)[:, None]
    non_self_mask = ~self_mask

    # Rank-then-cap, not mask-then-count: when >k+1 feature vectors are
    # tied at ~0 cosine distance (duplicate/near-duplicate transactions),
    # a node's own index can fail to appear among its returned k+1
    # nearest neighbours at all (sklearn's tie-break picks some other k+1
    # of the tied points instead). Relying on self_mask alone then leaves
    # k+1 neighbours, not k, for that node. kneighbors() returns each row
    # sorted by ascending distance, so taking the first k non-self columns
    # in that order is "top-k excluding self" regardless of whether self
    # itself was returned -- this guarantees the k cap unconditionally.
    rank_within_valid = np.cumsum(non_self_mask, axis=1)
    keep_topk = non_self_mask & (rank_within_valid <= k)
    keep = keep_topk & (cosine_sim > threshold)

    row_idx, col_pos = np.nonzero(keep)
    col_idx = indices[row_idx, col_pos]
    if row_idx.size == 0:
        return np.empty((2, 0), dtype=np.int64)
    edge_index = np.stack([row_idx.astype(np.int64), col_idx.astype(np.int64)])
    return edge_index


def save_relations(adj_tdt: np.ndarray, adj_tbt: np.ndarray, adj_tft: np.ndarray, processed_dir: Path = PROCESSED_DIR) -> None:
    processed_dir.mkdir(parents=True, exist_ok=True)
    with open(processed_dir / "adj_tdt.pkl", "wb") as f:
        pickle.dump(adj_tdt, f)
    with open(processed_dir / "adj_tbt.pkl", "wb") as f:
        pickle.dump(adj_tbt, f)
    with open(processed_dir / "adj_tft.pkl", "wb") as f:
        pickle.dump(adj_tft, f)


def load_relations(processed_dir: Path = PROCESSED_DIR):
    with open(processed_dir / "adj_tdt.pkl", "rb") as f:
        adj_tdt = pickle.load(f)
    with open(processed_dir / "adj_tbt.pkl", "rb") as f:
        adj_tbt = pickle.load(f)
    with open(processed_dir / "adj_tft.pkl", "rb") as f:
        adj_tft = pickle.load(f)
    return adj_tdt, adj_tbt, adj_tft


def run(node_index_map: dict, features: np.ndarray, time_steps: np.ndarray, edgelist_df, processed_dir: Path = PROCESSED_DIR, save: bool = True, seed: int = SEED):
    adj_tdt = build_tdt_edges(edgelist_df, node_index_map)
    adj_tbt = build_tbt_edges(time_steps, seed=seed)
    adj_tft = build_tft_edges(features)

    if save:
        save_relations(adj_tdt, adj_tbt, adj_tft, processed_dir)

    return adj_tdt, adj_tbt, adj_tft


if __name__ == "__main__":
    from graph.loader import run as load_run

    loaded = load_run(save=False)
    adj_tdt, adj_tbt, adj_tft = run(
        loaded["node_index_map"], loaded["features"], loaded["time_steps"], loaded["edgelist_df"]
    )
    print(f"tdt edges: {adj_tdt.shape[1]}")
    print(f"tbt edges: {adj_tbt.shape[1]}")
    print(f"tft edges: {adj_tft.shape[1]}")
