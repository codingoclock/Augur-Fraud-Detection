import numpy as np
import pytest

from config import CARE_GNN_CONFIG
from graph.loader import run as load_run
from graph.relations import (
    TBT_CAP_PER_BLOCK,
    TFT_TOP_K,
    build_tbt_edges,
    build_tdt_edges,
    build_tft_edges,
)

SEED = CARE_GNN_CONFIG["seed"]


@pytest.fixture(scope="module")
def loaded():
    return load_run(save=False)


@pytest.fixture(scope="module")
def relations(loaded):
    adj_tdt = build_tdt_edges(loaded["edgelist_df"], loaded["node_index_map"])
    adj_tbt = build_tbt_edges(loaded["time_steps"], seed=SEED)
    adj_tft = build_tft_edges(loaded["features"])
    return adj_tdt, adj_tbt, adj_tft


def test_node_index_map_is_bijection(loaded):
    node_index_map = loaded["node_index_map"]
    n = len(node_index_map)
    assert n == 203769
    values = sorted(node_index_map.values())
    # bijection with the raw txId set: contiguous 0..N-1, no gaps, no
    # collisions (a collision would shrink the value set below N).
    assert values == list(range(n))
    assert len(set(node_index_map.values())) == n


def test_tdt_edge_count_matches_raw(loaded, relations):
    adj_tdt, _, _ = relations
    raw_edge_count = len(loaded["edgelist_df"])
    assert raw_edge_count == 234355
    # Verified separately against the raw CSVs: zero self-loops, zero
    # duplicate directed edges, and every edgelist txId present in
    # features.csv. With no legitimate reason to drop or collapse an
    # edge, remapping should preserve the row count exactly.
    assert adj_tdt.shape[1] == raw_edge_count


def test_tbt_never_exceeds_cap_per_block(loaded, relations):
    _, adj_tbt, _ = relations
    time_steps = loaded["time_steps"]

    src, dst = adj_tbt[0], adj_tbt[1]
    src_steps = time_steps[src]
    dst_steps = time_steps[dst]
    assert np.array_equal(src_steps, dst_steps), "tbt edges must connect nodes within the same time_step"

    for step in np.unique(time_steps):
        block_edge_count = int((src_steps == step).sum())
        pair_count = block_edge_count // 2  # each undirected pair -> 2 directed edges
        assert pair_count <= TBT_CAP_PER_BLOCK, (
            f"time_step {step} has {pair_count} pairs, exceeds cap {TBT_CAP_PER_BLOCK}"
        )


def test_tft_respects_top_k_cap(relations):
    _, _, adj_tft = relations
    src = adj_tft[0]
    _, counts = np.unique(src, return_counts=True)
    assert counts.max() <= TFT_TOP_K, f"a node has {counts.max()} tft neighbours, exceeds cap {TFT_TOP_K}"


def test_feature_matrix_shape(loaded):
    assert loaded["features"].shape == (203769, 165)


def test_relations_reproducibility_same_seed(loaded, relations):
    """Running relations construction twice with the same config.py seed
    must produce identical edge sets -- required for ablation runs to be
    comparable across repeated executions."""
    adj_tdt_1, adj_tbt_1, adj_tft_1 = relations
    adj_tdt_2 = build_tdt_edges(loaded["edgelist_df"], loaded["node_index_map"])
    adj_tbt_2 = build_tbt_edges(loaded["time_steps"], seed=SEED)
    adj_tft_2 = build_tft_edges(loaded["features"])

    assert np.array_equal(adj_tdt_1, adj_tdt_2)
    assert np.array_equal(adj_tbt_1, adj_tbt_2)
    assert np.array_equal(adj_tft_1, adj_tft_2)
