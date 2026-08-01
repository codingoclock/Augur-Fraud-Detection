"""Raw CSV -> pandas -> node index mapping.

elliptic_txs_features.csv has 167 raw columns: col 0 = txId, col 1 = time_step
(1-49), cols 2-166 (165 columns) = the actual model features. time_step is
metadata used for the train/test split and the tbt relation, not a feature —
computing feature_dim as "raw_columns - 1" gives 166 and silently misaligns
with config.py's CARE_GNN_CONFIG["feature_dim"] = 165.
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from config import CARE_GNN_CONFIG, LABEL_MAP

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

NUM_FEATURE_COLS = CARE_GNN_CONFIG["feature_dim"]  # 165


def load_raw_frames(raw_dir: Path = RAW_DIR):
    """Load the three Elliptic CSVs as-is (no remapping yet)."""
    features_df = pd.read_csv(raw_dir / "elliptic_txs_features.csv", header=None)
    edgelist_df = pd.read_csv(raw_dir / "elliptic_txs_edgelist.csv")
    classes_df = pd.read_csv(raw_dir / "elliptic_txs_classes.csv")
    return features_df, edgelist_df, classes_df


def build_node_index_map(features_df: pd.DataFrame) -> dict:
    """txId -> contiguous integer node index, ordered by row appearance in
    the features file (the canonical, complete node set)."""
    tx_ids = features_df.iloc[:, 0].to_numpy()
    return {int(tx_id): idx for idx, tx_id in enumerate(tx_ids)}


def build_time_steps(features_df: pd.DataFrame, node_index_map: dict) -> np.ndarray:
    """time_step per node index (column 1 of the raw features file)."""
    n = len(node_index_map)
    time_steps = np.empty(n, dtype=np.int64)
    tx_ids = features_df.iloc[:, 0].to_numpy()
    steps = features_df.iloc[:, 1].to_numpy()
    for tx_id, step in zip(tx_ids, steps):
        time_steps[node_index_map[int(tx_id)]] = int(step)
    return time_steps


def build_feature_matrix(features_df: pd.DataFrame, node_index_map: dict) -> np.ndarray:
    """[N, 165] feature matrix from cols 2-166 (0-indexed), ordered by node index."""
    n = len(node_index_map)
    feature_cols = features_df.iloc[:, 2 : 2 + NUM_FEATURE_COLS].to_numpy(dtype=np.float32)
    assert feature_cols.shape[1] == NUM_FEATURE_COLS, (
        f"expected {NUM_FEATURE_COLS} feature columns, got {feature_cols.shape[1]}"
    )
    tx_ids = features_df.iloc[:, 0].to_numpy()
    features = np.empty((n, NUM_FEATURE_COLS), dtype=np.float32)
    for row_i, tx_id in enumerate(tx_ids):
        features[node_index_map[int(tx_id)]] = feature_cols[row_i]
    return features


def build_labels(classes_df: pd.DataFrame, node_index_map: dict, label_map: dict = LABEL_MAP) -> np.ndarray:
    """Map classes.csv 'class' column (1/2/unknown) -> {1, 0, -1} per node index."""
    n = len(node_index_map)
    labels = np.full(n, -1, dtype=np.int64)
    for tx_id, cls in zip(classes_df["txId"].to_numpy(), classes_df["class"].astype(str).to_numpy()):
        idx = node_index_map.get(int(tx_id))
        if idx is None:
            continue
        labels[idx] = label_map[cls]
    return labels


def save_processed(node_index_map: dict, features: np.ndarray, time_steps: np.ndarray, labels: np.ndarray, processed_dir: Path = PROCESSED_DIR) -> None:
    processed_dir.mkdir(parents=True, exist_ok=True)
    with open(processed_dir / "node_index_map.pkl", "wb") as f:
        pickle.dump(node_index_map, f)
    with open(processed_dir / "features.pkl", "wb") as f:
        pickle.dump(features, f)
    with open(processed_dir / "time_steps.pkl", "wb") as f:
        pickle.dump(time_steps, f)
    with open(processed_dir / "labels.pkl", "wb") as f:
        pickle.dump(labels, f)


def load_processed(processed_dir: Path = PROCESSED_DIR):
    with open(processed_dir / "node_index_map.pkl", "rb") as f:
        node_index_map = pickle.load(f)
    with open(processed_dir / "features.pkl", "rb") as f:
        features = pickle.load(f)
    with open(processed_dir / "time_steps.pkl", "rb") as f:
        time_steps = pickle.load(f)
    with open(processed_dir / "labels.pkl", "rb") as f:
        labels = pickle.load(f)
    return node_index_map, features, time_steps, labels


def run(raw_dir: Path = RAW_DIR, processed_dir: Path = PROCESSED_DIR, save: bool = True) -> dict:
    features_df, edgelist_df, classes_df = load_raw_frames(raw_dir)
    node_index_map = build_node_index_map(features_df)
    time_steps = build_time_steps(features_df, node_index_map)
    features = build_feature_matrix(features_df, node_index_map)
    labels = build_labels(classes_df, node_index_map)

    if save:
        save_processed(node_index_map, features, time_steps, labels, processed_dir)

    return {
        "node_index_map": node_index_map,
        "features": features,
        "time_steps": time_steps,
        "labels": labels,
        "edgelist_df": edgelist_df,
    }


if __name__ == "__main__":
    result = run()
    print(f"nodes: {len(result['node_index_map'])}")
    print(f"features shape: {result['features'].shape}")
    print(f"time_steps range: {result['time_steps'].min()}-{result['time_steps'].max()}")
    labels = result["labels"]
    print(f"labels: fraud={int((labels == 1).sum())} licit={int((labels == 0).sum())} unknown={int((labels == -1).sum())}")
