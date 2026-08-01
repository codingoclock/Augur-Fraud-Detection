"""AugurDataset: wraps loader + relations output into a PyG HeteroData graph."""

from pathlib import Path

import torch
from torch_geometric.data import HeteroData

from config import CARE_GNN_CONFIG
from graph.loader import PROCESSED_DIR, load_processed
from graph.relations import load_relations

NODE_TYPE = "transaction"


class AugurDataset:
    """Single-node-type, three-edge-type PyG HeteroData wrapper over the
    processed Elliptic graph.

    Train/test split is exposed as boolean masks over time_step (steps
    1-34 train, 35-49 test), not by physically partitioning the node set:
    CARE-GNN needs the full graph structure (all nodes, all edges)
    available at both train and test time -- only which labels are used
    for loss/eval should differ by split.
    """

    def __init__(self, processed_dir: Path = PROCESSED_DIR, config: dict = CARE_GNN_CONFIG):
        self.processed_dir = processed_dir
        self.config = config

        node_index_map, features, time_steps, labels = load_processed(processed_dir)
        adj_tdt, adj_tbt, adj_tft = load_relations(processed_dir)

        self.node_index_map = node_index_map
        self.time_steps = time_steps
        self.labels = labels

        data = HeteroData()
        data[NODE_TYPE].x = torch.tensor(features, dtype=torch.float32)
        data[NODE_TYPE].y = torch.tensor(labels, dtype=torch.long)
        data[NODE_TYPE].time_step = torch.tensor(time_steps, dtype=torch.long)

        data[NODE_TYPE, "tdt", NODE_TYPE].edge_index = torch.tensor(adj_tdt, dtype=torch.long)
        data[NODE_TYPE, "tbt", NODE_TYPE].edge_index = torch.tensor(adj_tbt, dtype=torch.long)
        data[NODE_TYPE, "tft", NODE_TYPE].edge_index = torch.tensor(adj_tft, dtype=torch.long)

        train_steps = set(config["train_time_steps"])
        test_steps = set(config["test_time_steps"])
        data[NODE_TYPE].train_mask = torch.tensor(
            [int(s) in train_steps for s in time_steps], dtype=torch.bool
        )
        data[NODE_TYPE].test_mask = torch.tensor(
            [int(s) in test_steps for s in time_steps], dtype=torch.bool
        )

        self.data = data

    @property
    def train_mask(self) -> torch.Tensor:
        return self.data[NODE_TYPE].train_mask

    @property
    def test_mask(self) -> torch.Tensor:
        return self.data[NODE_TYPE].test_mask

    @property
    def num_nodes(self) -> int:
        return self.data[NODE_TYPE].num_nodes


if __name__ == "__main__":
    dataset = AugurDataset()
    print(dataset.data)
    print(f"train nodes: {int(dataset.train_mask.sum())}")
    print(f"test nodes: {int(dataset.test_mask.sum())}")
