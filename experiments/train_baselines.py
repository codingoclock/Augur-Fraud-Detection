"""Entry point for training all three baselines: GraphSAGE, GAT, Isolation
Forest. Thin CLI wrapper around the already-built, already-verified training
code in models/baselines/ -- no new training logic here.
"""

import argparse
from pathlib import Path

from config import CARE_GNN_CONFIG
from models.baselines.gat import CHECKPOINT_PATH as GAT_DEFAULT_CHECKPOINT
from models.baselines.gat import train as train_gat
from models.baselines.graphsage import CHECKPOINT_PATH as GRAPHSAGE_DEFAULT_CHECKPOINT
from models.baselines.graphsage import train as train_graphsage
from models.baselines.isolation_forest import train as train_isolation_forest


def main():
    parser = argparse.ArgumentParser(description="Train GraphSAGE, GAT, and fit Isolation Forest.")
    parser.add_argument("--models", nargs="+", choices=["graphsage", "gat", "isolation_forest"],
                         default=["graphsage", "gat", "isolation_forest"])
    parser.add_argument("--epochs", type=int, default=CARE_GNN_CONFIG["epochs"], help="GraphSAGE/GAT only.")
    parser.add_argument("--balanced-undersampling", dest="balanced_undersampling", action="store_true", default=True)
    parser.add_argument("--no-balanced-undersampling", dest="balanced_undersampling", action="store_false")
    parser.add_argument("--benchmark-only", action="store_true", help="Run a single epoch, for a quick smoke test.")
    parser.add_argument("--graphsage-checkpoint-path", type=Path, default=GRAPHSAGE_DEFAULT_CHECKPOINT,
                         help="Override to avoid clobbering the real checkpoint during a quick smoke test.")
    parser.add_argument("--gat-checkpoint-path", type=Path, default=GAT_DEFAULT_CHECKPOINT)
    args = parser.parse_args()

    config = dict(CARE_GNN_CONFIG)
    config["epochs"] = args.epochs

    results = {}

    if "graphsage" in args.models:
        print("=== Training GraphSAGE ===")
        results["graphsage"] = train_graphsage(
            config=config, benchmark_only=args.benchmark_only,
            balanced_undersampling=args.balanced_undersampling, ablation_variant="graphsage_baseline",
            checkpoint_path=args.graphsage_checkpoint_path,
        )
        print(results["graphsage"])

    if "gat" in args.models:
        print("=== Training GAT ===")
        results["gat"] = train_gat(
            config=config, benchmark_only=args.benchmark_only,
            balanced_undersampling=args.balanced_undersampling, ablation_variant="gat_baseline",
            checkpoint_path=args.gat_checkpoint_path,
        )
        print(results["gat"])

    if "isolation_forest" in args.models:
        print("=== Fitting Isolation Forest ===")
        results["isolation_forest"] = train_isolation_forest(config=config)
        print(results["isolation_forest"])

    return results


if __name__ == "__main__":
    main()
