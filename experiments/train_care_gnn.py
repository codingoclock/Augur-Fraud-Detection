"""Entry point for training the full (3-relation) CARE-GNN model.

Thin CLI wrapper around training/trainer.py's train() -- no new training
logic here, this just exposes the already-built, already-verified trainer
as a runnable script. Defaults reflect the corrected protocol established
in Level 9 (balanced under-sampling; config.py's tau=0.02/lambda1=2 are
already the module defaults, not repeated here).

--checkpoint-path is required, not optional, matching Level 10's fix to
trainer.py: there is no shared default checkpoint path left to collide on.
"""

import argparse
from pathlib import Path

from config import CARE_GNN_CONFIG
from training.trainer import train


def main():
    parser = argparse.ArgumentParser(description="Train the full CARE-GNN model on Elliptic.")
    parser.add_argument("--checkpoint-path", type=Path, required=True, help="Where to save the best-test-F1 checkpoint.")
    # NOT CARE_GNN_CONFIG["epochs"] (100) -- the real corrected-protocol run
    # that produced the reported F1_illicit=0.5514 used 500 epochs via a
    # manually-overridden config, not config.py's own default. 500 here
    # matches that real, reported result rather than silently reproducing
    # a different, never-actually-evaluated 100-epoch run.
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--eval-every-n-epochs", type=int, default=20)
    parser.add_argument("--fraud-chunk-size", type=int, default=1024)
    parser.add_argument("--balanced-undersampling", dest="balanced_undersampling", action="store_true", default=True)
    parser.add_argument("--no-balanced-undersampling", dest="balanced_undersampling", action="store_false")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--benchmark-only", action="store_true", help="Run a single epoch, for a quick smoke test.")
    args = parser.parse_args()

    config = dict(CARE_GNN_CONFIG)
    config["epochs"] = args.epochs

    result = train(
        config=config,
        checkpoint_path=args.checkpoint_path,
        benchmark_only=args.benchmark_only,
        balanced_undersampling=args.balanced_undersampling,
        fraud_chunk_size=args.fraud_chunk_size,
        eval_every_n_epochs=args.eval_every_n_epochs,
        ablation_variant="full",
        run_name=args.run_name,
    )
    print(result)


if __name__ == "__main__":
    main()
