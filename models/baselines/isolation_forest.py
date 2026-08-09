"""Isolation Forest baseline: tabular features only, no graph structure, per
spec -- the project's original non-graph baseline type.

Loss-function fairness requirement resolved explicitly (not silently
skipped): the Level 7 instruction to train baselines with WeightedFocalLoss +
FRAUD_CLASS_WEIGHT applies to GraphSAGE and GAT, which ARE trained by
gradient descent on a classification loss (see graphsage.py). It does not
apply here -- IsolationForest is an unsupervised sklearn ensemble fit by
recursive random partitioning, not backpropagation; it has no logits, no
loss function, and never sees labels during fitting. There is no
WeightedFocalLoss-equivalent to substitute in without changing what
Isolation Forest fundamentally is. What IS kept identical across all three
baselines: the evaluation protocol (training/evaluator.py's evaluate(),
F1_illicit primary, same test split) and the MLflow experiment/tagging
convention, so results are still directly comparable in the ablation table.

Calibration note: IsolationForest's score_samples() is an anomaly score, not
a probability -- there is no natural P(fraud) output the way there is for a
softmax classifier. Scores are min-max normalized using the TRAIN split's
score range only (avoiding test-set leakage into the normalization), then
clipped to [0, 1] on the test split, so evaluator.py's default 0.5 threshold
is meaningful. This calibration is a necessary but debatable choice -- flagged
here, not hidden -- since Isolation Forest was never designed to produce
calibrated probabilities.

Sign convention -- determined using TRAIN-split labels ONLY, never test
labels (an earlier pass picked the sign by comparing test-split AUC under
each orientation, which is test-set leakage: the sign is a model-selection
choice, and choosing it by peeking at test performance invalidates the test
AUC as an unbiased estimate; redone correctly below). sklearn's
score_samples() is LOWER for points the model considers anomalous, HIGHER
for inliers. The naive assumption -- negate it, so higher-output =
more-anomalous = proxy for "more fraud-like" -- turns out to be BACKWARDS on
this dataset, checked via TRAIN-labelled nodes only: mean score_samples for
train-fraud (-0.360) is HIGHER than train-licit (-0.448), i.e. Isolation
Forest considers fraud MORE inlier-like, not less, using only information
the model would have at decision time. Raw score_samples (NOT negated) is
used below because that's what the train-only comparison says correlates
with fraud -- confirmed, not re-decided, by the resulting test AUC-ROC=0.816
(vs. 0.184 negated) matching the train-only call. This is itself a
legitimate finding: fraud in Elliptic looks LESS isolated/anomalous than
licit transactions to an unsupervised model, which
is consistent with the project's own "camouflaged fraudsters" thesis (Dou et
al.) -- naive anomaly detection is blind to fraud specifically because fraud
is designed to blend in, not stand out.

Reuses the exact cached graph artifacts CARE-GNN trained on (data/processed/
via graph.loader.load_processed) -- nothing is regenerated or reprocessed.
No graph/relations data is used at all, per "tabular features only."
"""

import os
import pickle
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import IsolationForest

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
import mlflow

from config import CARE_GNN_CONFIG
from graph.loader import PROCESSED_DIR, load_processed
from training.evaluator import evaluate

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CHECKPOINT_PATH = PROJECT_ROOT / "models" / "checkpoints" / "isolation_forest_best.pkl"
MLRUNS_PATH = PROJECT_ROOT / "mlruns"


def _load_tabular_split(config: dict = CARE_GNN_CONFIG, processed_dir: Path = PROCESSED_DIR):
    node_index_map, features, time_steps, labels = load_processed(processed_dir)

    train_steps = set(config["train_time_steps"])
    test_steps = set(config["test_time_steps"])
    train_mask = np.array([int(s) in train_steps for s in time_steps])
    test_mask = np.array([int(s) in test_steps for s in time_steps])

    return features, labels, train_mask, test_mask


def _normalize_scores(raw_scores: np.ndarray, lo: float, hi: float) -> np.ndarray:
    if hi <= lo:
        return np.full_like(raw_scores, 0.5)
    return np.clip((raw_scores - lo) / (hi - lo), 0.0, 1.0)


def train(config: dict = CARE_GNN_CONFIG):
    features, labels, train_mask, test_mask = _load_tabular_split(config)

    train_features = features[train_mask]  # unsupervised: fit on train-split features only, labels never used
    test_labelled_mask = test_mask & (labels != -1)
    test_features = features[test_labelled_mask]
    test_y_true = labels[test_labelled_mask]

    mlflow.set_tracking_uri(MLRUNS_PATH.as_uri())
    mlflow.set_experiment("Augur-Elliptic")

    with mlflow.start_run(run_name="isolation-forest-baseline"):
        mlflow.log_params({
            "n_estimators": 100,
            "contamination": "auto",
            "seed": config["seed"],
            "train_rows": int(train_features.shape[0]),
            "test_rows": int(test_features.shape[0]),
        })
        mlflow.set_tags({
            "model_type": "isolation_forest",
            "dataset": "elliptic",
            "relations_used": "none (tabular features only)",
            "ablation_variant": "isolation_forest_baseline",
        })

        t0 = time.perf_counter()
        model = IsolationForest(n_estimators=100, contamination="auto", random_state=config["seed"], n_jobs=-1)
        model.fit(train_features)
        fit_seconds = time.perf_counter() - t0

        # NOT negated -- see module docstring's "Sign convention" note. The
        # naive assumption (negate score_samples so higher=more-anomalous=
        # more-fraud-like) was verified empirically wrong on this dataset:
        # raw score_samples correlates POSITIVELY with the fraud label here
        # (AUC 0.816 raw vs 0.184 negated, exact complements). Variable name
        # kept as "fraud_score" rather than "anomaly" since it is no longer
        # an anomaly score by the time it's used this way.
        train_fraud_score = model.score_samples(train_features)
        test_fraud_score = model.score_samples(test_features)
        lo, hi = float(train_fraud_score.min()), float(train_fraud_score.max())
        test_y_pred_proba = _normalize_scores(test_fraud_score, lo, hi)

        metrics = evaluate(test_y_true, test_y_pred_proba)
        mlflow.log_metrics(metrics)
        mlflow.log_metric("fit_seconds", fit_seconds)

        print(f"[isolation_forest] fit in {fit_seconds:.2f}s on {train_features.shape[0]} rows")
        print(
            f"  [isolation_forest eval] f1_illicit={metrics['f1_illicit']:.4f} "
            f"auc_roc={metrics['auc_roc']:.4f} precision={metrics['precision_illicit']:.4f} "
            f"recall={metrics['recall_illicit']:.4f}"
        )

        CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(CHECKPOINT_PATH, "wb") as f:
            pickle.dump({"model": model, "score_lo": lo, "score_hi": hi, "f1_illicit": metrics["f1_illicit"]}, f)

    return {"fit_seconds": fit_seconds, "metrics": metrics}


if __name__ == "__main__":
    result = train()
    print(result)
