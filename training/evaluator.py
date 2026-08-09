"""Evaluation metrics for fraud detection. Accuracy is not reported: with
Elliptic's severe class imbalance, a model that predicts "licit" for every
node gets ~93-98% accuracy while catching zero fraud. F1 on the illicit
class is the primary metric -- it penalizes both missed fraud and
over-flagged legitimate transactions.

evaluate() expects y_true/y_pred_proba already filtered to LABELLED nodes
(0/1) for the split being scored -- filtering out unlabelled (-1) nodes is
the caller's responsibility (training/trainer.py), not this module's.
"""

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate(y_true, y_pred_proba, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true)
    y_pred_proba = np.asarray(y_pred_proba)
    y_pred = (y_pred_proba >= threshold).astype(int)

    # roc_auc_score/average_precision_score require both classes present;
    # guard rather than let a degenerate eval split crash the run.
    has_both_classes = len(np.unique(y_true)) > 1

    return {
        "f1_illicit": f1_score(y_true, y_pred, pos_label=1, zero_division=0),  # PRIMARY metric
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "auc_roc": roc_auc_score(y_true, y_pred_proba) if has_both_classes else float("nan"),
        "avg_precision": average_precision_score(y_true, y_pred_proba) if has_both_classes else float("nan"),
        "precision_illicit": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall_illicit": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
    }


def confusion(y_true, y_pred_proba, threshold: float = 0.5) -> np.ndarray:
    y_true = np.asarray(y_true)
    y_pred = (np.asarray(y_pred_proba) >= threshold).astype(int)
    return confusion_matrix(y_true, y_pred, labels=[0, 1])
