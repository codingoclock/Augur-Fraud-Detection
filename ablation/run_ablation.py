"""Assembles the final Augur ablation table from already-trained,
already-logged results. Does NOT retrain anything -- every variant here was
trained by training/trainer.py or models/baselines/*.py in earlier steps and
logged to the Augur-Elliptic MLflow experiment. This script only queries
MLflow (and one hardcoded, explicitly-flagged exception -- see
FULL_CARE_GNN_EPOCH10_REFERENCE below) and writes ablation/results/*.{csv,md}.

Retrieval failure handling: if a run can't be found or is missing the
metrics this script needs, that row is written to the table as MISSING
with a reason -- never silently retrained and never silently dropped.
"""

import csv
import os
from pathlib import Path

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
import mlflow

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MLRUNS_PATH = PROJECT_ROOT / "mlruns"
RESULTS_DIR = PROJECT_ROOT / "ablation" / "results"
EXPERIMENT_NAME = "Augur-Elliptic"

# ---------------------------------------------------------------------------
# The ONE hardcoded, non-MLflow-retrieved data point in this script.
#
# Full CARE-GNN's ORIGINAL live 100-epoch training run logged a periodic eval
# at epoch 10 (f1_illicit=0.2603, auc_roc=0.8715, precision=0.1527,
# recall=0.8800), needed here for a fair same-epoch-budget comparison
# against the two 10-epoch ablation variants. That run's MLflow record was
# later destroyed by an accidental `rm -rf mlruns` during Level 7 baseline
# work (see the full CARE-GNN row's own restoration, below, via checkpoint
# re-evaluation). Unlike the epoch-90 BEST checkpoint, no epoch-10 model
# checkpoint was ever saved (checkpointing only triggers on a NEW best
# f1_illicit, and epoch 10's 0.2603 was never the best-so-far at the time),
# so epoch 10's weights no longer exist and this number cannot be
# regenerated or re-verified by re-running inference the way the epoch-90
# number was. This value is transcribed directly from the live training
# console output captured in the project conversation record at the time,
# not from a live, re-queryable source. Flagged here explicitly rather than
# presented as an equal-footing MLflow query result.
# ---------------------------------------------------------------------------
FULL_CARE_GNN_EPOCH10_REFERENCE = {
    "f1_illicit": 0.2603,
    "auc_roc": 0.8715,
    "precision_illicit": 0.1527,
    "recall_illicit": 0.8800,
    "epoch": 10,
    "source": "transcript-recorded console output, NOT re-verifiable (no epoch-10 checkpoint was ever saved)",
}


def _get_client():
    mlflow.set_tracking_uri(MLRUNS_PATH.as_uri())
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        raise RuntimeError(f"MLflow experiment '{EXPERIMENT_NAME}' not found at {MLRUNS_PATH}")
    return client, experiment.experiment_id


def _find_run(client, experiment_id, run_name: str):
    """Most recent run with this exact run name, or None if not found."""
    runs = client.search_runs(
        experiment_id,
        filter_string=f"tags.mlflow.runName = '{run_name}'",
        order_by=["start_time DESC"],
        max_results=1,
    )
    return runs[0] if runs else None


def _best_over_history(client, run_id: str, metric_key: str):
    """Max value of a metric across its full logged history, plus the step
    (epoch) it occurred at and that step's full sibling metrics (precision/
    recall/auc at THAT SAME epoch, not independently-best values from
    different epochs, which would misrepresent a single coherent
    checkpoint)."""
    history = client.get_metric_history(run_id, metric_key)
    if not history:
        return None
    best = max(history, key=lambda h: h.value)
    return best.step, best.value


def _metrics_at_step(client, run_id: str, step: int, keys: list):
    result = {}
    for key in keys:
        history = client.get_metric_history(run_id, key)
        match = [h.value for h in history if h.step == step]
        result[key] = match[0] if match else None
    return result


def retrieve_variant(client, experiment_id, run_name: str, prefixed: bool = True, use_best_over_history: bool = True):
    """Returns a dict of {f1_illicit, auc_roc, precision_illicit,
    recall_illicit, epoch} for a logged run, or {"missing": reason} if
    retrieval fails for any reason -- never falls back to retraining."""
    run = _find_run(client, experiment_id, run_name)
    if run is None:
        return {"missing": f"no MLflow run found with name '{run_name}'"}

    prefix = "test/" if prefixed else ""
    keys = [f"{prefix}f1_illicit", f"{prefix}auc_roc", f"{prefix}precision_illicit", f"{prefix}recall_illicit"]

    if use_best_over_history:
        best = _best_over_history(client, run.info.run_id, keys[0])
        if best is None:
            return {"missing": f"run '{run_name}' found but has no '{keys[0]}' metric history"}
        step, f1 = best
        at_step = _metrics_at_step(client, run.info.run_id, step, keys)
        return {
            "f1_illicit": f1,
            "auc_roc": at_step[keys[1]],
            "precision_illicit": at_step[keys[2]],
            "recall_illicit": at_step[keys[3]],
            "epoch": step,
        }
    else:
        metrics = run.data.metrics
        missing_keys = [k for k in keys if k not in metrics]
        if missing_keys:
            return {"missing": f"run '{run_name}' found but missing metric(s) {missing_keys}"}
        return {
            "f1_illicit": metrics[keys[0]],
            "auc_roc": metrics[keys[1]],
            "precision_illicit": metrics[keys[2]],
            "recall_illicit": metrics[keys[3]],
            "epoch": None,
        }


def build_table():
    client, experiment_id = _get_client()

    # --- Main 7 variants -----------------------------------------------
    rows = []

    full_care_gnn = retrieve_variant(client, experiment_id, "CARE-GNN-full-v3-minibatch-RESTORED", use_best_over_history=False)
    if "missing" not in full_care_gnn:
        full_care_gnn["epoch"] = 90  # restored from the epoch-90 best checkpoint; not in MLflow history (single-point backfill)
    rows.append({"variant": "CARE-GNN (full)", "epochs_trained": 90, "epochs_budget": 100, **full_care_gnn})

    graphsage = retrieve_variant(client, experiment_id, "graphsage-baseline")
    rows.append({"variant": "GraphSAGE", "epochs_trained": graphsage.get("epoch"), "epochs_budget": 100, **graphsage})

    gat = retrieve_variant(client, experiment_id, "gat-baseline")
    rows.append({"variant": "GAT", "epochs_trained": gat.get("epoch"), "epochs_budget": 100, **gat})

    isolation_forest = retrieve_variant(client, experiment_id, "isolation-forest-baseline", prefixed=False, use_best_over_history=False)
    rows.append({"variant": "Isolation Forest", "epochs_trained": None, "epochs_budget": None, **isolation_forest})

    single_relation_tdt = retrieve_variant(client, experiment_id, "CARE-GNN-single_relation_tdt-100ep")
    rows.append({"variant": "CARE-GNN single relation (tdt only)", "epochs_trained": single_relation_tdt.get("epoch"), "epochs_budget": 100, **single_relation_tdt})

    no_rl = retrieve_variant(client, experiment_id, "CARE-GNN-no_rl_selector-10ep")
    rows.append({
        "variant": "CARE-GNN w/o RL selector (fixed p=0.5)",
        "epochs_trained": no_rl.get("epoch"), "epochs_budget": 10, "capped": True, **no_rl,
    })

    no_label_sim = retrieve_variant(client, experiment_id, "CARE-GNN-no_label_similarity-10ep")
    rows.append({
        "variant": "CARE-GNN w/o label-aware similarity (λ1=0, redefined for v3 -- see note)",
        "epochs_trained": no_label_sim.get("epoch"), "epochs_budget": 10, "capped": True, **no_label_sim,
    })

    # dedupe the 'epochs_trained' collision: for the capped variants above,
    # epoch is the BEST epoch within the 10-epoch run (should be 10, since
    # loss was still monotonically improving at cutoff), not a separate
    # "epochs_budget" -- both are kept in the row for clarity.

    # --- Same-epoch-budget reference table (10 epochs) ------------------
    epoch10_rows = [
        {"variant": "CARE-GNN (full model) @ epoch 10", **FULL_CARE_GNN_EPOCH10_REFERENCE,
         "note": FULL_CARE_GNN_EPOCH10_REFERENCE["source"]},
        {"variant": "CARE-GNN w/o RL selector @ epoch 10", "f1_illicit": no_rl.get("f1_illicit"),
         "auc_roc": no_rl.get("auc_roc"), "precision_illicit": no_rl.get("precision_illicit"),
         "recall_illicit": no_rl.get("recall_illicit"), "epoch": 10, "note": "MLflow-retrieved, epoch 10 (this variant's only full budget)"},
        {"variant": "CARE-GNN w/o label-aware similarity @ epoch 10", "f1_illicit": no_label_sim.get("f1_illicit"),
         "auc_roc": no_label_sim.get("auc_roc"), "precision_illicit": no_label_sim.get("precision_illicit"),
         "recall_illicit": no_label_sim.get("recall_illicit"), "epoch": 10, "note": "MLflow-retrieved, epoch 10 (this variant's only full budget)"},
    ]

    return rows, epoch10_rows


def _fmt(v):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def write_outputs(rows, epoch10_rows):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Sort main table by F1_illicit descending. Missing rows (no "f1_illicit"
    # key) sort last, not silently dropped.
    def sort_key(r):
        return r.get("f1_illicit") if r.get("f1_illicit") is not None else -1.0

    sorted_rows = sorted(rows, key=sort_key, reverse=True)

    # --- CSV: main table -------------------------------------------------
    csv_path = RESULTS_DIR / "ablation_results.csv"
    fieldnames = ["variant", "f1_illicit", "auc_roc", "precision_illicit", "recall_illicit",
                  "epoch", "epochs_trained", "epochs_budget", "capped", "missing"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in sorted_rows:
            writer.writerow(r)

    # --- CSV: epoch-10 same-budget reference table ------------------------
    csv_path_ep10 = RESULTS_DIR / "ablation_results_epoch10_reference.csv"
    fieldnames_ep10 = ["variant", "f1_illicit", "auc_roc", "precision_illicit", "recall_illicit", "epoch", "note"]
    with open(csv_path_ep10, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames_ep10, extrasaction="ignore")
        writer.writeheader()
        for r in epoch10_rows:
            writer.writerow(r)

    # --- Markdown ----------------------------------------------------------
    md_lines = []
    md_lines.append("# Augur Ablation Results\n")
    md_lines.append(
        "All CARE-GNN variants share identical hyperparameters (lr, weight_decay, "
        "hidden_dim, batch_size, seed) except the ablated component. GraphSAGE/GAT "
        "trained on the merged tdt+tbt+tft graph (fairness decision, see "
        "models/baselines/graphsage.py). Isolation Forest is tabular-only, unsupervised "
        "(no epoch concept).\n"
    )
    md_lines.append(
        "**Important -- read before comparing rows:** `CARE-GNN w/o RL selector` and "
        "`CARE-GNN w/o label-aware similarity` were trained for only **10 epochs**, "
        "not the 100 used for every other CARE-GNN variant, due to hardware/compute "
        "constraints on this machine (each would take ~13-16 hours at full length). "
        "Their rows below reflect the best result achievable within that 10-epoch "
        "budget -- they are **not** directly comparable epoch-for-epoch to the "
        "100-epoch full model's result. See the second table for the fair, "
        "same-epoch-budget comparison.\n"
    )

    md_lines.append("## Main results (sorted by F1_illicit, best achieved within each variant's own budget)\n")
    header = ["Variant", "F1_illicit", "AUC-ROC", "Precision", "Recall", "Best epoch", "Epochs trained/budget"]
    md_lines.append("| " + " | ".join(header) + " |")
    md_lines.append("|" + "---|" * len(header))
    for r in sorted_rows:
        if "missing" in r:
            md_lines.append(f"| {r['variant']} | MISSING | MISSING | - | - | - | {r['missing']} |")
            continue
        budget_str = f"{r.get('epochs_trained', '-')}/{r.get('epochs_budget', '-')}"
        if r.get("capped"):
            budget_str += " ⚠️ capped"
        md_lines.append(
            f"| {r['variant']} | {_fmt(r.get('f1_illicit'))} | {_fmt(r.get('auc_roc'))} | "
            f"{_fmt(r.get('precision_illicit'))} | {_fmt(r.get('recall_illicit'))} | "
            f"{_fmt(r.get('epoch'))} | {budget_str} |"
        )

    md_lines.append("")
    md_lines.append("## Same-epoch-budget reference: full CARE-GNN vs. the two 10-epoch variants, all @ epoch 10\n")
    md_lines.append(
        "This is the fair comparison for the two capped variants above -- full CARE-GNN's "
        "*best* (epoch 90) is a different, later point in its own training and should not "
        "be read as \"what full CARE-GNN looked like at 10 epochs.\"\n"
    )
    header2 = ["Variant", "F1_illicit", "AUC-ROC", "Precision", "Recall", "Epoch", "Note"]
    md_lines.append("| " + " | ".join(header2) + " |")
    md_lines.append("|" + "---|" * len(header2))
    for r in epoch10_rows:
        md_lines.append(
            f"| {r['variant']} | {_fmt(r.get('f1_illicit'))} | {_fmt(r.get('auc_roc'))} | "
            f"{_fmt(r.get('precision_illicit'))} | {_fmt(r.get('recall_illicit'))} | "
            f"{_fmt(r.get('epoch'))} | {r.get('note', '')} |"
        )

    md_lines.append("")
    md_lines.append(
        "**Note on the label-aware-similarity variant's redefinition:** the original "
        "build spec's \"w/o label-aware similarity\" ablation was designed against v1/v2's "
        "cosine+label-flag similarity module, which no longer exists in v3. v3's "
        "similarity module is a per-layer MLP trained via an auxiliary loss (Eq. 4) with "
        "no \"drop the label term, keep a feature-only term\" option -- label-training IS "
        "the mechanism. The variant here instead sets `lambda1=0`, disabling that "
        "auxiliary loss entirely so the layer-1 MLP never receives gradient and stays at "
        "random initialization all run, used for top-p filtering but never shaped by "
        "label information. This tests a different (but analogous) hypothesis than the "
        "original spec text describes, and the two should not be treated as equivalent "
        "if compared against documentation written for v1/v2.\n"
    )

    md_path = RESULTS_DIR / "ablation_results.md"
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines))

    return csv_path, csv_path_ep10, md_path


if __name__ == "__main__":
    rows, epoch10_rows = build_table()
    csv_path, csv_path_ep10, md_path = write_outputs(rows, epoch10_rows)
    print(f"Wrote {csv_path}")
    print(f"Wrote {csv_path_ep10}")
    print(f"Wrote {md_path}")
    for r in rows:
        print(r)
