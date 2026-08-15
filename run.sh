#!/usr/bin/env bash
# One-step: fresh clone -> running API, using already-trained checkpoints.
# Does NOT silently retrain (that's hours of work) -- it brings up the demo
# as-is and tells you exactly what to run yourself if something's missing.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

RAW_DIR="data/raw"
REQUIRED_CSVS=("elliptic_txs_features.csv" "elliptic_txs_edgelist.csv" "elliptic_txs_classes.csv")
PROCESSED_DIR="data/processed"
REQUIRED_PROCESSED=("features.pkl" "labels.pkl" "time_steps.pkl" "node_index_map.pkl" "adj_tdt.pkl" "adj_tbt.pkl" "adj_tft.pkl")
CHECKPOINTS_DIR="models/checkpoints"
REQUIRED_CHECKPOINTS=("care_gnn_full_best.pt" "care_gnn_single_relation_tdt_best.pt" "graphsage_best.pt" "gat_best.pt" "isolation_forest_best.pkl")

echo "=== Augur: one-step run ==="
echo

# --- Step 1: raw data ---------------------------------------------------
missing_csv=0
for f in "${REQUIRED_CSVS[@]}"; do
    if [ ! -f "$RAW_DIR/$f" ]; then
        missing_csv=1
        echo "MISSING: $RAW_DIR/$f"
    fi
done
if [ "$missing_csv" -eq 1 ]; then
    cat <<'EOF'

ERROR: Elliptic Bitcoin Dataset CSVs not found.

Download the dataset from Kaggle:
  https://www.kaggle.com/datasets/ellipticco/elliptic-data-set

and place these three files in data/raw/:
  elliptic_txs_features.csv
  elliptic_txs_edgelist.csv
  elliptic_txs_classes.csv

Then re-run this script.
EOF
    exit 1
fi
echo "[1/5] Raw data present."

# --- Step 2: cached graph artifacts -------------------------------------
missing_processed=0
for f in "${REQUIRED_PROCESSED[@]}"; do
    if [ ! -f "$PROCESSED_DIR/$f" ]; then
        missing_processed=1
    fi
done

if [ "$missing_processed" -eq 1 ]; then
    echo "[2/5] Cached graph artifacts not found in $PROCESSED_DIR/."
    echo "      Building the graph now -- this takes ~40 minutes (mostly the"
    echo "      tft relation's 203,769x203,769 cosine-similarity search)."
    echo "      Starting now, please wait..."
    docker compose run --rm app python3 -m graph.loader
    docker compose run --rm app python3 -m graph.relations
    echo "      Graph build complete."
else
    echo "[2/5] Cached graph artifacts present, skipping graph build."
fi

# --- Step 3: trained checkpoints -----------------------------------------
missing_ckpt=0
for f in "${REQUIRED_CHECKPOINTS[@]}"; do
    if [ ! -f "$CHECKPOINTS_DIR/$f" ]; then
        missing_ckpt=1
        echo "MISSING: $CHECKPOINTS_DIR/$f"
    fi
done

if [ "$missing_ckpt" -eq 1 ]; then
    cat <<'EOF'

[3/5] WARNING: one or more trained checkpoints are missing.

Training was NOT run automatically -- the full CARE-GNN + baseline suite
takes 4+ hours on a CPU-only machine, too long to run silently as part of
"one step". Bringing the API up anyway; /predict will return a clear error
for any model whose checkpoint is missing rather than this script hanging
for hours unexpectedly.

To train everything yourself:
  docker compose run app python3 -m experiments.train_care_gnn --checkpoint-path models/checkpoints/care_gnn_full_best.pt
  docker compose run app python3 -m experiments.train_baselines

EOF
else
    echo "[3/5] All 5 trained checkpoints present."
fi

# --- Step 4: bring up services -------------------------------------------
echo "[4/5] Starting mlflow and api services..."
docker compose up -d mlflow api

# --- Step 5: print real URLs ----------------------------------------------
echo "[5/5] Up. Endpoints:"
echo
echo "  API health check:  curl http://localhost:8000/"
echo "  MLflow UI:          http://localhost:5001"
echo
echo "  Example prediction (real node id from the dataset, any integer in [0, 203769)):"
echo "    curl -X POST http://localhost:8000/predict \\"
echo "      -H \"Content-Type: application/json\" \\"
echo "      -d '{\"node_id\": 136279, \"model\": \"graphsage\"}'"
echo
echo "  Fraud-ring visualization for a node:"
echo "    curl http://localhost:8000/subgraph/136279 -o subgraph.html"
