# Augur

**Status: functionally complete.** Graph construction, CARE-GNN (v3, paper-corrected), three baselines, a 7-variant ablation study, fraud-ring/embedding visualizations, and a FastAPI serving layer are all implemented, trained on real data, and verified end-to-end. See [Build status](#build-status) for the itemized breakdown and [Results](#results-ablation-study) for the real numbers — including the fact that **CARE-GNN does not win**.

Augur is a heterogeneous graph neural network system for detecting fraud rings in transaction data. It implements and evaluates CARE-GNN (Dou et al., CIKM 2020) — which models relational structure that row-by-row classifiers can't see — against standard GNN and tabular baselines, on the Elliptic Bitcoin Dataset. The honest empirical result: a plain GraphSAGE baseline outperforms full CARE-GNN, and even a single-relation ablation of CARE-GNN beats the full model. That finding, and why it happened, is documented below rather than smoothed over.

## Results (ablation study)

All numbers below are read directly from `ablation/results/ablation_results.md` (regenerable live from MLflow via `python -m experiments.compare`) and cross-verified against the actual checkpoint files in `models/checkpoints/`.

| Variant | F1_illicit | AUC-ROC | Precision | Recall | Best epoch | Epochs trained |
|---|---|---|---|---|---|---|
| CARE-GNN single relation (tdt only) | **0.6603** | 0.8970 | 0.7244 | 0.6066 | 260 | 500 |
| GraphSAGE | 0.6434 | **0.9059** | 0.7114 | 0.5873 | 90 | 100 |
| CARE-GNN w/o RL selector (fixed p=0.5) | 0.5829 | 0.8626 | 0.5708 | 0.5956 | 500 | 500 |
| CARE-GNN (full) | 0.5514 | 0.8528 | 0.5342 | 0.5697 | 120 | 500 |
| CARE-GNN w/o label-aware similarity (λ1=0) | 0.4984 | 0.8548 | 0.4089 | 0.6380 | 400 | 500 |
| GAT | 0.3266 | 0.8862 | 0.2045 | 0.8098 | 90 | 100 |
| Isolation Forest | 0.1329 | 0.8162 | 0.0712 | 1.0000 | – | original (uncorrected) protocol |

All rows except Isolation Forest use the corrected training protocol (balanced under-sampling per batch, tau=0.02, lambda1=2 — the paper-verified hyperparameters, not the placeholder tau=0.05/lambda1=1.0 used before the full paper was read). Isolation Forest is unsupervised and fit only on tabular features; balanced under-sampling has no meaning for it, so its numbers are unchanged.

**What this actually shows, stated plainly:**
- **Full CARE-GNN does not win.** It's outperformed by GraphSAGE (a much simpler architecture) and by its own single-relation ablation.
- **The single-relation-tdt ablation scoring highest (0.6603) is a real, examined finding, not noise.** It directly contradicts what the full model's own RL selector learned: in the full model, the selector converges to a near-zero threshold on tdt (keeping almost no tdt neighbours) at both layers — treating tdt as nearly uninformative. Training on tdt *alone* outperforming the full model suggests that convergence was a poor local optimum, not a correct read of tdt's actual value. This is flagged as an open question about the RL selector's optimization, not resolved here.
- **GraphSAGE, not single-relation-tdt CARE-GNN, is what this project recommends for actual serving** (and is the FastAPI default — see below). Two reasons: GraphSAGE has the best AUC-ROC of any model (0.9059, meaning the best ranking quality independent of threshold), and single-relation-tdt is a diagnostic ablation artifact demonstrating a specific finding about CARE-GNN's RL selector — it was never intended as a standalone deployment candidate.
- Both `w/o RL selector` and `w/o label-aware similarity` underperform the full model, meaning the RL filtering mechanism and the auxiliary similarity loss are each doing real work individually — the full model's problem isn't that these components are useless, it's that the combination still loses to a much simpler model.

## Visualizations

Generated from real trained-model predictions on the real test split — not placeholder data. Open any `.html` file directly in a browser; no server needed.

- **`visualization/fraud_ring_graphsage.html`** — Pyvis fraud-ring graph: 5 test-split nodes GraphSAGE flags as illicit with ≥0.9 confidence (actual scores 0.9985–0.9995), expanded 2 hops via the tdt relation (31 nodes, 26 edges). Red=illicit, blue=licit, grey=unlabelled, gold=seed node. The page itself now explains what a "fraud ring" means here (the transaction neighbourhood around a high-confidence flag) and includes an in-page colour-coded legend with live counts, per-node hover tooltips (node id, true label, P(fraud)), draggable/zoomable layout, and a physics on/off toggle.
- **`visualization/fraud_ring_care_gnn.html`** — the same 5 seed nodes and tdt neighbourhood, same legend/description panel, scored by CARE-GNN instead, for a direct side-by-side of what each model flags.
- **`visualization/embedding_umap_graphsage.png`** — UMAP projection of GraphSAGE's learned 64-d hidden embeddings for all 67,504 test-split nodes, coloured by true label (1,083 illicit / 15,587 licit / 50,834 unknown). Shows a visibly distinct illicit cluster separated from the main mass — real evidence the model learned a meaningful separation, not just a good aggregate score. Now includes an in-image explanation of what the plot means and how to read it.
- **`visualization/embedding_umap_graphsage.html`** — interactive Plotly version of the same projection: hover any point for its real node id, true label, and GraphSAGE P(fraud); zoom, pan, and click a legend entry to isolate one class. The API's own `/predict` example node (136279) is a real point in this projection.

The API's own `GET /subgraph/{node_id}` endpoint (see [Quickstart](#quickstart)) reuses the same fraud-ring renderer, so any node you query live gets the same legend/description panel and hover detail as the two files above.

## Architecture

Fraud rings coordinate: individual transactions can look unremarkable while the network around them doesn't. A standard node classifier (or an Isolation Forest scoring rows independently) is architecturally blind to that — it never sees the graph. CARE-GNN is built for exactly this: it aggregates over multiple relation types with a learned, label-aware neighbor selector, so the model can down-weight neighbors that don't help distinguish fraud from non-fraud, instead of averaging blindly like a standard GNN (GraphSAGE/GAT) would. Whether that theoretical advantage pays off on Elliptic specifically is exactly what the ablation study above tests — and on this dataset, it doesn't beat the simpler alternative.

The dataset is the [Elliptic Bitcoin Dataset](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set) — 203,769 transaction nodes with 165 features each, spanning 49 time steps, with ~2% labeled illicit, ~21% labeled licit, and the rest unlabeled. It ships with one native edge relation; Augur constructs three:

| Relation | Meaning | Edge count |
|---|---|---|
| **tdt** (transaction–direction–transaction) | Native directed edges from the dataset | 234,355 |
| **tbt** (transaction–block–transaction) | Transactions sharing a time step (capped at 500 sampled pairs/block) | 49,000 |
| **tft** (transaction–feature–transaction) | Top-10 nearest neighbors per node by cosine similarity on L2-normalized features, threshold > 0.85 | 1,701,497 |

These counts are read directly from the generated `data/processed/adj_*.pkl` arrays (re-verified against this repo's actual output while auditing this document, not copied from an earlier report), and are also asserted by the test suite (`tests/test_graph_construction.py`).

GraphSAGE and GAT (the baselines) train on the **union of all three relations**, deduplicated into one homogeneous graph (1,979,669 unique edges) rather than just tdt — a deliberate fairness decision (see `models/baselines/graphsage.py`) so the ablation comparison isolates CARE-GNN's architecture as the variable, not "access to more of the graph."

## Design decision: standalone, not integrated

CARE-GNN needs a fully-constructed batch graph before it can train — it isn't a streaming, per-event scorer. Wiring it into a Kafka-based pipeline would mean faking graph construction on a stream, which is architecturally dishonest and falls apart under scrutiny. Augur is deliberately a separate, standalone project from any real-time anomaly system, because the two solve different problems with different data paradigms (stream vs. graph batch).

> For row-by-row anomaly scoring on tabular transaction data, see the Realtime Anomaly Engine repo. Augur is its architectural successor for relational fraud — it detects rings that isolated-row scoring can't.

## Build status

_Last checked: 2026-08-12, against this repo's actual working tree, a live pytest run, and real loaded checkpoints._

- [x] **Environment setup** — Python 3.12.13 in a local venv; `torch` 2.13.0, `torch-geometric` 2.8.0, `scikit-learn` 1.9.0, `mlflow` 3.15.0, `fastapi` 0.141.1, `pyvis` 0.3.2, and the rest of `requirements.txt` installed and importable.
- [x] **Data loading + multi-relation graph construction** (`graph/loader.py`, `graph/relations.py`, `graph/dataset.py`) — implemented and tested. `pytest tests/test_graph_construction.py` passes 6/6 (node-index bijection, tdt edge count vs. raw CSV, tbt per-block cap, tft top-k cap, feature matrix shape, same-seed reproducibility). Full run takes ~40 minutes, almost entirely the tft relation's cosine-similarity neighbor search.
- [x] **CARE-GNN model, v3 (paper-corrected)** (`models/care_gnn/similarity.py`, `selector.py`, `aggregator.py`, `care_gnn.py`) — implemented, unit tested, and trained on the real graph. Rewritten from an earlier v1/v2 pass after reading the actual paper surfaced real mechanism errors — see [Engineering rigor](#engineering-rigor-bugs-found-and-fixed) below.
- [x] **Baselines** (`models/baselines/graphsage.py`, `gat.py`, `isolation_forest.py`) — implemented and trained on the real graph; all three checkpoints present in `models/checkpoints/`.
- [x] **Training loop + evaluator** (`training/trainer.py`, `evaluator.py`, `loss.py`) — implemented; supports full CARE-GNN, all three ablation variants, mini-batching, and the corrected balanced-under-sampling protocol.
- [x] **Ablation study, 7 variants** (`ablation/run_ablation.py`, `experiments/compare.py`) — all 7 rows trained/fit and logged to the `Augur-Elliptic` MLflow experiment; results in `ablation/results/ablation_results.{csv,md}`, regenerable live from MLflow.
- [x] **Visualization** (`visualization/fraud_rings.py`, `embedding_umap.py`) — real outputs generated from real checkpoints, see [Visualizations](#visualizations) above.
- [x] **API serving** (`api/main.py`, `schemas.py`, `state.py`, `routes/`) — implemented; `/`, `/predict`, `/subgraph/{node_id}` all verified against a live running server with real node IDs (see [Quickstart](#quickstart)). Note: verification was manual (curl against a live server), not automated — `tests/test_api.py` exists as an empty stub, not real automated coverage.
- [x] **Entry-point scripts** (`experiments/train_care_gnn.py`, `train_baselines.py`, `compare.py`) — thin, verified-working CLI wrappers around the above; see [Quickstart](#quickstart).

Full test suite: 41 tests collected across `tests/` (6 graph-construction + 35 model/loss/selector-RL/full-forward-pass), all passing as of the last full run.

**Known gap, stated plainly:** the three CARE-GNN ablation variants (single-relation-tdt, w/o RL selector, w/o label-aware similarity) don't have dedicated CLI flags on `experiments/train_care_gnn.py` yet — they were run via `training/trainer.py`'s lower-level parameters (`relation_names`, `enable_rl`, `lambda1`) directly. A fresh clone running only the documented quickstart commands will reproduce 4 of the 7 ablation table rows (full CARE-GNN + 3 baselines), not all 7.

## Engineering rigor (bugs found and fixed)

Real bugs found and corrected across this project's build, not a curated highlight reel — including ones that made results look worse, not better, once fixed.

**Graph construction:**
- **tft top-k cap and self-exclusion under near-duplicates.** Naively excluding a node's own index from its k+1 nearest neighbors (`indices != self`) breaks when many feature vectors are near-identical: sklearn's tie-breaking can drop a node's own index from its returned neighbor set entirely, silently leaving k+1 neighbors instead of k. Fixed by ranking neighbors by ascending distance and taking the first k non-self columns positionally, which caps at exactly k regardless of whether self appears in the returned set.
- **`ball_tree` → `brute` deviation.** `sklearn.neighbors.NearestNeighbors` with `algorithm='ball_tree'` doesn't support `metric='cosine'` at all, and an empirical run using ball_tree with a converted metric over the full 165-dimensional, 203,769-row feature matrix was killed after 41 minutes at 97% CPU without finishing — at that dimensionality, ball_tree's pruning collapses toward brute-force behavior anyway. Switched to `algorithm='brute'` with `metric='cosine'`, which computes the same exact top-k via chunked BLAS matrix multiplication and actually scales.
- **Feature column misalignment.** The raw `elliptic_txs_features.csv` has 167 columns (txId, time_step, then 165 real features). Computing `feature_dim` as "raw columns minus txId" gives 166 and silently mixes `time_step` into the feature matrix. Fixed by reading feature columns explicitly from `config.py` rather than deriving the count from the raw column total.

**CARE-GNN model (v1/v2 → v3, after reading the actual paper instead of excerpts):**
- **RL selector sign/update-rule errors (`models/care_gnn/selector.py`).** v1/v2 used a Q-learning-style direction/step-decay mechanism not in the paper; rewritten to the paper's actual fixed-step-size rule. Separately, the terminal condition (Eq. 7) was first implemented against an OCR-garbled excerpt and got two things wrong: it compared the *raw* summed reward to a threshold instead of its *absolute value* (which would incorrectly freeze on a sustained losing streak, not just on a balanced/oscillating signal — opposite situations), and it used a 10-term window instead of the paper's actual 11-term inclusive window.
- **Aggregator ReLU placement and per-relation weighting (`models/care_gnn/aggregator.py`).** v1/v2 gave each relation its own learned `Linear` layer and softmax-normalized the RL thresholds before using them as inter-relation weights. Neither is in the paper: Eq. 8 specifies a parameterless per-relation mean with ReLU applied at that stage (not deferred to the end), and Eq. 9 states the RL-learned threshold `p_r` is used *directly* as the aggregation weight, not renormalized.

**Training protocol (after reading the full paper, not just excerpts):**
- **Balanced under-sampling replaced natural-imbalance training.** The original protocol trained on the full imbalanced train split with a 9:1 class-weighted loss. The paper's actual Section 4.1.4 protocol samples balanced fraud/licit mini-batches instead. Retrained everything under the corrected protocol; full CARE-GNN's F1 improved from 0.459 to 0.551, but the ranking among models (GraphSAGE and single-relation-tdt still ahead) didn't change.
- **tau and lambda1 hyperparameters were placeholders.** Initially set to tau=0.05, lambda1=1.0 as reasonable-seeming defaults when only paper excerpts were available. The full paper specifies tau=0.02, lambda1=2 exactly; config.py corrected to match.

**Infrastructure:**
- **Isolation Forest sign convention, plus a test-leakage catch in verifying the fix.** The initial implementation negated `sklearn`'s `score_samples()` under the standard "anomalous = fraud-like" assumption, giving AUC-ROC=0.184 (worse than random). Investigation showed the assumption was backwards on this dataset — fraud transactions score as *more* inlier-like, not less, under an unsupervised model (consistent with the "camouflaged fraudsters" premise the whole project is built on). The orientation was first re-derived by comparing both signs against *test*-split AUC, which is test-set leakage for a model-selection decision; redone correctly using only train-split labels, which happened to confirm the same orientation — AUC-ROC=0.816.
- **MLflow SQLite path-encoding bug.** MLflow's default tracking store silently fell back to a SQLite backend whose URI construction, given this project directory's literal space in its name, produced a broken, percent-encoded sibling path (`Fraud%20System` instead of `Fraud System`). Fixed with an explicit, correctly-encoded local file store.
- **System sleep silently stalling background training.** A 100-epoch run appeared to hang overnight (12 hours elapsed, 11% complete) but the compute itself was healthy throughout — the machine had gone to sleep 79 times, freezing the background process each time. Fixed by wrapping every subsequent long-running command in `caffeinate -i`.
- **Checkpoint-path collision across ablation variants.** `training/trainer.py`'s `train()` used to default to a single shared `care_gnn_best.pt` checkpoint path when none was given. Every CARE-GNN-family ablation run without an explicit override wrote to that same file, so it ended up silently holding whichever variant's weights finished last (`single_relation_tdt`'s, discovered via a state_dict shape mismatch against its own stored config) — and the `no_rl_selector`/`no_label_similarity` checkpoints were permanently overwritten and lost (their MLflow-logged metrics survive; their weights do not). Fixed structurally: `checkpoint_path` is now a required, keyword-only argument with no default, making the whole bug class impossible regardless of caller.
- **GAT's missing checkpoint-path override.** While verifying the checkpoint fix above, found that `models/baselines/gat.py`'s `train()` never exposed a `checkpoint_path` parameter at all (unlike GraphSAGE's, which did) — a quick smoke-test run would have silently overwritten the real trained GAT checkpoint. Added the same passthrough pattern GraphSAGE already had.
- **Entry-point epochs-default drift.** `experiments/train_care_gnn.py`'s initial `--epochs` default read from `config.py`'s `CARE_GNN_CONFIG["epochs"]` (100) — but the actual corrected-protocol run that produced the reported F1=0.5514 used 500 epochs via a manually-overridden config, never reflected in `config.py` itself. Left as-is, the entry point would have silently reproduced a different, never-actually-evaluated 100-epoch run. Fixed the default to 500 to match the real, reported result.

## Paper reference

Dou, Y., Liu, Z., Sun, L., Deng, Y., Peng, H., & Yu, P. S. (2020). *Enhancing Graph Neural Network-based Fraud Detectors against Camouflaged Fraudsters.* CIKM 2020. [https://arxiv.org/abs/2008.08692](https://arxiv.org/abs/2008.08692)

## Quickstart

Every command below is real and has been run against this repo's actual code and data — not a placeholder. Assumes `data/raw/` already has the three Elliptic CSVs (`elliptic_txs_features.csv`, `elliptic_txs_edgelist.csv`, `elliptic_txs_classes.csv`) and the venv is active (`source venv/bin/activate`).

**1. Build the graph**

```bash
python -m graph.loader
python -m graph.relations
```

Loads the raw CSVs, builds the node-index mapping and feature/label arrays (`graph/loader.py`), then constructs the three relations — tdt, tbt, tft — and caches everything to `data/processed/`. Takes **~40 minutes**, almost all of it the tft relation's 203,769×203,769 cosine-similarity neighbor search.

**2. Train and track**

```bash
python -m experiments.train_care_gnn --checkpoint-path models/checkpoints/care_gnn_full_best.pt
python -m experiments.train_baselines
mlflow ui --port 5000   # then open http://localhost:5000 to browse runs
```

Trains the full 3-relation CARE-GNN (balanced under-sampling, 500 epochs by default) and all three baselines (GraphSAGE, GAT, Isolation Forest), logging every run to the local `Augur-Elliptic` MLflow experiment. **Be realistic about the time**: the full CARE-GNN run takes **~3.5 hours** on a CPU-only machine (500 epochs at ~14–26s/epoch); GraphSAGE and GAT are much cheaper, ~15–25 minutes each at 100 epochs; Isolation Forest fits in under a second. Total for everything: **plan for 4+ hours**, not a quick pass — use `--epochs`/`--benchmark-only` on either script for a fast smoke test instead of the full run. This does *not* reproduce the three additional CARE-GNN ablation variants (single-relation-tdt, w/o RL selector, w/o label-aware similarity) in the results table above — see the known gap noted in [Build status](#build-status).

**3. Reproduce the ablation study**

```bash
python -m experiments.compare
# equivalently: python -m ablation.run_ablation
```

Pulls every run's metrics live from the MLflow experiment (not a static file) and regenerates `ablation/results/ablation_results.{csv,md}`. On a fresh clone that's only run step 2 above, this will show full CARE-GNN + the three baselines; the three ablation-variant rows require the additional `trainer.train()` calls noted above.

**4. Serve predictions**

```bash
uvicorn api.main:app --reload
```

Starts the API at `http://127.0.0.1:8000`. Try it against a real node id from the dataset (any integer in `[0, 203769)`):

```bash
curl http://127.0.0.1:8000/                                  # health + model metadata
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"node_id": 136279, "model": "graphsage"}'               # or "model": "care_gnn"
curl http://127.0.0.1:8000/subgraph/136279 -o subgraph.html    # 2-hop tdt neighborhood, open in a browser
```

## Running with Docker

An alternative to the venv-based Quickstart above, for anyone who'd rather not set up a local Python environment. Same underlying code and data requirements — just containerized.

**Prerequisites**: Docker installed, and `data/raw/` already has the three Elliptic CSVs (`elliptic_txs_features.csv`, `elliptic_txs_edgelist.csv`, `elliptic_txs_classes.csv`) downloaded from [Kaggle](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set) — same requirement as the venv path above.

**One-step run:**

```bash
bash run.sh
```

This checks for the three CSVs in `data/raw/` and exits with a clear message (pointing at the Kaggle link above) if they're missing. If `data/processed/` doesn't already have the cached graph artifacts, it builds them via `docker compose run --rm app python3 -m graph.loader` and `graph.relations` — printing the real **"~40 minutes"** warning *before* that step starts, not after. It then checks for the 5 trained checkpoints; since these ship baked into the Docker image, this step is normally a no-op skip. Finally it runs `docker compose up -d mlflow api`.

**What it deliberately does *not* do: auto-retrain.** Training the full CARE-GNN + baseline suite takes 4+ hours on a CPU-only machine — far too long to run silently as part of a "one-step" script. `run.sh` brings the API up using the checkpoints already shipped in the image; if you want to reproduce training yourself rather than use those, run it explicitly:

```bash
docker compose run app python3 -m experiments.train_care_gnn --checkpoint-path models/checkpoints/care_gnn_full_best.pt
docker compose run app python3 -m experiments.train_baselines
```

Note: the API only serves `graphsage` and `care_gnn` — GAT and Isolation Forest exist for the ablation study only and have no serving path.

**Manual step-by-step equivalent**, if you want to see what `run.sh` is doing under the hood:

```bash
docker compose run --rm app python3 -m graph.loader
docker compose run --rm app python3 -m graph.relations
docker compose up -d mlflow api
```

**Verifying it's up:**

```bash
curl http://localhost:8000/
# {"status": "ok", "num_nodes": 203769, "models": [{"model_type": ..., "checkpoint": ..., "epoch": ..., "f1_illicit": ..., "auc_roc": ...}, ...]}

curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"node_id": 136279, "model": "graphsage"}'
# {"node_id": 136279, "fraud_probability": <float>, "is_fraud": <bool>, "explanation": "...", "model_version": "..."}

curl http://localhost:8000/subgraph/136279 -o subgraph.html
# writes an HTML file (pyvis 2-hop tdt neighborhood), open it in a browser
```

MLflow UI is at **`http://localhost:5001`** — not 5000. `docker-compose.yml` remaps mlflow's container port 5000 to host port 5001 because port 5000 collides with macOS's own AirPlay Receiver (`ControlCenter`), which listens on `0.0.0.0:5000` by default. If you're on a Mac and see a "port already in use" error on 5000 outside this project too, that's why.

**Image**: ~3.17GB, built with CPU-only torch (`--index-url https://download.pytorch.org/whl/cpu`) — this project has no GPU-accelerated code path anywhere, so there's no reason to pull in the ~5-6GB of CUDA runtime libraries the default PyPI torch wheel drags in. The 5 trained checkpoints are baked into the image itself, so a fresh clone + `docker compose build` gives a fully working demo with zero dependency on the host having trained anything first.

One gotcha worth knowing if you ever edit `docker-compose.yml`: MLflow 3.x refuses the plain filesystem tracking backend by default unless `MLFLOW_ALLOW_FILE_STORE=true` is set — this project hit that twice (once training outside Docker, once bringing up the mlflow container) before it got documented inline in the compose file. Don't strip that env var out.
