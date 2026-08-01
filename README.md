# Augur

**Status: early build — graph construction complete, model implementation not started.** This is a progress checkpoint, not a finished project. See [Build status](#build-status) for exactly what exists right now.

Augur is a heterogeneous graph neural network system for detecting fraud rings in transaction data — using CARE-GNN (Dou et al., CIKM 2020) to model relational structure that row-by-row classifiers can't see, instead of scoring transactions one at a time.

## Architecture

Fraud rings coordinate: individual transactions can look unremarkable while the network around them doesn't. A standard node classifier (or an Isolation Forest scoring rows independently) is architecturally blind to that — it never sees the graph. CARE-GNN is built for exactly this: it aggregates over multiple relation types with a learned, label-aware neighbor selector, so the model can down-weight neighbors that don't help distinguish fraud from non-fraud, instead of averaging blindly like a standard GNN (GraphSAGE/GAT) would.

The dataset is the [Elliptic Bitcoin Dataset](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set) — 203,769 transaction nodes with 165 features each, spanning 49 time steps, with ~2% labeled illicit, ~21% labeled licit, and the rest unlabeled. It ships with one native edge relation; Augur constructs three:

| Relation | Meaning | Edge count |
|---|---|---|
| **tdt** (transaction–direction–transaction) | Native directed edges from the dataset | 234,355 |
| **tbt** (transaction–block–transaction) | Transactions sharing a time step (capped at 500 sampled pairs/block) | 49,000 |
| **tft** (transaction–feature–transaction) | Top-10 nearest neighbors per node by cosine similarity on L2-normalized features, threshold > 0.85 | 1,701,497 |

These counts are read directly from the generated `data/processed/adj_*.pkl` arrays (verified via a Python session against this repo's actual output, not copied from a spec), and are also asserted by the test suite (`tests/test_graph_construction.py`).

## Design decision: standalone, not integrated

CARE-GNN needs a fully-constructed batch graph before it can train — it isn't a streaming, per-event scorer. Wiring it into a Kafka-based pipeline would mean faking graph construction on a stream, which is architecturally dishonest and falls apart under scrutiny. Augur is deliberately a separate, standalone project from any real-time anomaly system, because the two solve different problems with different data paradigms (stream vs. graph batch).

> For row-by-row anomaly scoring on tabular transaction data, see the Realtime Anomaly Engine repo. Augur is its architectural successor for relational fraud — it detects rings that isolated-row scoring can't.

## Build status

_Last checked: 2026-08-01, against this repo's actual working tree and a live pytest run._

- [x] **Environment setup** — Python 3.12.13 in a local venv; `torch` 2.13.0, `torch-geometric` 2.8.0, `scikit-learn` 1.9.0, `mlflow` 3.15.0, `fastapi` 0.141.1, `pyvis` 0.3.2, and the rest of `requirements.txt` installed and importable (verified via `pip freeze`).
- [x] **Data loading + multi-relation graph construction** (`graph/loader.py`, `graph/relations.py`, `graph/dataset.py`) — implemented and tested. `pytest tests/test_graph_construction.py` passes 6/6 (node-index bijection, tdt edge count vs. raw CSV, tbt per-block cap, tft top-k cap, feature matrix shape, and same-seed reproducibility). Full run takes ~38 minutes, almost entirely spent computing the 203,769×203,769 cosine-similarity neighbor search for tft.
- [ ] **CARE-GNN model** (`models/care_gnn/similarity.py`, `selector.py`, `aggregator.py`, `care_gnn.py`) — not started. All four files exist as empty (0-byte) stubs in the repo; none of the label-aware similarity, RL neighbor selector, relation-aware aggregator, or full model assembly has been written yet.
- [ ] **Baselines** (`models/baselines/graphsage.py`, `gat.py`, `isolation_forest.py`) — not started, empty stubs.
- [ ] **Training loop + evaluator** (`training/trainer.py`, `evaluator.py`, `loss.py`) — not started, empty stubs.
- [ ] **Ablation study** (`ablation/run_ablation.py`) — not started, empty stub.
- [ ] **Visualization** (`visualization/fraud_rings.py`, `embedding_umap.py`) — not started, empty stubs.
- [ ] **API** (`api/main.py`, `schemas.py`, `routes/`) — not started, empty stubs.

No model has been trained, so there are no results, no ablation table, and no fraud-ring visualization yet — those sections will be added once real experiments run, with real numbers only.

## Notable engineering decisions so far

These come from the graph-construction step, the only part of the system with real code behind it.

- **tft top-k cap and self-exclusion under near-duplicates.** Naively excluding a node's own index from its k+1 nearest neighbors (`indices != self`) breaks when many feature vectors are near-identical: sklearn's tie-breaking can drop a node's own index from its returned neighbor set entirely, which would silently leave k+1 neighbors instead of k. The fix ranks neighbors by ascending distance and takes the first k non-self columns positionally, which caps at exactly k regardless of whether self appears in the returned set.
- **`ball_tree` → `brute` deviation from the original plan.** The original approach called for `sklearn.neighbors.NearestNeighbors` with `algorithm='ball_tree'`. Two problems: ball_tree doesn't support `metric='cosine'` at all, and an empirical run using ball_tree with a converted metric over the full 165-dimensional, 203,769-row feature matrix was killed after running 41 minutes at 97% CPU without finishing — at that dimensionality, ball_tree's pruning collapses toward brute-force behavior anyway, so it adds tree-construction overhead for no speed benefit. Switched to `algorithm='brute'` with `metric='cosine'`, which computes the same exact top-k via chunked BLAS matrix multiplication and actually scales.
- **Feature column misalignment.** The raw `elliptic_txs_features.csv` has 167 columns (txId, time_step, then 165 real features). Computing `feature_dim` as "raw columns minus txId" gives 166 and silently mixes `time_step` into the feature matrix. `graph/loader.py` reads feature columns explicitly from config (`CARE_GNN_CONFIG["feature_dim"] = 165`) rather than deriving it from the raw column count.

## Paper reference

Dou, Y., Liu, Z., Sun, L., Deng, Y., Peng, H., & Yu, P. S. (2020). *Enhancing Graph Neural Network-based Fraud Detectors against Camouflaged Fraudsters.* CIKM 2020. [https://arxiv.org/abs/2008.08692](https://arxiv.org/abs/2008.08692)

## What's next

1. Implement `models/care_gnn/similarity.py` (label-aware similarity) and unit test it against synthetic tensors.
2. Implement `models/care_gnn/selector.py` (RL-based neighbor selector) and unit test it.
3. Implement `models/care_gnn/aggregator.py` (relation-aware aggregator), then assemble the full model in `care_gnn.py` and verify a forward pass on the real graph.
4. Build the training loop (`training/trainer.py`, `evaluator.py`) and run a first real training pass to get baseline numbers before touching baselines, ablations, or the API.
