# Augur Ablation Results

All CARE-GNN variants share identical hyperparameters (lr, weight_decay, hidden_dim, batch_size, seed) except the ablated component. GraphSAGE/GAT trained on the merged tdt+tbt+tft graph (fairness decision, see models/baselines/graphsage.py). Isolation Forest is tabular-only, unsupervised (no epoch concept).

**Important -- read before comparing rows:** `CARE-GNN w/o RL selector` and `CARE-GNN w/o label-aware similarity` were trained for only **10 epochs**, not the 100 used for every other CARE-GNN variant, due to hardware/compute constraints on this machine (each would take ~13-16 hours at full length). Their rows below reflect the best result achievable within that 10-epoch budget -- they are **not** directly comparable epoch-for-epoch to the 100-epoch full model's result. See the second table for the fair, same-epoch-budget comparison.

## Main results (sorted by F1_illicit, best achieved within each variant's own budget)

| Variant | F1_illicit | AUC-ROC | Precision | Recall | Best epoch | Epochs trained/budget |
|---|---|---|---|---|---|---|
| CARE-GNN single relation (tdt only) | 0.5713 | 0.9003 | 0.5025 | 0.6620 | 35 | 35/100 |
| GraphSAGE | 0.5301 | 0.9030 | 0.4206 | 0.7165 | 90 | 90/100 |
| CARE-GNN (full) | 0.4587 | 0.9002 | 0.3411 | 0.6999 | 90 | 90/100 |
| GAT | 0.2595 | 0.8842 | 0.1518 | 0.8929 | 85 | 85/100 |
| CARE-GNN w/o RL selector (fixed p=0.5) | 0.2375 | 0.8770 | 0.1365 | 0.9141 | 10 | 10/10 ⚠️ capped |
| CARE-GNN w/o label-aware similarity (λ1=0, redefined for v3 -- see note) | 0.1930 | 0.8366 | 0.1076 | 0.9354 | 10 | 10/10 ⚠️ capped |
| Isolation Forest | 0.1329 | 0.8162 | 0.0712 | 1.0000 | - | None/None |

## Same-epoch-budget reference: full CARE-GNN vs. the two 10-epoch variants, all @ epoch 10

This is the fair comparison for the two capped variants above -- full CARE-GNN's *best* (epoch 90) is a different, later point in its own training and should not be read as "what full CARE-GNN looked like at 10 epochs."

| Variant | F1_illicit | AUC-ROC | Precision | Recall | Epoch | Note |
|---|---|---|---|---|---|---|
| CARE-GNN (full model) @ epoch 10 | 0.2603 | 0.8715 | 0.1527 | 0.8800 | 10 | transcript-recorded console output, NOT re-verifiable (no epoch-10 checkpoint was ever saved) |
| CARE-GNN w/o RL selector @ epoch 10 | 0.2375 | 0.8770 | 0.1365 | 0.9141 | 10 | MLflow-retrieved, epoch 10 (this variant's only full budget) |
| CARE-GNN w/o label-aware similarity @ epoch 10 | 0.1930 | 0.8366 | 0.1076 | 0.9354 | 10 | MLflow-retrieved, epoch 10 (this variant's only full budget) |

**Note on the label-aware-similarity variant's redefinition:** the original build spec's "w/o label-aware similarity" ablation was designed against v1/v2's cosine+label-flag similarity module, which no longer exists in v3. v3's similarity module is a per-layer MLP trained via an auxiliary loss (Eq. 4) with no "drop the label term, keep a feature-only term" option -- label-training IS the mechanism. The variant here instead sets `lambda1=0`, disabling that auxiliary loss entirely so the layer-1 MLP never receives gradient and stays at random initialization all run, used for top-p filtering but never shaped by label information. This tests a different (but analogous) hypothesis than the original spec text describes, and the two should not be treated as equivalent if compared against documentation written for v1/v2.
