# Augur Ablation Results

All rows except Isolation Forest were trained/retrained under the **corrected protocol**: balanced under-sampling per batch (equal fraud/licit train-labelled centers, freshly resampled each batch), tau=0.02, lambda1=2 (paper-verified values, not the earlier placeholder tau=0.05/lambda1=1.0). All four CARE-GNN family rows (full, single-relation-tdt, w/o RL selector, w/o label-aware similarity) used an identical **500-epoch** budget. GraphSAGE/GAT used 100 epochs (their original, already-sufficient budget) on the same merged tdt+tbt+tft graph and balanced-sampling protocol. **Isolation Forest is not under the corrected protocol** -- it's unsupervised and fit only on features; batch-level under-sampling has no meaning for it, so its original numbers stand.

## Results (sorted by F1_illicit)

| Variant | F1_illicit | AUC-ROC | Precision | Recall | Best epoch | Epochs trained |
|---|---|---|---|---|---|---|
| CARE-GNN single relation (tdt only) | 0.6603 | 0.8970 | 0.7244 | 0.6066 | 260 | 500 |
| GraphSAGE | 0.6434 | 0.9059 | 0.7114 | 0.5873 | 90 | 100 |
| CARE-GNN w/o RL selector (fixed p=0.5) | 0.5829 | 0.8626 | 0.5708 | 0.5956 | 500 | 500 |
| CARE-GNN (full) | 0.5514 | 0.8528 | 0.5342 | 0.5697 | 120 | 500 |
| CARE-GNN w/o label-aware similarity (λ1=0, redefined for v3 -- see note) | 0.4984 | 0.8548 | 0.4089 | 0.6380 | 400 | 500 |
| GAT | 0.3266 | 0.8862 | 0.2045 | 0.8098 | 90 | 100 |
| Isolation Forest | 0.1329 | 0.8162 | 0.0712 | 1.0000 | - | None (original protocol, see note) |

**Note on the label-aware-similarity variant's redefinition:** the original build spec's "w/o label-aware similarity" ablation was designed against v1/v2's cosine+label-flag similarity module, which no longer exists in v3. v3's similarity module is a per-layer MLP trained via an auxiliary loss (Eq. 4) with no "drop the label term, keep a feature-only term" option -- label-training IS the mechanism. The variant here instead sets `lambda1=0`, disabling that auxiliary loss entirely so the layer-1 MLP never receives gradient and stays at random initialization all run, used for top-p filtering but never shaped by label information. This tests a different (but analogous) hypothesis than the original spec text describes, and the two should not be treated as equivalent if compared against documentation written for v1/v2.

**Note on Isolation Forest:** kept at its original, never-corrected numbers (unsupervised, fit only on tabular features -- balanced under-sampling has no applicable meaning for its training procedure).
