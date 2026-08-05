"""Loss functions for CARE-GNN (Dou et al. CIKM 2020, Eq. 4, 10, 11).

Note: WeightedFocalLoss did not previously exist in this repo (training/loss.py
was an empty stub) despite the v3 build spec describing it as "the existing
WeightedFocalLoss (unchanged)" -- it is written here for the first time,
using the spec's code verbatim (unchanged from v2, since v3's corrections
were only to Modules 1-3's mechanisms, not this one).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedFocalLoss(nn.Module):
    """
    Main classification loss (stands in for the paper's plain Eq. 10
    cross-entropy). This is a DELIBERATE, DOCUMENTED deviation: the paper's
    L_GNN = -log(y_v * sigma(MLP(z_v))) has no explicit class-weighting term,
    and Elliptic's imbalance (train ~11.6% fraud among labelled, test ~6.5%)
    is severe enough that an unweighted loss risks collapsing toward
    predicting the majority class. Focal loss + class weighting is a
    standard, defensible adaptation for this specific dataset's imbalance --
    keep it, but document it as a deviation from Eq. 10, not as "the" paper
    loss.
    """

    def __init__(self, class_weights: torch.Tensor, gamma: float = 2.0):
        super().__init__()
        self.class_weights = class_weights
        self.gamma = gamma

    def forward(self, logits, targets):
        # Only use labelled nodes (exclude -1, the {1,0,-1} convention's
        # "unlabelled" marker)
        mask = targets != -1
        logits, targets = logits[mask], targets[mask]
        ce = F.cross_entropy(logits, targets, weight=self.class_weights, reduction="none")
        pt = torch.exp(-ce)
        focal = ((1 - pt) ** self.gamma) * ce
        return focal.mean()


class SimilarityAuxLoss(nn.Module):
    """
    Eq. 4, LAYER 1 ONLY (paper: "we only use the similarity measure loss at
    the first layer"). Do not sum this over all layers, and do not call it
    with any layer other than similarities[0]'s output -- see
    models/care_gnn/similarity.py's module docstring for why every
    subsequent layer's MLP is intentionally never trained.

    Numerical-stability note: the paper's literal Eq. 4 is
    -log(y_v * sigma(MLP(h_v))) with sigma=tanh. Taking this literally is
    unsafe: tanh outputs are in (-1, 1), y_v in {-1, +1} (a SEPARATE signed
    encoding from this codebase's {1, 0, -1} label convention used
    everywhere else -- do not conflate the two, see config.py's LABEL_MAP
    comment), and whenever the prediction disagrees with the sign of y_v,
    y_v * tanh(pred) is negative and log() is undefined. Use a numerically
    stable equivalent instead: map (y_v * tanh(pred)) from (-1, 1) to (0, 1)
    via (1 + y_v*tanh(pred)) / 2 before taking -log, with a small epsilon
    clamp. This reformulation is an implementation detail, not a change to
    what's being optimized -- it preserves the same minimum (agreement=1
    when y_v and pred fully agree in sign and magnitude).
    """

    def __init__(self, eps: float = 1e-7):
        super().__init__()
        self.eps = eps

    def forward(self, layer1_pred: torch.Tensor, y_signed: torch.Tensor, mask: torch.Tensor):
        # layer1_pred: [N] tanh output from similarities[0]
        # y_signed: [N] labels remapped to {-1, +1}; caller must exclude
        #   unlabelled (-1 in the {1,0,-1} convention) nodes via `mask`
        #   BEFORE this remapping, since -1 means two different things in
        #   the two conventions and that collision must not leak into this
        #   function
        pred = layer1_pred[mask]
        y = y_signed[mask]
        agreement = (y * pred + 1.0) / 2.0  # remapped to (0, 1)
        agreement = agreement.clamp(min=self.eps, max=1.0 - self.eps)
        return -torch.log(agreement).mean()


def care_gnn_total_loss(
    gnn_logits, gnn_targets, layer1_pred, y_signed, labelled_mask,
    focal_loss_fn: WeightedFocalLoss, aux_loss_fn: SimilarityAuxLoss, lambda1: float,
):
    """
    Eq. 11: L_CARE = L_GNN + lambda1 * L_Simi^(1) + lambda2 * ||Theta||^2.
    lambda2 * L2 term is intentionally omitted here -- use the optimizer's
    `weight_decay` parameter instead (already in config.py's
    CARE_GNN_CONFIG["weight_decay"]) rather than computing ||Theta||^2
    manually in the loss. Applying both would double-regularize; pick one,
    and the optimizer's built-in mechanism is the simpler, standard choice.
    """
    l_gnn = focal_loss_fn(gnn_logits, gnn_targets)
    l_simi = aux_loss_fn(layer1_pred, y_signed, labelled_mask)
    return l_gnn + lambda1 * l_simi, {"l_gnn": l_gnn.item(), "l_simi": l_simi.item()}
