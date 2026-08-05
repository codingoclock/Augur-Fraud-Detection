"""Unit tests for training/loss.py (Module 4, v3): WeightedFocalLoss
(unchanged from v2, newly written since the file was previously an empty
stub), SimilarityAuxLoss (new -- Eq. 4's numerically-stable reformulation),
and care_gnn_total_loss (new -- Eq. 11's combination).
"""

import math

import torch

from training.loss import WeightedFocalLoss, SimilarityAuxLoss, care_gnn_total_loss


# ---------------------------------------------------------------------------
# WeightedFocalLoss
# ---------------------------------------------------------------------------


def test_focal_loss_matches_hand_computed_value_for_balanced_case():
    """Exact hand-computed value.

    Uniform class weights [1,1] (reduces to plain focal loss). Single
    sample, equal logits [0,0] for a 2-class problem -> softmax = [0.5,0.5]
    -> CE = -log(0.5) = 0.6931471805599453. pt = exp(-CE) = 0.5.
    focal = (1-pt)^2 * CE = 0.25 * 0.6931471805599453 = 0.17328679513998632.
    """
    loss_fn = WeightedFocalLoss(class_weights=torch.tensor([1.0, 1.0]), gamma=2.0)
    logits = torch.tensor([[0.0, 0.0]])
    targets = torch.tensor([0])

    loss = loss_fn(logits, targets)
    assert abs(loss.item() - 0.17328679513998632) < 1e-6


def test_focal_loss_excludes_unlabelled_nodes():
    """Exact hand-computed value: a -1 (unlabelled) target must be fully
    excluded from the mean, not treated as class 0 or contribute 0 loss."""
    loss_fn = WeightedFocalLoss(class_weights=torch.tensor([1.0, 1.0]), gamma=2.0)
    logits = torch.tensor([[0.0, 0.0], [100.0, -100.0]])  # 2nd row: confident, irrelevant if excluded
    targets = torch.tensor([0, -1])

    loss = loss_fn(logits, targets)
    # Only row 0 survives the mask -> identical to the single-sample case above.
    assert abs(loss.item() - 0.17328679513998632) < 1e-6


# ---------------------------------------------------------------------------
# SimilarityAuxLoss
# ---------------------------------------------------------------------------


def test_aux_loss_matches_hand_computed_value_for_agreement():
    """Exact hand-computed value.
    pred=0.6, y=+1 -> agreement = (1*0.6 + 1)/2 = 0.8 -> loss = -log(0.8)
    = 0.2231435513142097.
    """
    loss_fn = SimilarityAuxLoss()
    pred = torch.tensor([0.6])
    y_signed = torch.tensor([1.0])
    mask = torch.tensor([True])

    loss = loss_fn(pred, y_signed, mask)
    assert abs(loss.item() - 0.2231435513142097) < 1e-6


def test_aux_loss_matches_hand_computed_value_for_sign_disagreement():
    """Exact hand-computed value -- the required "y_v and prediction
    disagree in sign" edge case.
    pred=0.6 (positive), y=-1 (opposite sign) -> agreement =
    (-1*0.6 + 1)/2 = 0.2 -> loss = -log(0.2) = 1.6094379124341003.
    Confirms no NaN/Inf: the raw Eq. 4 (-log(y*tanh(pred))) would take
    -log(-0.6), undefined; the reformulation instead gives a large but
    finite, well-defined loss.
    """
    loss_fn = SimilarityAuxLoss()
    pred = torch.tensor([0.6])
    y_signed = torch.tensor([-1.0])
    mask = torch.tensor([True])

    loss = loss_fn(pred, y_signed, mask)
    assert torch.isfinite(loss)
    assert abs(loss.item() - 1.6094379124341003) < 1e-6


def test_aux_loss_extreme_disagreement_is_clamped_not_inf():
    """Exact hand-computed value at the eps clamp boundary.
    pred=1.0 (tanh-saturated), y=-1 -> raw agreement = (-1*1 + 1)/2 = 0.0
    exactly, which would make -log(0.0) = inf without the clamp. With
    eps=1e-7 clamping agreement to exactly eps, loss = -log(1e-7) =
    16.11809565095832 -- large, but finite.
    """
    loss_fn = SimilarityAuxLoss(eps=1e-7)
    pred = torch.tensor([1.0])
    y_signed = torch.tensor([-1.0])
    mask = torch.tensor([True])

    loss = loss_fn(pred, y_signed, mask)
    assert torch.isfinite(loss)
    assert abs(loss.item() - 16.11809565095832) < 1e-4


def test_aux_loss_mask_excludes_unlabelled_nodes_from_mean():
    """Exact hand-computed value: only masked-in (labelled) nodes should
    contribute to the mean; a masked-out node's pred/y values are decoys
    (deliberately chosen to blow up the loss if wrongly included) and must
    not affect the result."""
    loss_fn = SimilarityAuxLoss()
    # node 0: labelled, pred=0.6, y=+1 -> agreement 0.8 -> -log(0.8)
    # node 1: UNLABELLED (masked out) -- decoy values that would produce a
    #   huge loss (near-total disagreement) if incorrectly included
    pred = torch.tensor([0.6, -0.999999])
    y_signed = torch.tensor([1.0, 1.0])
    mask = torch.tensor([True, False])

    loss = loss_fn(pred, y_signed, mask)
    assert abs(loss.item() - 0.2231435513142097) < 1e-6


def test_aux_loss_gradient_flows_through_pred():
    """Existence/finiteness check, not an exact value."""
    loss_fn = SimilarityAuxLoss()
    pred = torch.tensor([0.3, -0.2, 0.9], requires_grad=True)
    y_signed = torch.tensor([1.0, -1.0, 1.0])
    mask = torch.tensor([True, True, True])

    loss = loss_fn(pred, y_signed, mask)
    loss.backward()

    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()


# ---------------------------------------------------------------------------
# care_gnn_total_loss
# ---------------------------------------------------------------------------


def test_total_loss_combines_gnn_and_aux_with_lambda1_exactly():
    """Exact hand-computed value: total = l_gnn + lambda1 * l_simi, and the
    returned breakdown dict must report the same two component values."""
    focal_loss_fn = WeightedFocalLoss(class_weights=torch.tensor([1.0, 1.0]), gamma=2.0)
    aux_loss_fn = SimilarityAuxLoss()

    gnn_logits = torch.tensor([[0.0, 0.0]])
    gnn_targets = torch.tensor([0])
    layer1_pred = torch.tensor([0.6])
    y_signed = torch.tensor([1.0])
    labelled_mask = torch.tensor([True])
    lambda1 = 2.0

    total, breakdown = care_gnn_total_loss(
        gnn_logits, gnn_targets, layer1_pred, y_signed, labelled_mask,
        focal_loss_fn, aux_loss_fn, lambda1,
    )

    expected_l_gnn = 0.17328679513998632
    expected_l_simi = 0.2231435513142097
    expected_total = expected_l_gnn + lambda1 * expected_l_simi

    assert abs(breakdown["l_gnn"] - expected_l_gnn) < 1e-6
    assert abs(breakdown["l_simi"] - expected_l_simi) < 1e-6
    assert abs(total.item() - expected_total) < 1e-6
