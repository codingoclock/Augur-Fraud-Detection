"""Unit tests for models/care_gnn/selector.py (Module 2, v3).

Covers both halves of SimilarityAwareSelector: the top-p forward filtering
(2a, unchanged mechanism from v2) and the RL threshold update rule (2b,
rewritten entirely in v3 -- fixed tau, direct reward-sign action, random
first-epoch action, Eq. 7 terminal freeze). Neither half had dedicated tests
before this file (tests/test_selector_rl.py previously existed only as an
empty stub); both are written fresh here against the real v3 module.
"""

import random

import torch

from models.care_gnn.selector import SimilarityAwareSelector, label_aware_distance


# ---------------------------------------------------------------------------
# 2a: top-p forward filtering (mechanism unchanged from v2)
# ---------------------------------------------------------------------------


def test_forward_keeps_exact_top_p_fraction_by_score():
    """Exact hand-computed value.

    Row 0: 4 valid neighbours, scores [0.9, 0.1, 0.5, 0.3], p=0.5 ->
    keep_counts = ceil(4*0.5) = 2 -> the top 2 by score are index 0 (0.9)
    and index 2 (0.5).
    Row 1: only 2 valid neighbours (mask [T,T,F,F]); the invalid slots carry
    deliberately huge decoy scores (999) that must be excluded regardless of
    their value. p=0.5 -> keep_counts = ceil(2*0.5) = 1 -> among the 2 valid
    scores [0.2, 0.8], keep only index 1 (0.8).
    """
    selector = SimilarityAwareSelector(num_relations=1, init_p=0.5)

    similarity_scores = torch.tensor([
        [0.9, 0.1, 0.5, 0.3],
        [0.2, 0.8, 999.0, 999.0],
    ])
    neighbor_mask = torch.tensor([
        [True, True, True, True],
        [True, True, False, False],
    ])

    selected = selector(similarity_scores, 0, neighbor_mask)

    expected = torch.tensor([
        [True, False, True, False],
        [False, True, False, False],
    ])
    assert torch.equal(selected, expected)


def test_forward_never_selects_masked_out_slots_regardless_of_p():
    """Exact hand-computed value: a node with zero valid neighbours must
    return an all-False row for any p, including the internal keep_counts
    clamp(min=1) -- the clamp is a harmless no-op here because the final
    result is ANDed with neighbor_mask (all False), never a fabricated
    selection.
    """
    selector = SimilarityAwareSelector(num_relations=1, init_p=0.9)
    similarity_scores = torch.tensor([[5.0, 5.0, 5.0]])
    neighbor_mask = torch.tensor([[False, False, False]])

    selected = selector(similarity_scores, 0, neighbor_mask)
    assert torch.equal(selected, torch.zeros(1, 3, dtype=torch.bool))


def test_forward_p_equals_one_keeps_all_valid_neighbours():
    """Exact hand-computed value: p=1.0 -> keep_counts == valid_counts for
    every row, so the result equals neighbor_mask exactly."""
    selector = SimilarityAwareSelector(num_relations=1, init_p=1.0)
    similarity_scores = torch.randn(3, 5)
    neighbor_mask = torch.tensor([
        [True, True, True, False, False],
        [True, False, False, False, False],
        [True, True, True, True, True],
    ])

    selected = selector(similarity_scores, 0, neighbor_mask)
    assert torch.equal(selected, neighbor_mask)


# ---------------------------------------------------------------------------
# 2b: RL threshold update (v3 fixed-tau rule)
# ---------------------------------------------------------------------------


def test_rl_update_matches_hand_computed_sequence_exactly():
    """Exact hand-computed value.

    Distance sequence [0.5, 0.3, 0.4, 0.2, 0.35, 0.15] (the exact sequence
    named in the build spec's own testing requirement), init_p=0.5, tau=0.05,
    seed=42.

    Epoch 1: no reference distance yet -> random +-tau action. seed=42's
    first draw from `random.Random(42).random()` is 0.6394... (verified
    independently below, not assumed) which is >= 0.5, so per
    `tau if draw < 0.5 else -tau` the epoch-1 action is -tau.
    p1 = 0.5 - 0.05 = 0.45

    Epoch 2: distance 0.3 < prev 0.5 -> improved -> reward +1 -> action +tau.
    p2 = 0.45 + 0.05 = 0.50

    Epoch 3: distance 0.4 > prev 0.3 -> not improved -> reward -1 -> -tau.
    p3 = 0.50 - 0.05 = 0.45

    Epoch 4: distance 0.2 < prev 0.4 -> improved -> +tau.
    p4 = 0.45 + 0.05 = 0.50

    Epoch 5: distance 0.35 > prev 0.2 -> not improved -> -tau.
    p5 = 0.50 - 0.05 = 0.45

    Epoch 6: distance 0.15 < prev 0.35 -> improved -> +tau.
    p6 = 0.45 + 0.05 = 0.50
    """
    # Confirm the seed=42 first draw independently, not assumed.
    assert random.Random(42).random() >= 0.5

    selector = SimilarityAwareSelector(num_relations=1, init_p=0.5, tau=0.05, seed=42)
    distances = [0.5, 0.3, 0.4, 0.2, 0.35, 0.15]
    expected_p = [0.45, 0.50, 0.45, 0.50, 0.45, 0.50]

    actual_p = [selector.rl_step_relation(0, d, epoch) for epoch, d in enumerate(distances, start=1)]

    for a, e in zip(actual_p, expected_p):
        assert abs(a - e) < 1e-9, f"expected {expected_p}, got {actual_p}"


def test_first_epoch_action_is_random_but_bounded_to_tau():
    """Exact hand-computed value, for two different seeds chosen to land on
    opposite sides of the module's `draw < 0.5` branch (verified
    independently against plain `random.Random`, not assumed):
      seed=1  -> random.Random(1).random()  ~= 0.1344 < 0.5 -> action = +tau
      seed=42 -> random.Random(42).random() ~= 0.6394 >= 0.5 -> action = -tau
    Both cases confirm the first-epoch action is bounded to exactly +-tau
    (never any other magnitude), and that it actually varies with the seed
    (i.e. is genuinely the random branch, not a hardcoded constant).
    """
    assert random.Random(1).random() < 0.5
    assert random.Random(42).random() >= 0.5

    tau = 0.05
    init_p = 0.5

    sel_plus = SimilarityAwareSelector(num_relations=1, init_p=init_p, tau=tau, seed=1)
    p_plus = sel_plus.rl_step_relation(0, distance=0.5, epoch=1)
    assert abs(p_plus - (init_p + tau)) < 1e-9

    sel_minus = SimilarityAwareSelector(num_relations=1, init_p=init_p, tau=tau, seed=42)
    p_minus = sel_minus.rl_step_relation(0, distance=0.5, epoch=1)
    assert abs(p_minus - (init_p - tau)) < 1e-9

    # And no reward is recorded on epoch 1 (no reference distance to compare against).
    assert len(sel_plus.get_state(0).reward_history) == 0
    assert len(sel_minus.get_state(0).reward_history) == 0


def test_sustained_losing_streak_does_not_terminate():
    """Exact hand-computed value + behavioural check.

    Eq. 7 (corrected): |sum_{e-10}^{e} reward| <= 2, window inclusive =
    11 terms. Distances strictly increase every epoch after epoch 1 (by
    1.0 each time), so every epoch 2-12 has `improved = False` -> reward
    = -1 every time -- a sustained LOSING streak, not a balanced one.

    By epoch 12, reward_history holds exactly the last 11 rewards (epochs
    2-12), all -1, summing to -11. |sum| = 11 > terminal_threshold (2.0)
    -> Eq. 7's condition is NOT satisfied -> the selector must NOT
    terminate, despite 11 full epochs of reward history existing. This is
    the corrected rule's whole point: a raw-sum check (pre-correction) would
    have wrongly frozen here too (-11 <= 2 is True), conflating "reliably
    getting worse" with "balanced/plateaued".

    tau=0.05, init_p=0.5. Epoch 1 is the random draw (seed=42 gives -tau,
    verified independently below); epochs 2-12 are then all -tau (11 of
    them, clamped into [0, 1]):
      p0=0.50 -> 0.45 (ep1) -> 0.40 (ep2) -> 0.35 (ep3) -> 0.30 (ep4)
      -> 0.25 (ep5) -> 0.20 (ep6) -> 0.15 (ep7) -> 0.10 (ep8) -> 0.05 (ep9)
      -> 0.00 (ep10, exact floor) -> 0.00 (ep11, clamped)
      -> 0.00 (ep12, clamped)
    """
    assert random.Random(42).random() >= 0.5  # epoch-1 action is -tau here too

    selector = SimilarityAwareSelector(
        num_relations=1, init_p=0.5, tau=0.05, seed=42,
        terminal_window=10, terminal_threshold=2.0,  # window_size = 11
    )

    distances = [1.0] + [1.0 + 1.0 * i for i in range(1, 12)]  # epochs 1..12, strictly increasing
    expected_p = [0.45, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15, 0.10, 0.05, 0.00, 0.00, 0.00]

    for epoch, (d, exp_p) in enumerate(zip(distances, expected_p), start=1):
        p = selector.rl_step_relation(0, d, epoch)
        assert abs(p - exp_p) < 1e-9, f"epoch {epoch}: expected {exp_p}, got {p}"
        assert selector.get_state(0).terminated is False, f"wrongly terminated at epoch {epoch}"

    state = selector.get_state(0)
    assert len(state.reward_history) == 11
    assert sum(state.reward_history) == -11.0
    assert abs(sum(state.reward_history)) > 2.0
    assert state.terminated is False


def test_balanced_oscillation_terminates_and_freezes_p_permanently():
    """Exact hand-computed value + behavioural check.

    Distances alternate worse/better every epoch after epoch 1
    (1.0, 2.0, 1.5, 2.5, 2.0, 3.0, 2.5, 3.5, 3.0, 4.0, 3.5, 4.5), producing
    an alternating reward sequence starting and ending with -1 across
    epochs 2-12: [-1,+1,-1,+1,-1,+1,-1,+1,-1,+1,-1] (6 negative, 5
    positive terms). Sum = 6*(-1) + 5*(+1) = -1. |sum| = 1 <=
    terminal_threshold (2.0) -> Eq. 7's condition IS satisfied at epoch 12
    (the first epoch where the 11-term window is fully populated) -> the
    selector terminates there, not earlier.

    tau=0.05, init_p=0.5. Epoch 1 is the random draw (seed=42 -> -tau,
    verified independently below); actions then alternate -tau/+tau in
    lockstep with the reward sequence:
      p0=0.50 -> 0.45 (ep1, -tau)
      -> 0.40 (ep2, reward -1) -> 0.45 (ep3, +1) -> 0.40 (ep4, -1)
      -> 0.45 (ep5, +1) -> 0.40 (ep6, -1) -> 0.45 (ep7, +1)
      -> 0.40 (ep8, -1) -> 0.45 (ep9, +1) -> 0.40 (ep10, -1)
      -> 0.45 (ep11, +1) -> 0.40 (ep12, -1, THEN terminates)
    """
    assert random.Random(42).random() >= 0.5  # epoch-1 action is -tau here too

    selector = SimilarityAwareSelector(
        num_relations=1, init_p=0.5, tau=0.05, seed=42,
        terminal_window=10, terminal_threshold=2.0,  # window_size = 11
    )

    distances = [1.0, 2.0, 1.5, 2.5, 2.0, 3.0, 2.5, 3.5, 3.0, 4.0, 3.5, 4.5]
    expected_p = [0.45, 0.40, 0.45, 0.40, 0.45, 0.40, 0.45, 0.40, 0.45, 0.40, 0.45, 0.40]

    for epoch, (d, exp_p) in enumerate(zip(distances, expected_p), start=1):
        p = selector.rl_step_relation(0, d, epoch)
        assert abs(p - exp_p) < 1e-9, f"epoch {epoch}: expected {exp_p}, got {p}"
        if epoch < 12:
            assert selector.get_state(0).terminated is False, f"terminated too early at epoch {epoch}"

    state = selector.get_state(0)
    assert len(state.reward_history) == 11
    assert sum(state.reward_history) == -1.0
    assert abs(sum(state.reward_history)) <= 2.0
    assert state.terminated is True

    frozen_p = state.p
    assert abs(frozen_p - 0.40) < 1e-9

    # Feed several more epochs with distances that would ordinarily push p
    # in either direction (including a large improvement, which pre-freeze
    # would have driven a +tau action) and confirm p never moves again.
    for epoch, d in enumerate([0.1, 0.01, 5.0], start=13):
        p = selector.rl_step_relation(0, d, epoch)
        assert p == frozen_p
        assert selector.get_state(0).terminated is True
    # selector.p is a float32 buffer; state.p is a plain Python float --
    # comparing across those two representations needs a tolerance, not
    # exact ==, purely for float32<->float64 storage precision reasons.
    assert abs(float(selector.p[0].item()) - frozen_p) < 1e-6


def test_label_aware_distance_averages_only_over_selected_neighbours():
    """Exact hand-computed value.

    2 fraud centers, max_k=3. Center 0's selected neighbours are slots 0,1
    (slot 2 masked out); center 1's selected neighbour is slot 1 only.
    layer1_pred gives each node a tanh-scalar; distance is |center - neighbor|
    averaged only over masked-in slots.
    """
    layer1_pred = torch.tensor([0.9, 0.1, 0.5, -0.3, 0.6, 0.2, 0.0])
    #                            0    1    2    3     4    5    6   <- node ids

    fraud_center_idx = torch.tensor([0, 4])  # nodes 0 and 4 are the two centers
    neighbor_idx = torch.tensor([
        [1, 2, 3],
        [5, 6, 1],
    ])
    selected_neighbors = torch.tensor([
        [True, True, False],
        [False, True, False],
    ])

    d = label_aware_distance(layer1_pred, fraud_center_idx, selected_neighbors, neighbor_idx)

    # Center 0 (pred 0.9) vs its 2 selected neighbours: node1 (0.1) -> |0.9-0.1|=0.8;
    #   node2 (0.5) -> |0.9-0.5|=0.4.
    # Center 4 (pred 0.6) vs its 1 selected neighbour: node6 (0.0) -> |0.6-0.0|=0.6.
    # Average over all 3 selected pairs: (0.8 + 0.4 + 0.6) / 3 = 1.8 / 3 = 0.6
    assert abs(d - 0.6) < 1e-6


def test_label_aware_distance_returns_zero_when_nothing_selected():
    """Exact hand-computed value: the documented divide-by-zero guard."""
    layer1_pred = torch.tensor([0.1, 0.2, 0.3])
    fraud_center_idx = torch.tensor([0])
    neighbor_idx = torch.tensor([[1, 2]])
    selected_neighbors = torch.zeros(1, 2, dtype=torch.bool)

    d = label_aware_distance(layer1_pred, fraud_center_idx, selected_neighbors, neighbor_idx)
    assert d == 0.0
