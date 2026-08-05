"""Forward-pass tests for CARE-GNN modules, built up module by module.

This file covers Module 1 (LabelAwareSimilarity, v3) and Module 3
(RelationAwareAggregator, v3) in isolation against synthetic tensors, plus
one end-to-end test of the fully assembled CAREGNN model against the real,
cached Level 2 Elliptic graph (test_full_model_forward_pass_on_real_graph).

v3 note: Module 1's tests below entirely replace the v2 tests that used to
live here (cosine-similarity-of-projected-features + label agreement). That
formula no longer exists in similarity.py -- v3 rewrote Module 1 from the
paper's actual Eq. 2-4 (a per-node tanh(MLP) scalar predictor, with pairwise
distance/similarity computed separately), so the old tests would test dead
code, not "the same module written differently." They are deleted, not kept
alongside the new ones.
"""

import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import CARE_GNN_CONFIG, FRAUD_CLASS_WEIGHT
from graph.dataset import AugurDataset
from models.care_gnn.aggregator import RelationAwareAggregator
from models.care_gnn.care_gnn import CAREGNN, build_relation_indices
from models.care_gnn.similarity import LabelAwareSimilarity, pairwise_distance, pairwise_similarity
from training.loss import WeightedFocalLoss, SimilarityAuxLoss, care_gnn_total_loss

FEATURE_DIM = 16


# ---------------------------------------------------------------------------
# Module 1 (v3): LabelAwareSimilarity + pairwise_distance/pairwise_similarity
# ---------------------------------------------------------------------------


def test_similarity_output_shape_and_batch_size():
    """Shape/existence check."""
    module = LabelAwareSimilarity(FEATURE_DIM)
    n = 7
    h = torch.randn(n, FEATURE_DIM)
    out = module(h)
    assert out.shape == (n,)


def test_identical_embeddings_similarity_is_one_exactly():
    """Exact hand-computed value.

    Calling the SAME module on the SAME input tensor twice is bit-identical
    computation -> bit-identical output, for any weight init (no pinning
    needed). D = |pred - pred| = 0.0 exactly (not approximately -- literally
    the same float subtracted from itself). S = 1 - 0.0 = 1.0 exactly.
    """
    module = LabelAwareSimilarity(FEATURE_DIM)
    h = torch.randn(5, FEATURE_DIM)

    pred_center = module(h)
    pred_neighbor = module(h)  # same input, same weights -> same output

    d = pairwise_distance(pred_center, pred_neighbor)
    s = pairwise_similarity(pred_center, pred_neighbor)

    assert torch.equal(d, torch.zeros(5))
    assert torch.equal(s, torch.ones(5))


def test_maximally_different_tanh_outputs_similarity_at_achievable_bound():
    """Exact hand-computed value, at the bound tanh saturation actually
    permits in float32 -- NOT a literal 0.0.

    Per Eq. 3 as specified (S = 1 - D, D = |pred_c - pred_n|), two nodes
    whose tanh outputs are +1 and -1 give D = |1 - (-1)| = 2.0 and
    S = 1 - 2.0 = -1.0 -- NOT 0.0. (Flagged separately in this report: the
    v3 spec's own prose for this test case says "similarity = 0.0 exactly
    ... document the achievable bound", which is inconsistent with the
    Eq. 3 code given in the same document. This test follows the literal
    Eq. 3 formula, which is the more authoritative of the two -- an actual
    equation vs. a prose aside -- and documents the bound that formula
    actually implies.)

    tanh never reaches exactly +-1 for finite input mathematically, but DOES
    round to exactly +-1.0 in float32 once |input| is large enough (verified
    empirically below, not assumed): 1 - tanh(x) ~ 2*exp(-2x) underflows
    float32's ~1.19e-7 epsilon around x ~ 9, and x=20 is used here to be
    safely into that saturation regime.
    """
    module = LabelAwareSimilarity(FEATURE_DIM)
    with torch.no_grad():
        # Force Linear(h) to a large positive scalar for "center" and a
        # large negative scalar for "neighbor", regardless of h's direction,
        # by zeroing the weight and using only the bias.
        module.mlp.weight.zero_()
        module.mlp.bias.fill_(20.0)
    center_pred = module(torch.zeros(1, FEATURE_DIM))

    with torch.no_grad():
        module.mlp.bias.fill_(-20.0)
    neighbor_pred = module(torch.zeros(1, FEATURE_DIM))

    # Confirm float32 saturation actually occurred before relying on it.
    assert torch.equal(center_pred, torch.tensor([1.0]))
    assert torch.equal(neighbor_pred, torch.tensor([-1.0]))

    d = pairwise_distance(center_pred, neighbor_pred)
    s = pairwise_similarity(center_pred, neighbor_pred)

    assert torch.equal(d, torch.tensor([2.0]))
    assert torch.equal(s, torch.tensor([-1.0]))


def test_similarity_gradient_flows_through_mlp():
    """Existence/finiteness check, not an exact value."""
    module = LabelAwareSimilarity(FEATURE_DIM)
    h = torch.randn(8, FEATURE_DIM)
    out = module(h)
    out.sum().backward()

    assert module.mlp.weight.grad is not None
    assert torch.isfinite(module.mlp.weight.grad).all()
    assert module.mlp.bias.grad is not None
    assert torch.isfinite(module.mlp.bias.grad).all()


def test_similarity_output_always_strictly_within_tanh_range():
    """Shape/bound check: for ordinary (non-saturating) inputs, tanh's
    output must be strictly inside the open interval (-1, 1)."""
    module = LabelAwareSimilarity(FEATURE_DIM)
    h = torch.randn(10_000, FEATURE_DIM)  # moderate magnitude, no saturation
    out = module(h)
    assert torch.all(out > -1.0)
    assert torch.all(out < 1.0)


def test_similarity_batch_size_one_runs_without_shape_errors():
    """Shape/existence check."""
    module = LabelAwareSimilarity(FEATURE_DIM)
    out = module(torch.randn(1, FEATURE_DIM))
    assert out.shape == (1,)


def test_similarity_large_batch_runs_without_shape_errors():
    """Shape/existence check."""
    module = LabelAwareSimilarity(FEATURE_DIM)
    n = 10_000
    out = module(torch.randn(n, FEATURE_DIM))
    assert out.shape == (n,)


# ---------------------------------------------------------------------------
# Module 3 (v3): RelationAwareAggregator
#
# `thresholds` are RAW p_r^(l) values from the RL-managed selector.p buffer
# -- v3 corrects v1/v2's softmax-normalization of these weights (paper,
# Section 3.4: "the optimal filtering threshold p_r^(l)... [is directly
# applied] as the inter-relation aggregation weights", no normalization
# mentioned or implied). These standalone tests pass thresholds in as a
# plain [num_relations] tensor, since care_gnn.py's wiring is Level 5's
# concern, not this module's.
# ---------------------------------------------------------------------------


def test_aggregator_output_shape_independent_of_feature_vs_output_dim():
    """Shape/existence check: feature_dim != output_dim must not break anything."""
    feature_dim, output_dim, num_relations = 6, 3, 2
    n, max_k = 5, 4
    module = RelationAwareAggregator(feature_dim, num_relations, output_dim)

    x = torch.randn(n, feature_dim)
    selected = [torch.randint(0, 2, (n, max_k)).bool() for _ in range(num_relations)]
    neighbor_feats = [torch.randn(n, max_k, feature_dim) for _ in range(num_relations)]
    thresholds = torch.randn(num_relations)

    out = module(x, selected, neighbor_feats, thresholds)
    assert out.shape == (n, output_dim)


def test_all_false_relation_contributes_exactly_zero():
    """Exact hand-computed value.

    Relation 0's selected_neighbors is all-False. `.clamp(min=1)` makes
    mean_agg well-defined at the zero vector (0 selected / clamped count of
    1 = [0, 0]); ReLU([0,0]) = [0,0] (Eq. 8, applied per-relation); weighted
    contribution = thresholds[0] * [0,0] = [0,0] regardless of the
    threshold's value (no per-relation bias exists in v3's parameterless
    design to leak a nonzero contribution the way v1/v2's W_per_relation
    bias could have -- this test is simpler than its v1/v2 predecessor for
    exactly that reason).

    agg_all pinned to identity, bias 0, so the whole pipeline is traceable:
      self term = x = [1.0, 2.0]
      relation 0 (all-False): contributes [0, 0]
      relation 1 (2 of 3 neighbours selected: [2,4] and [4,6], decoy
        [100,100] excluded): mean = (3.0, 5.0), ReLU(3,5) = (3,5).
        threshold[1] = 2.0 -> contributes 2.0 * (3,5) = (6,10)
      combined = [1,2] + [0,0] + [6,10] = [7, 12]
      agg_all(identity, bias 0) -> [7, 12]; final ReLU -> [7, 12] (already positive)
    """
    feature_dim, output_dim, num_relations = 2, 2, 2
    module = RelationAwareAggregator(feature_dim, num_relations, output_dim)

    with torch.no_grad():
        module.agg_all.weight.copy_(torch.eye(2))
        module.agg_all.bias.zero_()

    x = torch.tensor([[1.0, 2.0]])

    sel0 = torch.zeros(1, 3, dtype=torch.bool)
    nf0 = torch.tensor([[[999.0, -999.0], [123.0, 456.0], [-1.0, -1.0]]])

    sel1 = torch.tensor([[True, True, False]])
    nf1 = torch.tensor([[[2.0, 4.0], [4.0, 6.0], [100.0, 100.0]]])

    thresholds = torch.tensor([5.0, 2.0])  # relation 0's threshold value is irrelevant here

    out = module(x, [sel0, sel1], [nf0, nf1], thresholds)

    expected = torch.tensor([[7.0, 12.0]])
    assert torch.allclose(out, expected, atol=1e-6)


def test_raw_threshold_weighting_not_softmax_normalized():
    """Exact hand-computed value, deliberately constructed so softmax-
    normalized weighting (v1/v2's bug) and raw weighting (v3's correction)
    produce different, distinguishable outputs.

    thresholds = [2.0, 3.0] -- raw values that do NOT sum to 1, unlike
    softmax weights. agg_all pinned to identity, bias 0; x = [1,1]; each
    relation has one selected neighbour with mean already positive (so
    Eq. 8's ReLU is a no-op and doesn't obscure the weighting arithmetic):
      relation 0: neighbour mean [2,2], ReLU->[2,2]; weighted: 2.0*[2,2] = [4,4]
      relation 1: neighbour mean [4,4], ReLU->[4,4]; weighted: 3.0*[4,4] = [12,12]
      combined = [1,1] + [4,4] + [12,12] = [17,17]
      agg_all identity, bias 0 -> [17,17]; ReLU -> [17,17]

    Contrast: softmax([2,3]) ~= [0.2689, 0.7311] (computed here ONLY as a
    comparison baseline to prove the two approaches diverge, not as part of
    the expected-value derivation above) would give combined =
    [1,1] + 0.2689*[2,2] + 0.7311*[4,4] ~= [4.46, 4.46] -- nothing like
    [17, 17]. The raw literal [17.0, 17.0] is the only expected value
    asserted for correctness.
    """
    feature_dim, output_dim, num_relations = 2, 2, 2
    module = RelationAwareAggregator(feature_dim, num_relations, output_dim)

    with torch.no_grad():
        module.agg_all.weight.copy_(torch.eye(2))
        module.agg_all.bias.zero_()

    x = torch.tensor([[1.0, 1.0]])
    sel = torch.tensor([[True]])
    nf0 = torch.tensor([[[2.0, 2.0]]])
    nf1 = torch.tensor([[[4.0, 4.0]]])
    thresholds = torch.tensor([2.0, 3.0])

    out = module(x, [sel, sel], [nf0, nf1], thresholds)

    expected_raw = torch.tensor([[17.0, 17.0]])
    assert torch.allclose(out, expected_raw, atol=1e-6)

    # Comparison baseline only -- confirms the two weighting schemes truly
    # diverge for this input, i.e. this test would have caught v1/v2's bug.
    softmax_weights = F.softmax(thresholds, dim=0)
    softmax_combined = x[0] + softmax_weights[0] * torch.tensor([2.0, 2.0]) + softmax_weights[1] * torch.tensor([4.0, 4.0])
    assert not torch.allclose(out[0], softmax_combined, atol=0.5)


def test_relu_applied_per_relation_before_summing_not_deferred_to_end():
    """Exact hand-computed value, constructed so per-relation ReLU (Eq. 8,
    correct) and end-only ReLU (incorrect, what a naive reading might do)
    give different, distinguishable results.

    1-D features. x = [10.0]. Single relation, single selected neighbour
    with mean = -5.0 (negative). threshold = 1.0. agg_all pinned to
    identity (1x1, weight=1), bias 0.

    Correct (Eq. 8 -- ReLU applied to the relation's mean BEFORE summing):
      h_v_r = ReLU(-5.0) = 0.0
      combined = 10.0 + 1.0 * 0.0 = 10.0
      agg_all(10.0) = 10.0; final ReLU(10.0) = 10.0

    Incorrect alternative (ReLU deferred -- summing the raw signed mean
    before any ReLU except the final one):
      combined_wrong = 10.0 + 1.0 * (-5.0) = 5.0
      agg_all(5.0) = 5.0; final ReLU(5.0) = 5.0

    These two hypotheses give different exact values (10.0 vs 5.0) for the
    same inputs -- asserting 10.0 both confirms the correct behaviour AND
    would fail loudly (getting 5.0) under the incorrect one.
    """
    feature_dim, output_dim, num_relations = 1, 1, 1
    module = RelationAwareAggregator(feature_dim, num_relations, output_dim)

    with torch.no_grad():
        module.agg_all.weight.copy_(torch.tensor([[1.0]]))
        module.agg_all.bias.zero_()

    x = torch.tensor([[10.0]])
    sel = torch.tensor([[True]])
    nf = torch.tensor([[[-5.0]]])
    thresholds = torch.tensor([1.0])

    out = module(x, [sel], [nf], thresholds)

    assert torch.allclose(out, torch.tensor([[10.0]]), atol=1e-6)
    assert not torch.allclose(out, torch.tensor([[5.0]]), atol=1e-6)


def test_agg_all_is_the_only_learnable_transform():
    """Structural check: exactly one Linear (agg_all), no per-relation or
    self-connection weight matrices (the v1/v2 W_self/W_per_relation design
    is gone in v3), and total parameter count matches a single Linear's
    weight + bias exactly."""
    feature_dim, output_dim, num_relations = 6, 4, 3
    module = RelationAwareAggregator(feature_dim, num_relations, output_dim)

    assert isinstance(module.agg_all, nn.Linear)
    assert not hasattr(module, "W_self")
    assert not hasattr(module, "W_per_relation")

    named = dict(module.named_parameters())
    assert set(named.keys()) == {"agg_all.weight", "agg_all.bias"}

    total_params = sum(p.numel() for p in module.parameters())
    expected_params = feature_dim * output_dim + output_dim  # agg_all weight + bias
    assert total_params == expected_params


def test_self_connection_alone_when_all_relations_have_zero_neighbours():
    """Exact hand-computed value.

    All relations' selected_neighbors are all-False for every node. Each
    relation's contribution is forced to exactly zero (see the all-False
    test above), so combined == x exactly, and output must equal
    ReLU(agg_all(x)) exactly.
    """
    feature_dim, output_dim, num_relations = 3, 2, 3
    module = RelationAwareAggregator(feature_dim, num_relations, output_dim)

    n, max_k = 4, 5
    x = torch.randn(n, feature_dim)
    selected = [torch.zeros(n, max_k, dtype=torch.bool) for _ in range(num_relations)]
    neighbor_feats = [torch.randn(n, max_k, feature_dim) for _ in range(num_relations)]
    thresholds = torch.randn(num_relations)

    out = module(x, selected, neighbor_feats, thresholds)
    expected = F.relu(module.agg_all(x))

    assert torch.allclose(out, expected, atol=1e-6)


def test_aggregator_gradient_flows_through_agg_all():
    """Existence/finiteness check, not an exact value."""
    feature_dim, output_dim, num_relations = 4, 3, 3
    n, max_k = 6, 4
    module = RelationAwareAggregator(feature_dim, num_relations, output_dim)

    x = torch.randn(n, feature_dim)
    selected = [torch.randint(0, 2, (n, max_k)).bool() for _ in range(num_relations)]
    neighbor_feats = [torch.randn(n, max_k, feature_dim) for _ in range(num_relations)]
    thresholds = torch.randn(num_relations, requires_grad=True)

    out = module(x, selected, neighbor_feats, thresholds)
    out.sum().backward()

    assert module.agg_all.weight.grad is not None
    assert torch.isfinite(module.agg_all.weight.grad).all()
    assert module.agg_all.bias.grad is not None
    assert torch.isfinite(module.agg_all.bias.grad).all()


def test_aggregator_batch_size_one_runs_without_shape_errors():
    """Shape/existence check."""
    feature_dim, output_dim, num_relations = 5, 2, 2
    module = RelationAwareAggregator(feature_dim, num_relations, output_dim)

    x = torch.randn(1, feature_dim)
    selected = [torch.randint(0, 2, (1, 3)).bool() for _ in range(num_relations)]
    neighbor_feats = [torch.randn(1, 3, feature_dim) for _ in range(num_relations)]
    thresholds = torch.randn(num_relations)

    out = module(x, selected, neighbor_feats, thresholds)
    assert out.shape == (1, output_dim)


def test_aggregator_large_batch_runs_without_shape_errors():
    """Shape/existence check."""
    feature_dim, output_dim, num_relations = 8, 4, 3
    n, max_k = 10_000, 6
    module = RelationAwareAggregator(feature_dim, num_relations, output_dim)

    x = torch.randn(n, feature_dim)
    selected = [torch.randint(0, 2, (n, max_k)).bool() for _ in range(num_relations)]
    neighbor_feats = [torch.randn(n, max_k, feature_dim) for _ in range(num_relations)]
    thresholds = torch.randn(num_relations)

    out = module(x, selected, neighbor_feats, thresholds)
    assert out.shape == (n, output_dim)


def test_aggregator_num_relations_one_and_five_both_work():
    """Shape/existence check across num_relations values; also asserts the
    module doesn't hardcode num_relations=3 anywhere by construction."""
    feature_dim, output_dim, n, max_k = 4, 3, 5, 4

    for num_relations in (1, 5):
        module = RelationAwareAggregator(feature_dim, num_relations, output_dim)
        x = torch.randn(n, feature_dim)
        selected = [torch.randint(0, 2, (n, max_k)).bool() for _ in range(num_relations)]
        neighbor_feats = [torch.randn(n, max_k, feature_dim) for _ in range(num_relations)]
        thresholds = torch.randn(num_relations)

        out = module(x, selected, neighbor_feats, thresholds)
        assert out.shape == (n, output_dim)
        assert module.num_relations == num_relations


# ---------------------------------------------------------------------------
# Full model, real graph
# ---------------------------------------------------------------------------


def test_full_model_forward_pass_on_real_graph():
    """End-to-end verification against the real, cached Level 2 Elliptic
    graph (data/processed/*.pkl) -- not synthetic tensors. Loads the actual
    203,769-node graph, takes 100 real train-split nodes as centres, and
    checks the assembled v3 CAREGNN model produces valid, finite,
    gradient-flowing predictions under the real combined loss
    (care_gnn_total_loss), without ever letting gradients reach the
    RL-managed selector threshold buffers -- and, the single most
    paper-specific assertion in this whole corrective pass, confirms
    layer 0's similarity MLP receives gradient while layer 1's does not.
    """
    dataset = AugurDataset()
    data = dataset.data

    x_full = data["transaction"].x
    labels_full = data["transaction"].y  # {1, 0, -1} convention
    train_mask = dataset.train_mask

    adj_tdt = data["transaction", "tdt", "transaction"].edge_index.numpy()
    adj_tbt = data["transaction", "tbt", "transaction"].edge_index.numpy()
    adj_tft = data["transaction", "tft", "transaction"].edge_index.numpy()

    num_nodes = x_full.shape[0]
    adj_indices = build_relation_indices([adj_tdt, adj_tbt, adj_tft], num_nodes)

    train_idx = torch.nonzero(train_mask, as_tuple=False).squeeze(-1)
    center_nodes = train_idx[:100]
    assert center_nodes.shape[0] == 100

    torch.manual_seed(CARE_GNN_CONFIG["seed"])
    model = CAREGNN(
        feature_dim=CARE_GNN_CONFIG["feature_dim"],
        hidden_dim=CARE_GNN_CONFIG["hidden_dim"],
        num_classes=CARE_GNN_CONFIG["num_classes"],
        num_relations=CARE_GNN_CONFIG["num_relations"],
        num_layers=CARE_GNN_CONFIG["num_layers"],
    )
    assert CARE_GNN_CONFIG["num_layers"] == 2  # this test's layer0-vs-layer1 assertion assumes exactly 2

    p_before = [sel.p.clone() for sel in model.selectors]

    start = time.perf_counter()
    out, layer1_pred = model(x_full, adj_indices, center_nodes)
    elapsed = time.perf_counter() - start
    print(
        f"\n[test_full_model_forward_pass_on_real_graph] "
        f"forward pass on 100 real nodes took {elapsed:.3f}s"
    )

    # Shape and validity of a log_softmax output.
    assert out.shape == (100, CARE_GNN_CONFIG["num_classes"])
    assert torch.isfinite(out).all()
    probs_sum = out.exp().sum(dim=-1)
    assert torch.allclose(probs_sum, torch.ones_like(probs_sum), atol=1e-4)

    # layer1_pred: one tanh scalar per center node, from similarities[0].
    assert layer1_pred.shape == (100,)
    assert torch.isfinite(layer1_pred).all()
    assert torch.all(layer1_pred > -1.0) and torch.all(layer1_pred < 1.0)

    # Every layer's selector.p is a buffer, not touched by forward() --
    # only rl_step_relation() (called once per epoch by the training loop,
    # not from forward()) is allowed to change it. Verified empirically
    # here by comparing pre-forward-pass snapshots for BOTH layers, not
    # just by reading the code.
    for i, sel in enumerate(model.selectors):
        assert torch.equal(sel.p, p_before[i]), f"selectors[{i}].p changed during forward()"
        assert sel.p.requires_grad is False

    # Build the real combined loss (Eq. 11) using the real graph's labels.
    gnn_targets = labels_full[center_nodes]  # {1, 0, -1}
    labelled_mask = gnn_targets != -1
    y_signed = torch.where(gnn_targets == 1, torch.tensor(1.0), torch.tensor(-1.0))

    focal_loss_fn = WeightedFocalLoss(class_weights=torch.tensor([1.0, FRAUD_CLASS_WEIGHT]))
    aux_loss_fn = SimilarityAuxLoss()

    total_loss, breakdown = care_gnn_total_loss(
        out, gnn_targets, layer1_pred, y_signed, labelled_mask,
        focal_loss_fn, aux_loss_fn, lambda1=CARE_GNN_CONFIG["lambda1"],
    )
    assert torch.isfinite(total_loss)
    total_loss.backward()

    # The single most paper-specific assertion in this whole corrective
    # pass: layer 0's similarity MLP is the ONLY one that trains. It
    # receives gradient via SimilarityAuxLoss (Eq. 4, layer-1-only), which
    # depends directly and differentiably on layer1_pred = similarities[0]
    # (h_local). Layer 1's similarity MLP output feeds only the hard top-p
    # selector (non-differentiable argsort/rank comparison) both here and
    # at layer 0 -- so layer 0's gradient path is EXCLUSIVELY through the
    # auxiliary loss, not through the classification loss, and layer 1 has
    # no path to gradient at all. Confirmed empirically, not assumed.
    assert model.similarities[0].mlp.weight.grad is not None, (
        "similarities[0] (layer 1) should receive gradient via SimilarityAuxLoss"
    )
    assert torch.isfinite(model.similarities[0].mlp.weight.grad).all()
    assert model.similarities[1].mlp.weight.grad is None, (
        "similarities[1] (layer 2) unexpectedly received a gradient -- "
        "per the paper, only the first layer's similarity measure is trained"
    )

    for i, agg in enumerate(model.aggregators):
        assert agg.agg_all.weight.grad is not None, f"aggregators[{i}].agg_all has no grad"
        assert torch.isfinite(agg.agg_all.weight.grad).all()

    # selector.p must NEVER receive gradients, in either layer, even though
    # it flows straight into each aggregator's weighting as `thresholds` on
    # every forward pass. This is the exact constraint the entire Module 2
    # rewrite (buffer, not nn.Parameter) exists to protect.
    for i, sel in enumerate(model.selectors):
        assert sel.p.grad is None, f"selectors[{i}].p unexpectedly received a gradient"
