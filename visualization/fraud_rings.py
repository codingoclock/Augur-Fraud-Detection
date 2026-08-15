"""Fraud subgraph visualisation (Pyvis), built from real trained-model
predictions -- no synthetic/placeholder data.

Primary model: GraphSAGE (models/checkpoints/graphsage_best.pt), per the
completed ablation study (ablation/results/ablation_results.md) -- GraphSAGE
is Augur's best-performing model (F1_illicit=0.6434), so it's the headline
visual, not CARE-GNN's.

Checkpoint mixup found in Level 9, fixed in Level 10: models/checkpoints/
care_gnn_best.pt (the old implicit default path) had been silently
overwritten by whichever CARE-GNN-family ablation run finished last without
an explicit checkpoint_path override. training/trainer.py's `train()` now
requires checkpoint_path explicitly (no default, keyword-only) so this class
of bug can't recur. The checkpoint files were renamed to be unambiguous:
the genuine full 3-relation model is now models/checkpoints/
care_gnn_full_best.pt (epoch 120, f1_illicit=0.5514 -- matches the
full-CARE-GNN row in the ablation table); the old care_gnn_best.pt (which
actually held single_relation_tdt's weights) is now
care_gnn_single_relation_tdt_best.pt. That full-model file is used below.

Seed-node selection and 2-hop expansion happen entirely over REAL data: test
split predictions from a real forward pass, real tdt adjacency from
data/processed/adj_tdt.pkl.
"""

import os
from pathlib import Path

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import networkx as nx
import numpy as np
import torch
from pyvis.network import Network

from config import CARE_GNN_CONFIG
from graph.loader import PROCESSED_DIR, load_processed
from graph.relations import load_relations
from models.baselines.graphsage import GraphSAGEBaseline, build_merged_edge_index
from models.care_gnn.care_gnn import CAREGNN, build_relation_indices
from training.trainer import _chunk

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINTS_DIR = PROJECT_ROOT / "models" / "checkpoints"
GRAPHSAGE_CHECKPOINT = CHECKPOINTS_DIR / "graphsage_best.pt"
CARE_GNN_CHECKPOINT = CHECKPOINTS_DIR / "care_gnn_full_best.pt"
OUTPUT_DIR = PROJECT_ROOT / "visualization"

NUM_SEEDS = 5  # judgment call, see report: enough to show ring structure,
               # few enough that 2-hop tdt expansion stays legible
SEED_CONFIDENCE_THRESHOLD = 0.9  # P(fraud) cutoff for "high confidence"
MAX_HOP_FANOUT = 15  # per-node cap during expansion, guards against a rare
                       # tdt hub node blowing up the visualization


def _load_graph_data():
    node_index_map, features, time_steps, labels = load_processed(PROCESSED_DIR)
    adj_tdt, adj_tbt, adj_tft = load_relations(PROCESSED_DIR)

    train_steps = set(CARE_GNN_CONFIG["train_time_steps"])
    test_steps = set(CARE_GNN_CONFIG["test_time_steps"])
    train_mask = np.array([int(s) in train_steps for s in time_steps])
    test_mask = np.array([int(s) in test_steps for s in time_steps])

    return {
        "features": features, "labels": labels, "time_steps": time_steps,
        "node_index_map": node_index_map,
        "adj_tdt": adj_tdt, "adj_tbt": adj_tbt, "adj_tft": adj_tft,
        "train_mask": train_mask, "test_mask": test_mask,
    }


def _graphsage_predict_all(data: dict) -> np.ndarray:
    """Real forward pass of the trained GraphSAGE checkpoint over the whole
    graph -- P(fraud) per node, index-aligned with node_index_map."""
    ckpt = torch.load(GRAPHSAGE_CHECKPOINT, map_location="cpu", weights_only=False)
    config = ckpt["config"]

    merged_edges = build_merged_edge_index(data["adj_tdt"], data["adj_tbt"], data["adj_tft"])
    x_full = torch.tensor(data["features"], dtype=torch.float32)
    edge_index = torch.tensor(merged_edges, dtype=torch.long)

    model = GraphSAGEBaseline(
        feature_dim=config["feature_dim"], hidden_dim=config["hidden_dim"],
        num_classes=config["num_classes"], dropout=config["dropout"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    with torch.no_grad():
        out = model(x_full, edge_index)
    return out[:, 1].exp().numpy()  # P(fraud) per node


def _care_gnn_predict(data: dict, center_nodes: np.ndarray) -> np.ndarray:
    """Real forward pass of the (correct, full 3-relation) CARE-GNN
    checkpoint, restricted to `center_nodes` -- CAREGNN.forward() requires
    explicit center_nodes rather than computing every node at once."""
    ckpt = torch.load(CARE_GNN_CHECKPOINT, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    num_nodes = data["features"].shape[0]

    adj_indices = build_relation_indices([data["adj_tdt"], data["adj_tbt"], data["adj_tft"]], num_nodes)
    x_full = torch.tensor(data["features"], dtype=torch.float32)

    model = CAREGNN(
        feature_dim=config["feature_dim"], hidden_dim=config["hidden_dim"],
        num_classes=config["num_classes"], num_relations=config["num_relations"],
        num_layers=config["num_layers"], max_neighbors=64, config=config,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    center_t = torch.tensor(center_nodes, dtype=torch.long)
    probs = np.zeros(len(center_nodes), dtype=np.float32)
    with torch.no_grad():
        for i, batch in enumerate(_chunk(center_t, 2048)):
            out, _ = model(x_full, adj_indices, batch)
            start = i * 2048
            probs[start : start + batch.shape[0]] = out[:, 1].exp().numpy()
    return probs


def select_seed_nodes(data: dict, pred_proba: np.ndarray, num_seeds: int = NUM_SEEDS, threshold: float = SEED_CONFIDENCE_THRESHOLD) -> np.ndarray:
    """Test-split nodes GraphSAGE flags as illicit with high confidence.
    Judgment call (documented, not silent): rank by P(fraud) descending among
    test-split nodes above `threshold`, take the top `num_seeds`. If fewer
    than num_seeds nodes clear the threshold, backfill with the next-highest
    scoring test nodes so the visualization always has seeds."""
    test_idx = np.nonzero(data["test_mask"])[0]
    test_scores = pred_proba[test_idx]
    order = np.argsort(-test_scores)  # descending
    ranked = test_idx[order]
    above = ranked[test_scores[order] >= threshold]
    seeds = above[:num_seeds] if len(above) >= num_seeds else ranked[:num_seeds]
    return seeds


def expand_tdt_hops(adj_tdt: np.ndarray, seed_nodes: np.ndarray, hops: int = 2, max_fanout: int = MAX_HOP_FANOUT) -> set:
    """BOTH-direction 2-hop expansion over tdt edges (src->dst is directed
    "who paid whom", but expansion follows both directions so we see money
    flowing in AND out of a seed, not just downstream) -- per-node fan-out
    capped to keep the rendered graph legible."""
    src, dst = adj_tdt[0], adj_tdt[1]
    out_neighbors: dict = {}
    in_neighbors: dict = {}
    for s, d in zip(src.tolist(), dst.tolist()):
        out_neighbors.setdefault(s, []).append(d)
        in_neighbors.setdefault(d, []).append(s)

    frontier = set(seed_nodes.tolist())
    discovered = set(frontier)
    for _ in range(hops):
        next_frontier = set()
        for node in frontier:
            neighbors = out_neighbors.get(node, [])[:max_fanout] + in_neighbors.get(node, [])[:max_fanout]
            for n in neighbors:
                if n not in discovered:
                    next_frontier.add(n)
        discovered.update(next_frontier)
        frontier = next_frontier
    return discovered


def _node_color(node_id: int, data: dict, pred_proba: np.ndarray) -> str:
    """red=illicit, blue=licit, grey=unlabelled-and-unscored. True label
    used where known; model prediction as fallback for unlabelled nodes
    (every node has a prediction here, since GraphSAGE/CARE-GNN both produce
    a score for the full node set -- grey is a defensive fallback, not
    expected to actually appear)."""
    label = data["labels"][node_id]
    if label == 1:
        return "red"
    if label == 0:
        return "blue"
    if pred_proba is None or node_id >= len(pred_proba):
        return "grey"
    return "red" if pred_proba[node_id] >= 0.5 else "blue"


def _describe_node(nid: int, data: dict, pred_proba: np.ndarray, is_seed: bool) -> str:
    label_val = data["labels"][nid]
    true_label = "fraud (labelled)" if label_val == 1 else "licit (labelled)" if label_val == 0 else "unlabelled"
    score = float(pred_proba[nid]) if pred_proba is not None and nid < len(pred_proba) and not np.isnan(pred_proba[nid]) else None
    lines = [f"Node {nid}", f"True label: {true_label}"]
    lines.append(f"Model P(fraud): {score:.3f}" if score is not None else "Model P(fraud): not scored")
    if is_seed:
        lines.append("Role: SEED — this is the flagged node the ring is centred on")
    return "\n".join(lines)


def build_subgraph_html(data: dict, pred_proba: np.ndarray, seed_nodes: np.ndarray, output_path: Path, title: str):
    """Renders an interactive Pyvis fraud-ring graph, wrapped in a header
    panel that explains what a "ring" is here and how to read the colours --
    the raw vis-network canvas alone (colours + a hover tooltip) isn't
    self-explanatory to someone who hasn't read this codebase.
    """
    discovered = expand_tdt_hops(data["adj_tdt"], seed_nodes)
    seed_set = set(seed_nodes.tolist())

    G = nx.DiGraph()
    counts = {"illicit": 0, "licit": 0, "unlabelled": 0}
    for nid in discovered:
        is_seed = nid in seed_set
        color = _node_color(nid, data, pred_proba)
        label_val = data["labels"][nid]
        counts["illicit" if label_val == 1 else "licit" if label_val == 0 else "unlabelled"] += 1
        G.add_node(
            nid,
            color=("gold" if is_seed else color),
            size=30 if is_seed else 14,
            title=_describe_node(nid, data, pred_proba, is_seed),
            label=str(nid) if is_seed else "",
            borderWidth=3 if is_seed else 1,
        )

    src, dst = data["adj_tdt"][0], data["adj_tdt"][1]
    for s, d in zip(src.tolist(), dst.tolist()):
        if s in discovered and d in discovered:
            G.add_edge(s, d, title="tdt: real Bitcoin transaction, arrow points from payer to payee")

    net = Network(height="750px", width="100%", bgcolor="#0d1117", font_color="white", directed=True)
    net.from_nx(G)
    # Physics on/off toggle folded directly into this JSON (rather than a
    # separate net.show_buttons() call) -- pyvis's set_options() always
    # replaces net.options with a plain dict (Options.set() does
    # json.loads() and returns a dict, not an Options object), so any
    # later show_buttons() call crashes on `self.options.configure`
    # (AttributeError: 'dict' object has no attribute 'configure').
    net.set_options("""{
        "physics": {"stabilization": {"iterations": 200}},
        "edges": {"arrows": {"to": {"enabled": true}}, "color": {"color": "#4a5568", "highlight": "#f0c419"}, "smooth": {"type": "continuous"}},
        "interaction": {"hover": true, "tooltipDelay": 100, "navigationButtons": true, "keyboard": true},
        "configure": {"enabled": true, "filter": ["physics"], "showButton": true}
    }""")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    net.write_html(str(output_path))

    n_seeds = len(seed_set)
    n_nodes = len(discovered)
    n_edges = G.number_of_edges()
    panel_html = f"""
<div style="font-family: -apple-system, Segoe UI, Helvetica, Arial, sans-serif; background:#0d1117; color:#e6edf3; padding: 20px 28px 8px 28px;">
  <h1 style="margin: 0 0 6px 0; font-size: 22px;">{title}</h1>
  <p style="max-width: 900px; line-height: 1.5; color:#c9d1d9; font-size: 14px;">
    A <strong>fraud ring</strong> here means the local transaction neighbourhood around one or more
    nodes the model flags as likely illicit: starting from the <strong style="color:#f0c419;">gold seed node(s)</strong>,
    the graph expands outward <strong>2 hops</strong> over real <strong>tdt</strong> edges
    (directed Bitcoin transactions — arrows point from payer to payee), in <em>both</em> directions,
    so you see money flowing both into and out of the seed. Coordinated fraud tends to show up
    structurally as a seed transacting with a cluster of other illicit-looking nodes rather than
    in isolation — that clustering is what this view is meant to make visible.
  </p>
  <div style="display:flex; gap: 28px; flex-wrap: wrap; margin: 14px 0 10px 0; font-size: 13px;">
    <div><span style="display:inline-block;width:14px;height:14px;border-radius:50%;background:#f0c419;border:2px solid #fff;margin-right:6px;vertical-align:middle;"></span>Seed node — the flagged/queried node ({n_seeds})</div>
    <div><span style="display:inline-block;width:12px;height:12px;border-radius:50%;background:red;margin-right:6px;vertical-align:middle;"></span>Illicit — labelled fraud, or model P(fraud) ≥ 0.5 if unlabelled ({counts['illicit']})</div>
    <div><span style="display:inline-block;width:12px;height:12px;border-radius:50%;background:blue;margin-right:6px;vertical-align:middle;"></span>Licit — labelled non-fraud, or model P(fraud) &lt; 0.5 if unlabelled ({counts['licit']})</div>
    <div><span style="display:inline-block;width:12px;height:12px;border-radius:50%;background:grey;margin-right:6px;vertical-align:middle;"></span>Unlabelled / unscored ({counts['unlabelled']})</div>
  </div>
  <p style="font-size: 12.5px; color:#8b949e; margin: 0 0 10px 0;">
    {n_nodes} nodes · {n_edges} tdt edges shown. Drag nodes to rearrange, scroll to zoom, hover any
    node or edge for its real label/score, use the Physics panel below the graph to freeze the layout.
  </p>
</div>
"""
    html = output_path.read_text()
    html = html.replace("<body>", "<body>" + panel_html, 1)
    output_path.write_text(html)

    return n_nodes, n_edges


def run():
    data = _load_graph_data()

    print("Running real GraphSAGE inference over the full graph...")
    graphsage_proba = _graphsage_predict_all(data)

    seeds = select_seed_nodes(data, graphsage_proba)
    print(f"Seed nodes (test-split, GraphSAGE P(fraud) >= {SEED_CONFIDENCE_THRESHOLD} where possible): {seeds.tolist()}")
    print(f"  scores: {[round(float(graphsage_proba[s]), 4) for s in seeds]}")

    n_nodes, n_edges = build_subgraph_html(
        data, graphsage_proba, seeds, OUTPUT_DIR / "fraud_ring_graphsage.html",
        "GraphSAGE fraud ring — top 5 highest-confidence illicit predictions, test split"
    )
    print(f"GraphSAGE fraud ring: {n_nodes} nodes, {n_edges} tdt edges -> {OUTPUT_DIR / 'fraud_ring_graphsage.html'}")

    print("Running real CARE-GNN (full, corrected-protocol checkpoint) inference on the same seed neighbourhood...")
    discovered = expand_tdt_hops(data["adj_tdt"], seeds)
    discovered_arr = np.array(sorted(discovered), dtype=np.int64)
    care_gnn_proba_subset = _care_gnn_predict(data, discovered_arr)
    care_gnn_proba_full = np.full(data["features"].shape[0], np.nan, dtype=np.float32)
    care_gnn_proba_full[discovered_arr] = care_gnn_proba_subset

    n_nodes2, n_edges2 = build_subgraph_html(
        data, care_gnn_proba_full, seeds, OUTPUT_DIR / "fraud_ring_care_gnn.html",
        "CARE-GNN fraud ring — same 5 seed nodes as GraphSAGE, scored by CARE-GNN instead"
    )
    print(f"CARE-GNN fraud ring (same seeds): {n_nodes2} nodes, {n_edges2} tdt edges -> {OUTPUT_DIR / 'fraud_ring_care_gnn.html'}")

    return {
        "seeds": seeds.tolist(),
        "graphsage_output": str(OUTPUT_DIR / "fraud_ring_graphsage.html"),
        "care_gnn_output": str(OUTPUT_DIR / "fraud_ring_care_gnn.html"),
        "graphsage_nodes": n_nodes, "graphsage_edges": n_edges,
        "care_gnn_nodes": n_nodes2, "care_gnn_edges": n_edges2,
    }


if __name__ == "__main__":
    result = run()
    print(result)
