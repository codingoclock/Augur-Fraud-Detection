"""UMAP projection of GraphSAGE's learned node embeddings (test split),
coloured by true label -- built from a real forward pass through the actual
trained checkpoint (models/checkpoints/graphsage_best.pt), not synthetic
data. Two outputs from the same projection: a static PNG (matplotlib/
seaborn, for READMEs/reports where a JS-free image is needed) and an
interactive HTML (Plotly: hover per-point node id/true label/P(fraud),
zoom, pan, click-to-isolate a class in the legend).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import seaborn as sns
import torch
import torch.nn.functional as F
import umap

from config import CARE_GNN_CONFIG
from graph.loader import PROCESSED_DIR, load_processed
from graph.relations import load_relations
from models.baselines.graphsage import GraphSAGEBaseline, build_merged_edge_index

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GRAPHSAGE_CHECKPOINT = PROJECT_ROOT / "models" / "checkpoints" / "graphsage_best.pt"
OUTPUT_PATH = PROJECT_ROOT / "visualization" / "embedding_umap_graphsage.png"
OUTPUT_HTML_PATH = PROJECT_ROOT / "visualization" / "embedding_umap_graphsage.html"


def _extract_hidden_embeddings(x_full: torch.Tensor, edge_index: torch.Tensor, model: GraphSAGEBaseline) -> np.ndarray:
    """The learned representation is conv1's output (post-ReLU, hidden_dim=64),
    i.e. everything the model learned BEFORE the final linear classification
    layer (conv2) -- the standard choice for "what did the model learn to
    separate", not the 2-class logits themselves."""
    model.eval()
    with torch.no_grad():
        h = model.conv1(x_full, edge_index)
        h = F.relu(h)
    return h.numpy()


def run(sample_size: int | None = None, seed: int = CARE_GNN_CONFIG["seed"]):
    node_index_map, features, time_steps, labels = load_processed(PROCESSED_DIR)
    adj_tdt, adj_tbt, adj_tft = load_relations(PROCESSED_DIR)

    test_steps = set(CARE_GNN_CONFIG["test_time_steps"])
    test_mask = np.array([int(s) in test_steps for s in time_steps])

    ckpt = torch.load(GRAPHSAGE_CHECKPOINT, map_location="cpu", weights_only=False)
    config = ckpt["config"]

    merged_edges = build_merged_edge_index(adj_tdt, adj_tbt, adj_tft)
    x_full = torch.tensor(features, dtype=torch.float32)
    edge_index = torch.tensor(merged_edges, dtype=torch.long)

    model = GraphSAGEBaseline(
        feature_dim=config["feature_dim"], hidden_dim=config["hidden_dim"],
        num_classes=config["num_classes"], dropout=config["dropout"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print("Running real GraphSAGE forward pass to extract hidden embeddings...")
    embeddings_full = _extract_hidden_embeddings(x_full, edge_index, model)
    with torch.no_grad():
        proba_full = model(x_full, edge_index)[:, 1].exp().numpy()  # P(fraud) per node

    # test_idx values are node ids directly (features/labels/proba are all
    # indexed by the same node-index convention used throughout the API --
    # e.g. api/state.py's graphsage_proba[node_id] -- so they double as the
    # real node id shown in hover text below, not a separate lookup).
    test_idx = np.nonzero(test_mask)[0]
    embeddings = embeddings_full[test_idx]
    test_labels = labels[test_idx]
    test_proba = proba_full[test_idx]
    node_ids = test_idx.copy()

    rng = np.random.RandomState(seed)
    if sample_size is not None and sample_size < len(test_idx):
        sample_pos = rng.choice(len(test_idx), size=sample_size, replace=False)
        embeddings = embeddings[sample_pos]
        test_labels = test_labels[sample_pos]
        test_proba = test_proba[sample_pos]
        node_ids = node_ids[sample_pos]

    print(f"Projecting {embeddings.shape[0]} test-split node embeddings ({embeddings.shape[1]}-d) via UMAP...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="euclidean", random_state=seed)
    proj = reducer.fit_transform(embeddings)

    label_names = {1: "illicit", 0: "licit", -1: "unknown"}
    colors = {"illicit": "#e74c3c", "licit": "#3498db", "unknown": "#95a5a6"}

    fig, ax = plt.subplots(figsize=(11, 8.5))
    for label_val, name in label_names.items():
        mask = test_labels == label_val
        ax.scatter(
            proj[mask, 0], proj[mask, 1],
            s=8 if name == "unknown" else 14,
            alpha=0.35 if name == "unknown" else 0.75,
            c=colors[name], label=f"{name} (n={int(mask.sum())})",
            edgecolors="none",
        )
    ax.set_title("GraphSAGE learned node embeddings (test split), UMAP projection", fontsize=13, pad=12)
    ax.set_xlabel("UMAP-1 (no fixed physical meaning -- axes preserve relative neighbourhoods, not real units)")
    ax.set_ylabel("UMAP-2")
    ax.legend(markerscale=2, frameon=True, title="True label")
    sns.despine(ax=ax)

    explanation = (
        "What this shows: each point is one test-split transaction, plotted by GraphSAGE's learned 64-d\n"
        "hidden representation (post-conv1, pre-classifier) after UMAP compresses it to 2-D. Points close\n"
        "together were treated as similar by the model. A visibly separate red cluster means GraphSAGE\n"
        "learned real structure that distinguishes fraud from non-fraud -- not just a good aggregate\n"
        "F1/AUC score by chance. Grey (unlabelled) points are Elliptic's unresolved transactions, shown\n"
        "for context, not scored against ground truth. Interactive version with per-point node id and\n"
        "P(fraud) on hover: embedding_umap_graphsage.html"
    )
    fig.subplots_adjust(bottom=0.24)
    fig.text(0.02, 0.02, explanation, fontsize=8.5, color="#444444", va="bottom", family="monospace")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=150)
    plt.close(fig)
    print(f"Wrote {OUTPUT_PATH}")

    # --- Interactive Plotly version: real hover data per point, no ---
    # --- pre-baked tooltip text -- click a legend entry to isolate a class.
    fig_html = go.Figure()
    for label_val, name in label_names.items():
        mask = test_labels == label_val
        ids = node_ids[mask]
        probs = test_proba[mask]
        hover = [
            f"node {nid}<br>true label: {name}<br>GraphSAGE P(fraud): {p:.3f}"
            for nid, p in zip(ids.tolist(), probs.tolist())
        ]
        fig_html.add_trace(go.Scattergl(
            x=proj[mask, 0], y=proj[mask, 1],
            mode="markers",
            marker=dict(
                size=5 if name == "unknown" else 7,
                color=colors[name],
                opacity=0.35 if name == "unknown" else 0.8,
                line=dict(width=0),
            ),
            name=f"{name} (n={int(mask.sum())})",
            text=hover,
            hoverinfo="text",
        ))
    fig_html.update_layout(
        title="GraphSAGE learned node embeddings (test split), UMAP projection -- interactive",
        xaxis_title="UMAP-1",
        yaxis_title="UMAP-2",
        template="plotly_dark",
        legend_title_text="True label (click to isolate)",
        width=1100, height=800,
        annotations=[dict(
            text=(
                "Each point = one test-split transaction, positioned by GraphSAGE's learned 64-d hidden "
                "representation compressed to 2-D via UMAP. Hover a point for its real node id, true "
                "label, and predicted P(fraud) -- try that node id against POST /predict. A visibly "
                "separate illicit (red) cluster is evidence the model learned real distinguishing "
                "structure, not just a good aggregate score."
            ),
            xref="paper", yref="paper", x=0, y=-0.12, showarrow=False,
            align="left", font=dict(size=11, color="#aaaaaa"),
        )],
        margin=dict(b=90),
    )
    fig_html.write_html(str(OUTPUT_HTML_PATH), include_plotlyjs="cdn")
    print(f"Wrote {OUTPUT_HTML_PATH}")

    return {
        "output_path": str(OUTPUT_PATH),
        "output_html_path": str(OUTPUT_HTML_PATH),
        "n_points": embeddings.shape[0],
    }


if __name__ == "__main__":
    result = run()
    print(result)
