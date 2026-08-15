"""GET /subgraph/{node_id} -- reuses Level 9's visualization/fraud_rings.py
logic directly (not reimplemented) to return the 2-hop tdt-expanded
neighbourhood around a given node, same red/blue/grey + gold-seed visual
convention already established."""

import tempfile
from pathlib import Path

import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse

from api.state import STATE
from visualization.fraud_rings import build_subgraph_html

router = APIRouter()


@router.get("/subgraph/{node_id}", response_class=HTMLResponse)
async def get_subgraph(node_id: int):
    STATE.load()

    if not (0 <= node_id < STATE.num_nodes):
        raise HTTPException(status_code=404, detail=f"node_id {node_id} out of range [0, {STATE.num_nodes})")

    data = {"labels": STATE.labels, "adj_tdt": STATE.adj_tdt}
    seed_nodes = np.array([node_id], dtype=np.int64)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / f"subgraph_{node_id}.html"
        build_subgraph_html(
            data, STATE.graphsage_proba, seed_nodes, output_path,
            title=f"2-hop transaction neighbourhood around node {node_id} (GraphSAGE-scored)"
        )
        html_content = output_path.read_text()

    return HTMLResponse(content=html_content)
