"""Augur FastAPI app. Default model is GraphSAGE (F1_illicit=0.6434, the
ablation-study winner), not CARE-GNN -- see api/routes/predict.py's
docstring. Models/graph data are loaded once at startup (api/state.py),
not per-request."""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.routes import console, health, predict, subgraph
from api.state import STATE


@asynccontextmanager
async def lifespan(app: FastAPI):
    STATE.load()
    yield


app = FastAPI(title="Augur", version="1.0.0", lifespan=lifespan)

app.include_router(health.router)
app.include_router(predict.router)
app.include_router(subgraph.router)
app.include_router(console.router)
