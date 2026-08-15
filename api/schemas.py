"""Pydantic request/response models for the Augur API."""

from typing import Literal

from pydantic import BaseModel, Field

ModelName = Literal["graphsage", "care_gnn"]


class PredictRequest(BaseModel):
    node_id: int = Field(..., description="Transaction node id from the cached Elliptic graph (data/processed/).")
    model: ModelName = Field("graphsage", description="Which trained model to score with. Default: GraphSAGE, the ablation-study winner.")


class PredictResponse(BaseModel):
    node_id: int
    fraud_probability: float
    is_fraud: bool
    explanation: str
    model_version: str


class ModelInfo(BaseModel):
    model_type: str
    checkpoint: str
    epoch: int | None = None
    f1_illicit: float | None = None
    auc_roc: float | None = None


class HealthResponse(BaseModel):
    status: str
    num_nodes: int
    models: list[ModelInfo]
