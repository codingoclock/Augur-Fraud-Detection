"""POST /predict -- scores a transaction node. Default model is GraphSAGE
(the ablation-study winner, F1_illicit=0.6434), not CARE-GNN -- this reflects
the project's actual conclusion, not the original build spec's outdated
assumption that CARE-GNN would win. `model=care_gnn` requests the full,
correctly-checkpointed CARE-GNN model instead (see Level 10's checkpoint
rename), so the two can be compared directly through the API."""

from fastapi import APIRouter, HTTPException

from api.schemas import PredictRequest, PredictResponse
from api.state import STATE

router = APIRouter()

MODEL_VERSIONS = {
    "graphsage": "GraphSAGE-corrected-protocol-v1",
    "care_gnn": "CARE-GNN-full-corrected-protocol-v1",
}


@router.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    STATE.load()

    try:
        if req.model == "graphsage":
            proba = STATE.graphsage_predict(req.node_id)
        else:
            proba = STATE.care_gnn_predict(req.node_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    is_fraud = proba >= 0.5
    explanation = (
        f"{req.model} predicts P(fraud)={proba:.4f} for node {req.node_id}, "
        f"based on its neighbourhood in the combined tdt+tbt+tft transaction "
        f"graph ({'GraphSAGE: merged-relation message passing' if req.model == 'graphsage' else 'CARE-GNN: RL-filtered, relation-weighted aggregation'}). "
        f"{'Flagged as likely fraud' if is_fraud else 'Not flagged'} at the standard 0.5 threshold."
    )

    return PredictResponse(
        node_id=req.node_id,
        fraud_probability=proba,
        is_fraud=is_fraud,
        explanation=explanation,
        model_version=MODEL_VERSIONS[req.model],
    )
