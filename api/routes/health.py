"""GET / -- health check + model metadata, so a caller can see the model's
known performance characteristics (from the real ablation study,
ablation/results/ablation_results.csv) rather than getting a bare score
back with no context for how much to trust it."""

from fastapi import APIRouter

from api.schemas import HealthResponse, ModelInfo
from api.state import STATE

router = APIRouter()

GRAPHSAGE_CHECKPOINT_NAME = "graphsage_best.pt"
CARE_GNN_CHECKPOINT_NAME = "care_gnn_full_best.pt"


@router.get("/", response_model=HealthResponse)
async def health():
    STATE.load()

    graphsage_row = STATE.ablation_metrics.get("GraphSAGE", {})
    care_gnn_row = STATE.ablation_metrics.get("CARE-GNN (full)", {})

    models = [
        ModelInfo(
            model_type="graphsage",
            checkpoint=GRAPHSAGE_CHECKPOINT_NAME,
            epoch=STATE.graphsage_epoch,
            f1_illicit=float(graphsage_row["f1_illicit"]) if graphsage_row.get("f1_illicit") else STATE.graphsage_f1,
            auc_roc=float(graphsage_row["auc_roc"]) if graphsage_row.get("auc_roc") else None,
        ),
        ModelInfo(
            model_type="care_gnn",
            checkpoint=CARE_GNN_CHECKPOINT_NAME,
            epoch=STATE.care_gnn_epoch,
            f1_illicit=float(care_gnn_row["f1_illicit"]) if care_gnn_row.get("f1_illicit") else STATE.care_gnn_f1,
            auc_roc=float(care_gnn_row["auc_roc"]) if care_gnn_row.get("auc_roc") else None,
        ),
    ]

    return HealthResponse(status="ok", num_nodes=STATE.num_nodes, models=models)
