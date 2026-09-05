"""GET /app -- serves the static single-page console (api/static/console.html)
that combines /predict and /subgraph/{node_id} into one UI. Pure static file
serving: this route does not touch STATE or duplicate any prediction/graph
logic, it only hands back the page; the page's own JS calls /predict and
/subgraph/{node_id} exactly as they exist.

GET /cover -- serves the project's augur_cover.html landing page, so the
console's "AUGUR" heading can link back to it."""

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse

router = APIRouter()

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


@router.get("/app", include_in_schema=False)
async def console():
    return FileResponse(STATIC_DIR / "console.html")


@router.get("/cover", include_in_schema=False)
async def cover():
    return FileResponse(PROJECT_ROOT / "augur_cover.html")
