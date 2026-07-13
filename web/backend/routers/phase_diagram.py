"""Serves the precomputed phase-diagram cache (notebook 00) so the frontend
doesn't need to duplicate solver logic just to draw the background regions
of its phase-diagram view."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException

from web.backend.schemas import PhaseDiagramResponse

router = APIRouter()

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CACHE_PATH = _REPO_ROOT / "data" / "phase_diagram_cache" / "phase_diagram.json"


@router.get("/phase-diagram", response_model=PhaseDiagramResponse)
def get_phase_diagram() -> PhaseDiagramResponse:
    if not _CACHE_PATH.exists():
        raise HTTPException(status_code=404, detail="Phase diagram cache not found -- run notebook 00 first.")
    with open(_CACHE_PATH) as f:
        data = json.load(f)
    return PhaseDiagramResponse(**data)
