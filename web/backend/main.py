"""FastAPI backend: the one part of this project that has to run server-side.

Everything else (the forward playground) runs client-side via ONNX in the
browser -- see web/frontend/. This service exists only because the inverse
design optimization needs PyTorch backprop, which onnxruntime-web cannot do.

Run from the repo root:
    uv run uvicorn web.backend.main:app --reload --port 8001
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from web.backend.model_state import get_model
from web.backend.routers import inverse, phase_diagram

app = FastAPI(title="graydiff inverse-design API")

# The frontend is served as a separate static site (e.g. `python -m
# http.server` from web/frontend/), so it's a different origin than this
# API -- CORS has to be explicitly opened for local development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(inverse.router)
app.include_router(phase_diagram.router)


@app.on_event("startup")
def _load_model_at_startup() -> None:
    get_model()  # load once, eagerly, so the first real request isn't slow


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
