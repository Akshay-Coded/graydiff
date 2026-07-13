"""Pydantic request/response models for the inverse-design API.

Kept deliberately thin: the actual physics and optimization logic all live
in graydiff (imported directly, not reimplemented here) — this module only
describes the JSON shapes at the HTTP boundary.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class InverseDesignRequest(BaseModel):
    mask: list[list[float]] = Field(
        ..., description="2D grayscale mask from the draw canvas, any resolution, values in [0,1] or [0,255]"
    )
    n_starts: int = Field(4, ge=1, le=8)
    n_iters: int = Field(150, ge=10, le=500)


class InverseDesignResponse(BaseModel):
    F: float
    k: float
    F_history: list[float]
    k_history: list[float]
    loss_history: list[float]
    surrogate_field: list[list[float]]
    solver_field: list[list[float]]
    regime: str
    elapsed_seconds: float


class PhaseDiagramResponse(BaseModel):
    F_range: list[float]
    k_range: list[float]
    grid_size: int
    n_steps: int
    F_values: list[float]
    k_values: list[float]
    labels: list[list[str]]
    solver_steps_per_sec: float
