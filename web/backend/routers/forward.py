"""Runs the REAL NumPy Gray-Scott solver server-side for a long rollout.

The ONNX surrogate powering the in-browser playground only stays physically
accurate for roughly 1,000-1,500 autoregressive steps (see notebook 04) --
well short of the several thousand steps a pattern typically needs to fully
form (see notebook 00). The real solver has no such ceiling: it's also fast
enough (~20,000 steps/sec at this grid size, measured in notebook 00) that
running it for a full rollout on demand is a sub-second server call, not a
batch job. This endpoint exists purely so the frontend can show genuine,
fully-converged Turing patterns rather than what the surrogate can reach.
"""

from __future__ import annotations

import time

from fastapi import APIRouter

from graydiff.phase_classify import classify_pattern
from graydiff.solver import rollout as solver_rollout
from graydiff.solver import standard_seed
from web.backend.model_state import GRID
from web.backend.schemas import ForwardSolveRequest, ForwardSolveResponse, ForwardSolveSnapshot

router = APIRouter()

N_SNAPSHOTS = 6


@router.post("/forward-solve", response_model=ForwardSolveResponse)
def forward_solve(request: ForwardSolveRequest) -> ForwardSolveResponse:
    n_steps = request.n_steps
    save_every = max(n_steps // N_SNAPSHOTS, 1)

    U, V = standard_seed(GRID, GRID)
    t0 = time.perf_counter()
    _, Vf, snaps = solver_rollout(U, V, n_steps=n_steps, F=request.F, k=request.k, save_every=save_every)
    elapsed = time.perf_counter() - t0

    snapshot_steps = list(range(0, n_steps, save_every))[: len(snaps)]
    snapshots = [
        ForwardSolveSnapshot(step=step, field=v.tolist())
        for step, (_, v) in zip(snapshot_steps, snaps)
    ]

    return ForwardSolveResponse(
        final_field=Vf.tolist(),
        snapshots=snapshots,
        n_steps=n_steps,
        regime=classify_pattern(Vf),
        elapsed_seconds=elapsed,
    )
