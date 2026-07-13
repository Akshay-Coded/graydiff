"""The inverse-design endpoint.

Deliberately thin: imports graydiff directly and calls the exact same
functions the notebooks already validated (graydiff.inverse,
graydiff.target_utils, graydiff.solver) rather than reimplementing any of
the physics or optimization logic here. This is also why the inverse solve
has to live in a backend at all — it needs PyTorch backprop, which
onnxruntime-web (the forward playground's in-browser runtime) cannot do.
"""

from __future__ import annotations

import time

import numpy as np
import torch
from fastapi import APIRouter

from graydiff.inverse import multi_start_inverse_design, rollout_surrogate, seed_state_tensor
from graydiff.phase_classify import classify_pattern
from graydiff.solver import rollout as solver_rollout
from graydiff.solver import standard_seed
from graydiff.target_utils import preprocess_target
from web.backend.model_state import GRID, N_STEPS, get_device, get_model
from web.backend.schemas import InverseDesignRequest, InverseDesignResponse

router = APIRouter()


@router.post("/inverse-design", response_model=InverseDesignResponse)
def solve_inverse_design(request: InverseDesignRequest) -> InverseDesignResponse:
    model = get_model()
    device = get_device()

    mask = np.array(request.mask, dtype=np.float32)
    target = preprocess_target(mask, grid_size=GRID).unsqueeze(0).to(device)
    seed = seed_state_tensor(GRID, device=device)

    t0 = time.perf_counter()
    result = multi_start_inverse_design(
        model, target, seed,
        n_starts=request.n_starts, grid_n=10,
        n_steps=N_STEPS, n_iters=request.n_iters, lr=0.003,
    )
    elapsed = time.perf_counter() - t0

    with torch.no_grad():
        surrogate_final = rollout_surrogate(model, torch.tensor([result.F], device=device),
                                             torch.tensor([result.k], device=device), seed, N_STEPS)
    surrogate_field = surrogate_final[0, 1].clamp(0, 1).cpu().tolist()

    # Independent verification with the REAL solver -- never grade the
    # surrogate's recovery using only the surrogate itself.
    U, V = standard_seed(GRID, GRID)
    _, solver_field_arr, _ = solver_rollout(U, V, n_steps=N_STEPS, F=result.F, k=result.k)

    U_conv, V_conv = standard_seed(GRID, GRID)
    _, converged_field, _ = solver_rollout(U_conv, V_conv, n_steps=8000, F=result.F, k=result.k)
    regime = classify_pattern(converged_field)

    return InverseDesignResponse(
        F=result.F,
        k=result.k,
        F_history=result.F_history,
        k_history=result.k_history,
        loss_history=result.loss_history,
        surrogate_field=surrogate_field,
        solver_field=solver_field_arr.tolist(),
        regime=regime,
        elapsed_seconds=elapsed,
    )
