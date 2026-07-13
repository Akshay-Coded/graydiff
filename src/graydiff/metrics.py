"""Phase 4: validation utilities for the FORWARD surrogate.

This module exists entirely in service of one discipline: validate the
forward model honestly before trusting any gradient computed through it
(Phase 5). Nothing here trains anything — it only measures.
"""

from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn

from graydiff.model import make_input

# The real solver's U, V stay within roughly [0, 1] for stable parameters
# (see tests/test_solver.py::test_gray_scott_step_bounded_for_stable_params).
# A long AUTOREGRESSIVE surrogate rollout — fed its own predictions rather
# than ground truth every step — can drift outside the region the network
# was ever trained on and, once there, extrapolate into runaway growth
# (confirmed empirically: this project's trained checkpoint stays accurate
# for hundreds of steps, then diverges rather than saturating gently — see
# notebook 04's rollout-stability section for the measured horizon). This
# clamp is a physically-motivated NUMERICAL SAFETY NET — it stops NaN/Inf
# from poisoning a long diagnostic rollout or an inverse-design gradient —
# not a claim that the surrogate is accurate beyond its measured stable
# horizon. Every rollout length used elsewhere in this project (Phase 4's
# phase-diagram match, Phase 5's inverse design) is deliberately kept well
# inside that measured horizon; this clamp only guards the exploratory long
# rollouts that intentionally run past it to find where it breaks.
STATE_CLAMP_RANGE = (-0.5, 1.5)


@torch.no_grad()
def surrogate_rollout_trajectory(
    model: nn.Module,
    F_val: torch.Tensor,
    k_val: torch.Tensor,
    seed_state: torch.Tensor,
    n_steps: int,
    clamp_range: tuple[float, float] | None = STATE_CLAMP_RANGE,
) -> torch.Tensor:
    """Run the surrogate autoregressively (feeding its own output back in)
    for n_steps, no gradients. Returns [n_steps+1, 2, H, W] — the full
    trajectory, including the seed state at index 0, so it's directly
    comparable to graydiff.solver.rollout's snapshot output."""
    model.eval()
    state = seed_state
    trajectory = [state[0].clone()]
    for _ in range(n_steps):
        state = model(make_input(state, F_val, k_val))
        if clamp_range is not None:
            state = torch.clamp(state, *clamp_range)
        trajectory.append(state[0].clone())
    return torch.stack(trajectory)


def rollout_error_curve(
    surrogate_trajectory: torch.Tensor, solver_trajectory: np.ndarray
) -> np.ndarray:
    """Per-step MSE between a surrogate trajectory [T, 2, H, W] and a solver
    trajectory of the same shape — the honest "how long does it stay
    accurate" curve for Phase 4's rollout-stability test."""
    surrogate_np = surrogate_trajectory.detach().cpu().numpy()
    diff = surrogate_np - solver_trajectory
    return np.mean(diff**2, axis=(1, 2, 3))


def nearest_training_distance(F: float, k: float, train_F: np.ndarray, train_k: np.ndarray) -> float:
    """Euclidean distance in (F, k) space from a query point to the nearest
    point actually seen during training — the x-axis for Phase 4's honest
    OOD-interpolation check (error vs. distance from nearest training point)."""
    return float(np.min(np.hypot(train_F - F, train_k - k)))


def time_fn(fn, *args, n_repeats: int = 3, **kwargs) -> float:
    """Median wall-clock seconds for n_repeats calls to fn(*args, **kwargs).
    Median (not mean) to be robust to one-off warmup/scheduling spikes --
    the project's standing rule is to always measure, never invent, a
    performance number."""
    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        fn(*args, **kwargs)
        times.append(time.perf_counter() - t0)
    return float(np.median(times))
