"""Phase 5: INVERSE DESIGN — the centerpiece of the project.

Everything before this phase was scaffolding to make this result trustworthy.
Given a target pattern, freeze the trained surrogate and gradient-descend on
the PHYSICS (F, k) — not the network weights — until the surrogate's rollout
from a standard seed produces something matching the target, using
graydiff.losses.pattern_loss (never pixel-MSE — see that module's docstring
for why). This is possible at all only because the surrogate is
differentiable in (F, k); the classical solver has no gradients to offer.

The spec calls out three things that reliably go wrong, and this module's
structure is a direct response to each:

  1. MEMORY — backpropagating through a long rollout stores activations for
     every step. `rollout_surrogate` defaults to whatever `n_steps` the
     caller passes (the recommended lever is simply to use a SHORTER
     rollout for optimization than for a full animation — 50-100 steps is
     often enough for the pattern character to be established) and supports
     gradient checkpointing (`use_checkpoint=True`, backed by
     torch.utils.checkpoint with use_reentrant=False) to trade compute for
     memory on longer rollouts.

  2. THE LOSS — always graydiff.losses.pattern_loss, which compares
     frequency-domain pattern character rather than pixel position.

  3. LOCAL MINIMA — the (F, k) loss landscape is not convex, but with only
     two parameters a coarse grid search (the surrogate is fast — a 10x10
     grid takes seconds) followed by gradient refinement from several
     nearby starts is a cheap, honest, defensible fix
     (`multi_start_inverse_design`).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from graydiff.constants import DEFAULT_GRID_SIZE, F_RANGE, K_RANGE
from graydiff.losses import pattern_loss
from graydiff.model import make_input
from graydiff.solver import standard_seed


def seed_state_tensor(grid_size: int = DEFAULT_GRID_SIZE, device: torch.device | str = "cpu") -> torch.Tensor:
    """The standard, deterministic starting state (graydiff.solver.standard_seed)
    as a [1, 2, H, W] float32 tensor. Every inverse-design iteration starts
    from exactly this state — the optimization is only well-posed if the
    starting point is held fixed across gradient steps."""
    U, V = standard_seed(grid_size, grid_size)
    state = np.stack([U, V]).astype(np.float32)
    return torch.from_numpy(state).unsqueeze(0).to(device)


def rollout_surrogate(
    model: nn.Module,
    F_val: torch.Tensor,
    k_val: torch.Tensor,
    seed_state: torch.Tensor,
    n_steps: int,
    use_checkpoint: bool = False,
) -> torch.Tensor:
    """Differentiable forward rollout: n_steps of the FROZEN surrogate,
    starting from seed_state, with F_val/k_val re-broadcast into the input
    at every step. Gradients flow from the returned state all the way back
    to F_val and k_val through the entire unroll."""
    state = seed_state

    def step(s: torch.Tensor) -> torch.Tensor:
        return model(make_input(s, F_val, k_val))

    for _ in range(n_steps):
        state = checkpoint(step, state, use_reentrant=False) if use_checkpoint else step(state)
    return state


def grid_search(
    model: nn.Module,
    target: torch.Tensor,
    seed_state: torch.Tensor,
    F_range: tuple[float, float] = F_RANGE,
    k_range: tuple[float, float] = K_RANGE,
    n_grid: int = 10,
    n_steps: int = 80,
) -> tuple[float, float]:
    """Evaluate pattern_loss (no gradients) on an n_grid x n_grid sweep of
    (F, k) and return the best point — a cheap, robust starting region for
    the gradient refinement that follows."""
    model.eval()
    F_vals = torch.linspace(*F_range, n_grid)
    k_vals = torch.linspace(*k_range, n_grid)
    best_loss = float("inf")
    best = (float(F_vals.mean()), float(k_vals.mean()))
    with torch.no_grad():
        for F in F_vals:
            for k in k_vals:
                final = rollout_surrogate(model, F.reshape(1), k.reshape(1), seed_state, n_steps)
                loss = pattern_loss(final[:, 1], target).item()
                if loss < best_loss:
                    best_loss = loss
                    best = (float(F), float(k))
    return best


@dataclass
class InverseResult:
    F: float
    k: float
    F_history: list[float] = field(default_factory=list)
    k_history: list[float] = field(default_factory=list)
    loss_history: list[float] = field(default_factory=list)


def inverse_design(
    model: nn.Module,
    target: torch.Tensor,
    seed_state: torch.Tensor,
    F_init: float | None = None,
    k_init: float | None = None,
    F_range: tuple[float, float] = F_RANGE,
    k_range: tuple[float, float] = K_RANGE,
    n_steps: int = 80,
    n_iters: int = 200,
    lr: float = 0.01,
    use_checkpoint: bool = False,
    grid_n: int = 10,
) -> InverseResult:
    """Given a target pattern, find the (F, k) that produce it. The MODEL IS
    FROZEN throughout — only F and k are optimized. If no (F_init, k_init)
    is given, a coarse grid search picks the starting point (grid-then-
    gradient, per the spec's recommendation)."""
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    if F_init is None or k_init is None:
        F_init, k_init = grid_search(model, target, seed_state, F_range, k_range, n_grid=grid_n, n_steps=n_steps)

    F_val = torch.tensor([F_init], dtype=torch.float32, requires_grad=True)
    k_val = torch.tensor([k_init], dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.Adam([F_val, k_val], lr=lr)

    result = InverseResult(F=F_init, k=k_init)
    for _ in range(n_iters):
        final_state = rollout_surrogate(model, F_val, k_val, seed_state, n_steps, use_checkpoint=use_checkpoint)
        loss = pattern_loss(final_state[:, 1], target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            F_val.clamp_(*F_range)
            k_val.clamp_(*k_range)

        result.F_history.append(F_val.item())
        result.k_history.append(k_val.item())
        result.loss_history.append(loss.item())

    result.F = F_val.item()
    result.k = k_val.item()
    return result


def multi_start_inverse_design(
    model: nn.Module,
    target: torch.Tensor,
    seed_state: torch.Tensor,
    n_starts: int = 4,
    jitter: float = 0.004,
    rng: np.random.Generator | None = None,
    F_range: tuple[float, float] = F_RANGE,
    k_range: tuple[float, float] = K_RANGE,
    **inverse_design_kwargs,
) -> InverseResult:
    """Run inverse_design from several starting points and keep the best
    (lowest final loss) — the spec's cheap, honest fix for the (F,k) loss
    landscape's local minima, since there are only two parameters to
    restart. One grid search locates a promising region; the remaining
    starts jitter around it so gradient descent explores its neighborhood
    from slightly different points rather than repeating the identical
    optimization trajectory `n_starts` times.
    """
    if rng is None:
        rng = np.random.default_rng()

    grid_n = inverse_design_kwargs.pop("grid_n", 10)
    n_steps = inverse_design_kwargs.get("n_steps", 80)
    F0, k0 = grid_search(model, target, seed_state, F_range, k_range, n_grid=grid_n, n_steps=n_steps)

    starts = [(F0, k0)]
    for _ in range(n_starts - 1):
        F_j = float(np.clip(F0 + rng.normal(0, jitter), *F_range))
        k_j = float(np.clip(k0 + rng.normal(0, jitter), *k_range))
        starts.append((F_j, k_j))

    results = [
        inverse_design(
            model, target, seed_state,
            F_init=F, k_init=k, F_range=F_range, k_range=k_range,
            **inverse_design_kwargs,
        )
        for F, k in starts
    ]
    return min(results, key=lambda r: r.loss_history[-1])
