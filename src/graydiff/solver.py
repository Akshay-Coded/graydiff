"""Ground-truth Gray-Scott reaction-diffusion solver (NumPy, float64).

This is the source of truth for the whole project: it generates the training
data the surrogate learns from, and it is the thing every recovered inverse-
design (F, k) gets verified against at the end. It stays in float64 and on
CPU deliberately — this is the one place in the project where numerical
precision must not be a variable.

Equations (periodic boundaries, 5-point Laplacian stencil):
    dU/dt = Du * lap(U) - U*V^2       + F*(1 - U)
    dV/dt = Dv * lap(V) + U*V^2       - (F + k)*V
"""

from __future__ import annotations

import numpy as np

from graydiff.constants import DEFAULT_GRID_SIZE, DU, DV


def laplacian(Z: np.ndarray) -> np.ndarray:
    """5-point stencil Laplacian with periodic (wrap-around) boundaries.

    ``roll`` in each of the four axis-directions implements the wrap-around;
    this must match the surrogate's circular padding exactly, or the network
    learns the wrong edge physics (see graydiff.model.CircularConv2d).
    """
    return (
        np.roll(Z, 1, axis=0)
        + np.roll(Z, -1, axis=0)
        + np.roll(Z, 1, axis=1)
        + np.roll(Z, -1, axis=1)
        - 4 * Z
    )


def gray_scott_step(
    U: np.ndarray,
    V: np.ndarray,
    Du: float = DU,
    Dv: float = DV,
    F: float = 0.060,
    k: float = 0.062,
) -> tuple[np.ndarray, np.ndarray]:
    """Advance (U, V) by one explicit Euler step (dt=1, absorbed into the rates).

    Mutates and returns U, V in place — callers that need the pre-step state
    preserved (e.g. training-data generation) must pass copies.
    """
    uvv = U * V * V
    U += Du * laplacian(U) - uvv + F * (1 - U)
    V += Dv * laplacian(V) + uvv - (F + k) * V
    return U, V


def rollout(
    U: np.ndarray,
    V: np.ndarray,
    n_steps: int,
    Du: float = DU,
    Dv: float = DV,
    F: float = 0.060,
    k: float = 0.062,
    save_every: int | None = None,
) -> tuple[np.ndarray, np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
    """Run n_steps of the solver from (U, V). Returns the final state and,
    if save_every is set, a list of (U, V) snapshots taken every save_every
    steps (including the initial state).
    """
    U, V = U.copy(), V.copy()
    snapshots: list[tuple[np.ndarray, np.ndarray]] = []
    for t in range(n_steps):
        if save_every is not None and t % save_every == 0:
            snapshots.append((U.copy(), V.copy()))
        U, V = gray_scott_step(U, V, Du=Du, Dv=Dv, F=F, k=k)
    return U, V, snapshots


def _blob_plus_global_noise(
    H: int,
    W: int,
    rng: np.random.Generator,
    n_blobs: int = 1,
    blob_r_frac: int = 10,
    noise_amp: float = 0.2,
    center: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Shared seeding recipe: a background of U=1, V=0 (the "empty dish" fixed
    point — see the module-level note on why this state is always locally
    stable) perturbed by square blobs of U=0.50, V=0.25 AND uniform noise
    across the *whole* domain, not just inside the blobs.

    Both ingredients matter and were tuned empirically against the spec's
    named checkpoints. (1,0) — the "empty dish" — is a linearly stable fixed
    point for every (F, k) > 0: the Jacobian of the reaction kinetics there,
    d(dU/dt)/d(U,V) and d(dV/dt)/d(U,V) evaluated at U=1,V=0, is diagonal
    with both eigenvalues negative (-F and -(F+k)). So Gray-Scott pattern
    formation is NOT a small-perturbation Turing instability growing out of
    the uniform state — it requires a large-enough nucleating disturbance to
    kick the system away from (1,0) at all. Empirically, a single blob alone
    reliably nucleates low-F "maze" regions but lets high-F "spot" regions
    decay straight back to (1,0); domain-wide noise alone does the reverse.
    Only the combination nucleates both reliably (verified against
    F=0.035,k=0.065 -> spots and F=0.029,k=0.057 -> mazes).
    """
    U = np.ones((H, W), dtype=np.float64)
    V = np.zeros((H, W), dtype=np.float64)

    for _ in range(n_blobs):
        r_h = max(H // blob_r_frac, 2)
        r_w = max(W // blob_r_frac, 2)
        cy = H // 2 if center else rng.integers(0, H)
        cx = W // 2 if center else rng.integers(0, W)
        rows = np.arange(cy - r_h, cy + r_h) % H
        cols = np.arange(cx - r_w, cx + r_w) % W
        U[np.ix_(rows, cols)] = 0.50
        V[np.ix_(rows, cols)] = 0.25

    U += noise_amp * (rng.random((H, W)) - 0.5)
    V += noise_amp * (rng.random((H, W)) - 0.5)
    np.clip(U, 0.0, 1.0, out=U)
    np.clip(V, 0.0, 1.0, out=V)
    return U, V


def random_seed(
    H: int = DEFAULT_GRID_SIZE,
    W: int = DEFAULT_GRID_SIZE,
    rng: np.random.Generator | None = None,
    n_blobs: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """A varied initial condition for TRAINING DATA generation: randomized
    blob placement/count plus domain-wide noise, so training data covers
    varied basins of attraction rather than one fixed starting shape.
    """
    if rng is None:
        rng = np.random.default_rng()
    if n_blobs is None:
        n_blobs = int(rng.integers(1, 4))
    return _blob_plus_global_noise(H, W, rng, n_blobs=n_blobs, center=False)


def standard_seed(H: int = DEFAULT_GRID_SIZE, W: int = DEFAULT_GRID_SIZE) -> tuple[np.ndarray, np.ndarray]:
    """A single fixed, DETERMINISTIC initial condition — one centered blob
    plus a fixed-seed noise field. Used everywhere a comparable starting
    point across runs matters: notebook 00's phase-diagram grid and,
    critically, the inverse-design optimization loop
    (graydiff.inverse.inverse_design), where every gradient step must start
    from the exact same state for the optimization to be well-posed.
    """
    rng = np.random.default_rng(0)
    return _blob_plus_global_noise(H, W, rng, n_blobs=1, center=True)
