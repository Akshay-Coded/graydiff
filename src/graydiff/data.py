"""Phase 1: training-data generation across the (F, k) phase space.

The single most important design decision in the project lives downstream of
this module, in graydiff.model: the surrogate must see (F, k) as INPUT
channels, not be trained at one fixed value, or there is nothing to take a
gradient with respect to and inverse design is impossible. This module is
where that decision starts — every generated sample carries its own (F, k)
alongside the state.

Each run saves a short, CONTIGUOUS trajectory window (not an isolated
single-step pair), starting after a RANDOMIZED warm-up period (sampled per
run from `warmup_range`, which includes 0). A window of `window_len + 1`
consecutive solver states gives every single-step pair within it for free
(Phase 3's warm-up training) AND every rollout length up to `window_len` as
a sub-slice (Phase 3's multi-step rollout training) — one generation pass
serves both training regimes.

The warm-up is randomized, not fixed, for a reason discovered the hard way
(see notebook 04): every downstream use of the trained surrogate — the
phase-diagram-match validation, the inverse-design optimization — starts its
own rollout from `graydiff.solver.standard_seed()` at t=0, i.e. from the RAW,
freshly-nucleating initial blob. A first version of this module always used
a FIXED warmup_steps=2000, so every training window started well past
nucleation and the model never saw t=0-ish dynamics at all during training —
a real train/inference distribution mismatch that manifested as immediate,
(F,k)-independent high-frequency noise the moment the trained model was
rolled out autoregressively from a fresh seed. Sampling the warmup per run
from a range that includes 0 gives the model broad exposure to every stage
of the trajectory, nucleation included.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from graydiff.constants import DEFAULT_GRID_SIZE, DU, DV, F_RANGE, K_RANGE
from graydiff.solver import gray_scott_step, random_seed


@dataclass
class GenerationConfig:
    n_runs: int = 2000
    grid_size: int = DEFAULT_GRID_SIZE
    warmup_range: tuple[int, int] = (0, 2000)
    window_len: int = 10  # states saved per run = window_len + 1
    Du: float = DU
    Dv: float = DV
    F_range: tuple[float, float] = field(default_factory=lambda: F_RANGE)
    k_range: tuple[float, float] = field(default_factory=lambda: K_RANGE)
    seed: int = 0


def generate_dataset(config: GenerationConfig) -> dict[str, np.ndarray]:
    """Run `config.n_runs` independent (F, k, random seed) rollouts. Each is
    warmed up for a RANDOM number of steps (sampled per run from
    `warmup_range`, which includes 0 -- see module docstring) then
    snapshotted over a contiguous window of `window_len + 1` states.

    Returns:
        states: [n_runs, window_len+1, 2, H, W] float32 (channel 0=U, 1=V)
        F, k:   [n_runs] float32
    """
    rng = np.random.default_rng(config.seed)
    H = W = config.grid_size
    n_states = config.window_len + 1
    states = np.empty((config.n_runs, n_states, 2, H, W), dtype=np.float32)
    Fs = np.empty(config.n_runs, dtype=np.float32)
    ks = np.empty(config.n_runs, dtype=np.float32)

    for i in range(config.n_runs):
        F = float(rng.uniform(*config.F_range))
        k = float(rng.uniform(*config.k_range))
        U, V = random_seed(H, W, rng=rng)
        warmup_steps = int(rng.integers(config.warmup_range[0], config.warmup_range[1] + 1))
        for _ in range(warmup_steps):
            U, V = gray_scott_step(U, V, Du=config.Du, Dv=config.Dv, F=F, k=k)
        for t in range(n_states):
            states[i, t, 0] = U
            states[i, t, 1] = V
            if t < config.window_len:
                U, V = gray_scott_step(U, V, Du=config.Du, Dv=config.Dv, F=F, k=k)
        Fs[i] = F
        ks[i] = k

    return {"states": states, "F": Fs, "k": ks}


def save_dataset(data: dict[str, np.ndarray], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **data)


def load_dataset(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path) as f:
        return {key: f[key] for key in f.files}


def train_val_split_by_run(
    data: dict[str, np.ndarray], val_frac: float = 0.15, seed: int = 0
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Split by RUN (not by individual frame), so validation (F, k) values
    are entirely unseen during training — F, k are sampled continuously, so
    distinct runs have distinct (F, k) with probability 1. This is what lets
    Phase 3/4 honestly claim the model was validated on held-out physics
    parameters, not just held-out frames of the same trajectories.
    """
    n_runs = data["F"].shape[0]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_runs)
    n_val = int(n_runs * val_frac)
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    train = {key: arr[train_idx] for key, arr in data.items()}
    val = {key: arr[val_idx] for key, arr in data.items()}
    return train, val


class SingleStepDataset(Dataset):
    """Every consecutive (state, F, k) -> next_state pair drawn from the
    saved trajectory windows. Used for Phase 3's single-step warm-up."""

    def __init__(self, data: dict[str, np.ndarray]):
        self.states = data["states"]  # [N, T+1, 2, H, W]
        self.F = data["F"]
        self.k = data["k"]
        self.n_runs, n_states = self.states.shape[:2]
        self.window_len = n_states - 1

    def __len__(self) -> int:
        return self.n_runs * self.window_len

    def __getitem__(self, idx: int):
        run_idx, t = divmod(idx, self.window_len)
        state = torch.from_numpy(self.states[run_idx, t])
        next_state = torch.from_numpy(self.states[run_idx, t + 1])
        F = torch.tensor(float(self.F[run_idx]), dtype=torch.float32)
        k = torch.tensor(float(self.k[run_idx]), dtype=torch.float32)
        return state, F, k, next_state


class RolloutDataset(Dataset):
    """Full-window trajectories per run, for multi-step rollout training
    (Phase 3): unroll the model `rollout_len` steps and compare against
    these true intermediate solver states."""

    def __init__(self, data: dict[str, np.ndarray], rollout_len: int | None = None):
        self.states = data["states"]
        self.F = data["F"]
        self.k = data["k"]
        self.n_runs, n_states = self.states.shape[:2]
        self.max_rollout_len = n_states - 1
        self.rollout_len = rollout_len or self.max_rollout_len
        if self.rollout_len > self.max_rollout_len:
            raise ValueError(
                f"rollout_len={self.rollout_len} exceeds the saved window length={self.max_rollout_len}"
            )

    def __len__(self) -> int:
        return self.n_runs

    def __getitem__(self, idx: int):
        traj = torch.from_numpy(self.states[idx, : self.rollout_len + 1])  # [rollout_len+1, 2, H, W]
        F = torch.tensor(float(self.F[idx]), dtype=torch.float32)
        k = torch.tensor(float(self.k[idx]), dtype=torch.float32)
        return traj, F, k
