"""Loads the trained surrogate once at process startup and reuses it across
requests, rather than reloading from disk on every call."""

from __future__ import annotations

from pathlib import Path

import torch

from graydiff.model import Surrogate

GRID = 64
N_STEPS = 120  # matches the horizon validated in notebooks 04/05 -- comfortably
                # inside the surrogate's confirmed-stable window

_REPO_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_PATH = _REPO_ROOT / "models" / "checkpoints" / "surrogate_rollout.pt"

_model: Surrogate | None = None
_device: torch.device | None = None


def get_device() -> torch.device:
    global _device
    if _device is None:
        # Measured (not assumed): a single inverse-design request (n_starts=2,
        # n_iters=30, n_steps=120) took 55s on CPU vs 12s on MPS on this
        # machine -- MPS wins even at batch size 1, consistent with notebook
        # 03's training-time finding. Falls back to CPU automatically on a
        # deploy target without MPS.
        _device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    return _device


def get_model() -> Surrogate:
    global _model
    if _model is None:
        model = Surrogate(hidden=64)
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location="cpu"))
        model.to(get_device())
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        _model = model
    return _model
