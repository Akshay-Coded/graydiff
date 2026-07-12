"""Phase 3: training the surrogate.

Two stages, run in sequence (see notebook 03):

  1. Single-step warm-up (`train_single_step`): MSE between the model's
     one-step prediction and the solver's true next state, on individual
     (state, F, k) -> next_state pairs. Fast, and gets the model into a
     reasonable basin before the more expensive stage below.

  2. Multi-step rollout training (`train_rollout`): unroll the model
     `rollout_len` steps AUTOREGRESSIVELY — feeding its own predicted state
     back in as the next input — and backprop through the whole unrolled
     trajectory against the solver's true trajectory. This is what teaches
     the model to stay stable when fed its own imperfect predictions.
     Skipping it is not a minor quality loss: without it, long autoregressive
     rollouts (Phase 4's stability test, Phase 4's phase-diagram match, and
     Phase 5's inverse-design optimization, which backprops through exactly
     this kind of rollout) drift and the resulting gradients become
     meaningless.

Both stages hold out entire (F, k) VALUES for validation (via
graydiff.data.train_val_split_by_run), not just held-out frames of seen
trajectories — the point is to know the model interpolates across phase
space, since the inverse optimizer (Phase 5) will wander through (F, k)
values it never trained on exactly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from graydiff.model import make_input


@dataclass
class TrainHistory:
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    lr: list[float] = field(default_factory=list)
    best_epoch: int = -1


def _single_step_loss(model: nn.Module, batch, device: torch.device) -> torch.Tensor:
    state, F_val, k_val, next_state = batch
    state, F_val, k_val, next_state = (
        state.to(device), F_val.to(device), k_val.to(device), next_state.to(device)
    )
    pred = model(make_input(state, F_val, k_val))
    return nn.functional.mse_loss(pred, next_state)


def _rollout_loss(model: nn.Module, batch, device: torch.device, rollout_len: int) -> torch.Tensor:
    traj, F_val, k_val = batch
    traj, F_val, k_val = traj.to(device), F_val.to(device), k_val.to(device)
    state = traj[:, 0]
    total = state.new_zeros(())
    for t in range(rollout_len):
        state = model(make_input(state, F_val, k_val))
        total = total + nn.functional.mse_loss(state, traj[:, t + 1])
    return total / rollout_len


def _run_training_loop(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    loss_fn,
    epochs: int,
    lr: float,
    patience: int,
) -> TrainHistory:
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2)
    history = TrainHistory()
    best_val = float("inf")
    best_state = None
    epochs_since_improvement = 0

    for epoch in range(epochs):
        model.train()
        train_losses = []
        for batch in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model, batch, device)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                val_losses.append(loss_fn(model, batch, device).item())

        train_loss = sum(train_losses) / len(train_losses)
        val_loss = sum(val_losses) / len(val_losses)
        scheduler.step(val_loss)

        history.train_loss.append(train_loss)
        history.val_loss.append(val_loss)
        history.lr.append(optimizer.param_groups[0]["lr"])

        if val_loss < best_val - 1e-9:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            history.best_epoch = epoch
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1
            if epochs_since_improvement >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return history


def train_single_step(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 15,
    lr: float = 1e-3,
    patience: int = 4,
) -> TrainHistory:
    """Stage 1: single-step MSE warm-up."""
    return _run_training_loop(model, train_loader, val_loader, device, _single_step_loss, epochs, lr, patience)


def train_rollout(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    rollout_len: int = 6,
    epochs: int = 15,
    lr: float = 5e-4,
    patience: int = 4,
) -> TrainHistory:
    """Stage 2: multi-step rollout training, backpropagating through the
    full autoregressive unroll."""
    loss_fn = lambda model, batch, device: _rollout_loss(model, batch, device, rollout_len)
    return _run_training_loop(model, train_loader, val_loader, device, loss_fn, epochs, lr, patience)
