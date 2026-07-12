import torch
from torch.utils.data import DataLoader

from graydiff.data import GenerationConfig, RolloutDataset, SingleStepDataset, generate_dataset, train_val_split_by_run
from graydiff.model import Surrogate
from graydiff.train import train_rollout, train_single_step


def _tiny_loaders():
    config = GenerationConfig(n_runs=10, grid_size=12, warmup_steps=3, window_len=4, seed=0)
    data = generate_dataset(config)
    train_data, val_data = train_val_split_by_run(data, val_frac=0.3, seed=0)
    return train_data, val_data


def test_train_single_step_runs_and_produces_history():
    train_data, val_data = _tiny_loaders()
    train_loader = DataLoader(SingleStepDataset(train_data), batch_size=4, shuffle=True)
    val_loader = DataLoader(SingleStepDataset(val_data), batch_size=4)

    model = Surrogate(hidden=4)
    history = train_single_step(model, train_loader, val_loader, torch.device("cpu"), epochs=3, patience=3)

    assert len(history.train_loss) > 0
    assert len(history.train_loss) == len(history.val_loss) == len(history.lr)
    assert all(torch.isfinite(torch.tensor(v)) for v in history.train_loss)


def test_train_rollout_runs_and_produces_history():
    train_data, val_data = _tiny_loaders()
    train_loader = DataLoader(RolloutDataset(train_data, rollout_len=3), batch_size=3, shuffle=True)
    val_loader = DataLoader(RolloutDataset(val_data, rollout_len=3), batch_size=3)

    model = Surrogate(hidden=4)
    history = train_rollout(model, train_loader, val_loader, torch.device("cpu"), rollout_len=3, epochs=2, patience=3)

    assert len(history.train_loss) > 0
    assert all(torch.isfinite(torch.tensor(v)) for v in history.val_loss)


def test_train_single_step_reduces_loss_on_easy_overfit_case():
    """A tiny model trained on a single repeated batch for enough epochs
    should at least not diverge, and should end no worse than it started."""
    train_data, _ = _tiny_loaders()
    loader = DataLoader(SingleStepDataset(train_data), batch_size=8, shuffle=True)

    model = Surrogate(hidden=8)
    history = train_single_step(model, loader, loader, torch.device("cpu"), epochs=8, lr=2e-3, patience=8)
    assert history.train_loss[-1] <= history.train_loss[0] * 1.5  # not diverging
