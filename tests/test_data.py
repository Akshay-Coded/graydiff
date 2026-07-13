import numpy as np
import pytest
import torch

from graydiff.data import (
    GenerationConfig,
    RolloutDataset,
    SingleStepDataset,
    generate_dataset,
    load_dataset,
    save_dataset,
    train_val_split_by_run,
)


@pytest.fixture
def tiny_data():
    config = GenerationConfig(
        n_runs=6, grid_size=16, warmup_range=(0, 5), window_len=4, seed=0
    )
    return generate_dataset(config), config


def test_generate_dataset_shapes(tiny_data):
    data, config = tiny_data
    n_states = config.window_len + 1
    assert data["states"].shape == (config.n_runs, n_states, 2, config.grid_size, config.grid_size)
    assert data["F"].shape == (config.n_runs,)
    assert data["k"].shape == (config.n_runs,)
    assert data["states"].dtype == np.float32


def test_generate_dataset_params_within_range(tiny_data):
    data, config = tiny_data
    assert np.all(data["F"] >= config.F_range[0]) and np.all(data["F"] <= config.F_range[1])
    assert np.all(data["k"] >= config.k_range[0]) and np.all(data["k"] <= config.k_range[1])


def test_generate_dataset_reproducible():
    config = GenerationConfig(n_runs=3, grid_size=16, warmup_range=(0, 5), window_len=3, seed=7)
    d1 = generate_dataset(config)
    d2 = generate_dataset(config)
    assert np.array_equal(d1["states"], d2["states"])
    assert np.array_equal(d1["F"], d2["F"])


def test_save_load_roundtrip(tmp_path, tiny_data):
    data, _ = tiny_data
    path = tmp_path / "ds.npz"
    save_dataset(data, path)
    loaded = load_dataset(path)
    assert np.array_equal(data["states"], loaded["states"])
    assert np.array_equal(data["F"], loaded["F"])
    assert np.array_equal(data["k"], loaded["k"])


def test_train_val_split_disjoint_and_covers_all(tiny_data):
    data, config = tiny_data
    train, val = train_val_split_by_run(data, val_frac=0.34, seed=1)
    assert train["F"].shape[0] + val["F"].shape[0] == config.n_runs
    # every (F, k) run pair appears in exactly one split
    train_pairs = set(zip(train["F"].tolist(), train["k"].tolist()))
    val_pairs = set(zip(val["F"].tolist(), val["k"].tolist()))
    assert train_pairs.isdisjoint(val_pairs)


def test_single_step_dataset_shapes(tiny_data):
    data, config = tiny_data
    ds = SingleStepDataset(data)
    assert len(ds) == config.n_runs * config.window_len
    state, F, k, next_state = ds[0]
    assert state.shape == (2, config.grid_size, config.grid_size)
    assert next_state.shape == (2, config.grid_size, config.grid_size)
    assert isinstance(F, torch.Tensor) and F.ndim == 0
    assert isinstance(k, torch.Tensor) and k.ndim == 0


def test_single_step_dataset_consecutive_pairs_match_states(tiny_data):
    data, config = tiny_data
    ds = SingleStepDataset(data)
    # idx=0 -> run 0, t=0: state should equal states[0,0], next_state states[0,1]
    state, _, _, next_state = ds[0]
    assert torch.allclose(state, torch.from_numpy(data["states"][0, 0]))
    assert torch.allclose(next_state, torch.from_numpy(data["states"][0, 1]))


def test_rollout_dataset_shapes(tiny_data):
    data, config = tiny_data
    ds = RolloutDataset(data, rollout_len=3)
    assert len(ds) == config.n_runs
    traj, F, k = ds[0]
    assert traj.shape == (4, 2, config.grid_size, config.grid_size)


def test_rollout_dataset_rejects_too_long_rollout(tiny_data):
    data, config = tiny_data
    with pytest.raises(ValueError):
        RolloutDataset(data, rollout_len=config.window_len + 1)
