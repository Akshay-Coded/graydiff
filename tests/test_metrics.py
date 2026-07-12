import numpy as np
import torch

from graydiff.metrics import (
    nearest_training_distance,
    rollout_error_curve,
    surrogate_rollout_trajectory,
    time_fn,
)
from graydiff.model import Surrogate


def test_surrogate_rollout_trajectory_shape():
    model = Surrogate(hidden=4)
    seed = torch.rand(1, 2, 8, 8)
    F_val = torch.tensor([0.045])
    k_val = torch.tensor([0.06])
    traj = surrogate_rollout_trajectory(model, F_val, k_val, seed, n_steps=5)
    assert traj.shape == (6, 2, 8, 8)
    assert torch.allclose(traj[0], seed[0])


def test_rollout_error_curve_zero_for_identical_trajectories():
    traj = torch.rand(4, 2, 8, 8)
    err = rollout_error_curve(traj, traj.numpy())
    assert err.shape == (4,)
    assert np.allclose(err, 0.0)


def test_rollout_error_curve_nonzero_for_different_trajectories():
    traj_a = torch.zeros(3, 2, 4, 4)
    traj_b = np.ones((3, 2, 4, 4))
    err = rollout_error_curve(traj_a, traj_b)
    assert np.all(err > 0)


def test_nearest_training_distance():
    train_F = np.array([0.02, 0.05, 0.07])
    train_k = np.array([0.05, 0.06, 0.07])
    d = nearest_training_distance(0.05, 0.06, train_F, train_k)
    assert d == 0.0
    d2 = nearest_training_distance(0.0, 0.0, train_F, train_k)
    assert d2 > 0


def test_time_fn_returns_positive_float():
    result = time_fn(lambda: sum(range(1000)), n_repeats=2)
    assert isinstance(result, float)
    assert result >= 0.0
