import torch

from graydiff.inverse import (
    grid_search,
    inverse_design,
    multi_start_inverse_design,
    rollout_surrogate,
    seed_state_tensor,
)
from graydiff.model import Surrogate


def _tiny_model_and_seed():
    torch.manual_seed(0)
    model = Surrogate(hidden=6)
    seed = seed_state_tensor(grid_size=12)
    return model, seed


def test_seed_state_tensor_shape():
    seed = seed_state_tensor(grid_size=16)
    assert seed.shape == (1, 2, 16, 16)


def test_rollout_surrogate_shape():
    model, seed = _tiny_model_and_seed()
    F_val = torch.tensor([0.045])
    k_val = torch.tensor([0.06])
    out = rollout_surrogate(model, F_val, k_val, seed, n_steps=3)
    assert out.shape == seed.shape


def test_checkpointed_rollout_matches_non_checkpointed():
    """Gradient checkpointing must be numerically equivalent, not just
    memory-cheaper -- confirm forward AND backward agree."""
    model, seed = _tiny_model_and_seed()
    F_val = torch.tensor([0.045], requires_grad=True)
    k_val = torch.tensor([0.06], requires_grad=True)
    out1 = rollout_surrogate(model, F_val, k_val, seed, n_steps=4, use_checkpoint=False)
    loss1 = out1.pow(2).mean()
    loss1.backward()
    grad_F1, grad_k1 = F_val.grad.clone(), k_val.grad.clone()

    F_val2 = torch.tensor([0.045], requires_grad=True)
    k_val2 = torch.tensor([0.06], requires_grad=True)
    out2 = rollout_surrogate(model, F_val2, k_val2, seed, n_steps=4, use_checkpoint=True)
    loss2 = out2.pow(2).mean()
    loss2.backward()

    assert torch.allclose(out1, out2, atol=1e-6)
    assert torch.allclose(grad_F1, F_val2.grad, atol=1e-5)
    assert torch.allclose(grad_k1, k_val2.grad, atol=1e-5)


def test_grid_search_returns_value_in_range():
    model, seed = _tiny_model_and_seed()
    target = torch.rand(1, 12, 12)
    F0, k0 = grid_search(model, target, seed, F_range=(0.02, 0.07), k_range=(0.05, 0.07), n_grid=3, n_steps=2)
    assert 0.02 - 1e-6 <= F0 <= 0.07 + 1e-6
    assert 0.05 - 1e-6 <= k0 <= 0.07 + 1e-6


def test_inverse_design_smoke():
    model, seed = _tiny_model_and_seed()
    target = torch.rand(1, 12, 12)
    result = inverse_design(
        model, target, seed,
        F_init=0.04, k_init=0.06,
        F_range=(0.02, 0.07), k_range=(0.05, 0.07),
        n_steps=3, n_iters=5, lr=0.01,
    )
    assert 0.02 - 1e-6 <= result.F <= 0.07 + 1e-6
    assert 0.05 - 1e-6 <= result.k <= 0.07 + 1e-6
    assert len(result.loss_history) == 5
    assert all(torch.isfinite(torch.tensor(v)) for v in result.loss_history)
    # params should actually have moved from init
    assert (result.F != 0.04) or (result.k != 0.06)


def test_inverse_design_params_stay_within_clamp_bounds():
    model, seed = _tiny_model_and_seed()
    target = torch.rand(1, 12, 12)
    result = inverse_design(
        model, target, seed,
        F_init=0.069, k_init=0.0695,  # start near the boundary
        F_range=(0.02, 0.07), k_range=(0.05, 0.07),
        n_steps=2, n_iters=10, lr=0.05,
    )
    assert all(0.02 - 1e-6 <= f <= 0.07 + 1e-6 for f in result.F_history)
    assert all(0.05 - 1e-6 <= k <= 0.07 + 1e-6 for k in result.k_history)


def test_multi_start_returns_best_of_starts():
    model, seed = _tiny_model_and_seed()
    target = torch.rand(1, 12, 12)
    result = multi_start_inverse_design(
        model, target, seed,
        n_starts=3, grid_n=3,
        F_range=(0.02, 0.07), k_range=(0.05, 0.07),
        n_steps=2, n_iters=3, lr=0.01,
    )
    assert 0.02 - 1e-6 <= result.F <= 0.07 + 1e-6
    assert 0.05 - 1e-6 <= result.k <= 0.07 + 1e-6
