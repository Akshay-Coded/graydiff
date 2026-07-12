import pytest
import torch

from graydiff.model import CircularConv2d, N_OUTPUT_CHANNELS, Surrogate, make_input


def test_output_shape():
    model = Surrogate(hidden=8)
    x = torch.randn(3, 4, 16, 16)
    y = model(x)
    assert y.shape == (3, N_OUTPUT_CHANNELS, 16, 16)


def test_make_input_broadcasts_scalars():
    state = torch.randn(2, 2, 8, 8)
    F_val = torch.tensor([0.035, 0.029])
    k_val = torch.tensor([0.065, 0.057])
    x = make_input(state, F_val, k_val)
    assert x.shape == (2, 4, 8, 8)
    assert torch.allclose(x[0, 2], torch.full((8, 8), 0.035))
    assert torch.allclose(x[1, 3], torch.full((8, 8), 0.057))


def test_circular_padding_translation_equivariance():
    """Rolling the input spatially should roll the output identically —
    this directly regression-tests that the boundary condition matches the
    solver's periodic Laplacian. A model with mismatched (e.g. zero) padding
    would NOT have this property."""
    torch.manual_seed(0)
    conv = CircularConv2d(4, 4)
    x = torch.randn(1, 4, 12, 12)
    shift = (3, -2)
    y1 = conv(x)
    y2 = conv(torch.roll(x, shifts=shift, dims=(2, 3)))
    assert torch.allclose(torch.roll(y1, shifts=shift, dims=(2, 3)), y2, atol=1e-5)


def test_full_model_translation_equivariance():
    torch.manual_seed(0)
    model = Surrogate(hidden=8)
    model.eval()
    x = torch.randn(1, 4, 12, 12)
    shift = (2, 5)
    y1 = model(x)
    y2 = model(torch.roll(x, shifts=shift, dims=(2, 3)))
    assert torch.allclose(torch.roll(y1, shifts=shift, dims=(2, 3)), y2, atol=1e-4)


def test_gradient_flows_to_F_and_k():
    """The single most load-bearing test in the suite: the entire
    inverse-design mechanism (Phase 5) depends on d(loss)/dF and d(loss)/dk
    being well-defined and nonzero through the surrogate."""
    model = Surrogate(hidden=8)
    state = torch.rand(1, 2, 10, 10)
    F_val = torch.tensor([0.045], requires_grad=True)
    k_val = torch.tensor([0.060], requires_grad=True)

    x = make_input(state, F_val, k_val)
    out = model(x)
    loss = out.pow(2).mean()
    loss.backward()

    assert F_val.grad is not None
    assert k_val.grad is not None
    assert F_val.grad.abs().item() > 0
    assert k_val.grad.abs().item() > 0


def test_residual_output_close_to_input_for_zero_weights():
    """Sanity check on the residual wiring: if the conv stack contributes
    (near) nothing, the output should equal the input U, V channels."""
    model = Surrogate(hidden=4)
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
    x = torch.rand(1, 4, 8, 8)
    y = model(x)
    assert torch.allclose(y, x[:, :N_OUTPUT_CHANNELS], atol=1e-6)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_forward_backward_on_mps():
    device = torch.device("mps")
    model = Surrogate(hidden=8).to(device)
    x = torch.randn(2, 4, 16, 16, device=device, requires_grad=True)
    y = model(x)
    loss = y.pow(2).mean()
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(y).all()
