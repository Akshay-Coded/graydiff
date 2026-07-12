import torch

from graydiff.losses import pattern_loss


def _stripes(H=32, W=32, period=8, shift=0):
    # Clean integer-modulo square wave (avoids floating-point zero-crossing
    # artifacts that sin(...) > 0 hits exactly at multiples of the period).
    x = torch.arange(W)
    row = (((x + shift) % period) < period // 2).float()
    return row.unsqueeze(0).expand(H, W).clone()


def test_zero_loss_for_identical_fields():
    field = _stripes()
    loss = pattern_loss(field, field.clone())
    assert loss.item() < 1e-6


def test_near_translation_invariant():
    """The whole justification for this loss over pixel-MSE: a shifted
    version of the same pattern should score as nearly identical, while
    plain pixel-MSE would score it as very different."""
    field = _stripes(shift=0)
    shifted = _stripes(shift=4)  # half a period shift

    fft_loss = pattern_loss(field, shifted).item()
    pixel_mse = (field - shifted).pow(2).mean().item()

    assert pixel_mse > 0.4  # genuinely different at the pixel level
    assert fft_loss < 0.05  # but recognized as the same pattern character


def test_different_pattern_types_score_higher_than_shifted_same_type():
    stripes = _stripes(period=8)
    fine_stripes = _stripes(period=2)  # much higher spatial frequency
    shifted_stripes = _stripes(period=8, shift=3)

    loss_same_type = pattern_loss(stripes, shifted_stripes).item()
    loss_diff_type = pattern_loss(stripes, fine_stripes).item()
    assert loss_diff_type > loss_same_type


def test_gradient_flows_through_loss():
    field = torch.rand(1, 16, 16, requires_grad=True)
    target = torch.rand(1, 16, 16)
    loss = pattern_loss(field, target)
    loss.backward()
    assert field.grad is not None
    assert torch.isfinite(field.grad).all()
