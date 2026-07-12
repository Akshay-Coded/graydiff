"""Phase 5: a pattern-CHARACTER loss for inverse design.

Pixel-MSE is the WRONG loss here, and the reason is specific: two patterns of
the same regime (e.g. both stripes) but offset in space are, under periodic
boundaries, physically the same pattern — Gray-Scott has no preferred origin.
Pixel-MSE would score them as almost completely different, so gradient
descent on (F, k) would get a misleading gradient that chases exact pixel
alignment instead of the correct physical regime.

The fix: compare spatial-frequency content instead of pixel position. Whether
a pattern is spots, stripes, or mazes is fundamentally a frequency-domain
signature — a characteristic wavelength (blob spacing / stripe width) and a
degree of isotropy (spots and mazes are roughly isotropic; stripes are not).
The 2D FFT power spectrum captures exactly that, and — critically — is
naturally translation-invariant: |FFT(shift(x))| = |FFT(x)|, so a shifted
target produces the same target power spectrum. It's also fully
differentiable (built from FFT + elementwise ops), so gradients flow back
through it to F and k exactly as the pixel-MSE alternative would, just
without the misleading component.
"""

from __future__ import annotations

import torch
import torch.nn.functional as Fnn


def fft_power_spectrum(field: torch.Tensor) -> torch.Tensor:
    """2D FFT power spectrum (|FFT|^2), zero-frequency centered. `field` is
    [..., H, W] (e.g. [B, H, W] or [H, W]) — a single channel, typically V."""
    spectrum = torch.fft.fft2(field)
    spectrum = torch.fft.fftshift(spectrum, dim=(-2, -1))
    return spectrum.real**2 + spectrum.imag**2


def pattern_loss(
    field: torch.Tensor,
    target: torch.Tensor,
    freq_weight: float = 1.0,
    stat_weight: float = 0.1,
) -> torch.Tensor:
    """Pattern-character loss between a produced field and a target field.

    Two terms:
      - freq_weight * MSE(log power spectrum): the primary signal. Log-scaled
        so the loss isn't dominated purely by the DC/low-frequency component,
        which would otherwise swamp the higher-frequency structure that
        actually distinguishes spots from stripes from mazes.
      - stat_weight * (mean, std) MSE: a small regularizer that keeps the
        optimizer honest about overall coverage/intensity level, which the
        frequency spectrum alone under-constrains (a spectrum is invariant
        to the field's mean, for instance).
    """
    log_field = torch.log1p(fft_power_spectrum(field))
    log_target = torch.log1p(fft_power_spectrum(target))
    freq_loss = Fnn.mse_loss(log_field, log_target)

    dims = tuple(range(-2, 0))
    stat_loss = (field.mean(dim=dims) - target.mean(dim=dims)).pow(2).mean() + (
        field.std(dim=dims) - target.std(dim=dims)
    ).pow(2).mean()

    return freq_weight * freq_loss + stat_weight * stat_loss
