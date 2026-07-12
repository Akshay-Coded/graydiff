"""Phase 5: turning a user-drawn sketch into a target field the inverse-
design loss can meaningfully compare against the surrogate's output.

A raw hand-drawn mask (crisp edges, effectively binary) has a very different
frequency signature than a real Gray-Scott field, which is smooth and
band-limited by diffusion. Comparing them directly under
graydiff.losses.pattern_loss (an FFT power-spectrum loss) would penalize the
target for being "too sharp" in a way that has nothing to do with the
physical pattern regime the user actually intended — the sharp edges inject
high-frequency content no real V field has. Lightly blurring the drawn mask
before using it as an optimization target brings it into the same rough
frequency range as an actual field.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def _gaussian_kernel1d(sigma: float, radius: int) -> torch.Tensor:
    x = torch.arange(-radius, radius + 1, dtype=torch.float32)
    kernel = torch.exp(-(x**2) / (2 * sigma**2))
    return kernel / kernel.sum()


def gaussian_blur(field: torch.Tensor, sigma: float = 1.5) -> torch.Tensor:
    """Separable Gaussian blur of a [H, W] field, with circular (wrap-around)
    padding to match the surrogate's own periodic-boundary convention."""
    radius = max(1, int(3 * sigma))
    kernel = _gaussian_kernel1d(sigma, radius).to(field.device)

    x = field[None, None]  # [1, 1, H, W]
    x = F.pad(x, (radius, radius, 0, 0), mode="circular")
    x = F.conv2d(x, kernel.view(1, 1, 1, -1))
    x = F.pad(x, (0, 0, radius, radius), mode="circular")
    x = F.conv2d(x, kernel.view(1, 1, -1, 1))
    return x[0, 0]


def preprocess_target(
    mask: np.ndarray | torch.Tensor,
    grid_size: int,
    blur_sigma: float = 1.5,
) -> torch.Tensor:
    """Convert a drawn mask (any input resolution, values in [0,1] or
    [0,255]) into a [grid_size, grid_size] float32 target field: resized to
    the model's grid, normalized to [0,1], and lightly blurred so its
    frequency content resembles a real V field rather than a crisp sketch.
    """
    if isinstance(mask, np.ndarray):
        mask = torch.from_numpy(mask)
    mask = mask.float()
    if mask.max() > 1.0:
        mask = mask / 255.0

    resized = F.interpolate(
        mask[None, None], size=(grid_size, grid_size), mode="bilinear", align_corners=False
    )[0, 0]
    blurred = gaussian_blur(resized, sigma=blur_sigma)
    span = blurred.max() - blurred.min()
    return (blurred - blurred.min()) / (span + 1e-8)
