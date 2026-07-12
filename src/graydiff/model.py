"""Phase 2: the neural surrogate architecture.

Three choices here do real work, per the spec, and all three are load-bearing
for the inverse-design result later:

  circular padding — Gray-Scott uses wrap-around (periodic) boundaries, so the
  convolutions must wrap too, exactly matching graydiff.solver.laplacian's
  np.roll boundary handling. Mismatching the boundary condition teaches the
  model wrong edge physics.

  residual output — the state barely changes per step (confirmed empirically
  in notebook 01: max |delta V| per step is small relative to V's [0,1]
  range), so predicting the small delta and adding it to the input is far
  easier and far more stable than predicting the whole next frame from
  scratch. This single choice is what makes long autoregressive rollouts
  (Phase 3/4) and long-rollout inverse-design optimization (Phase 5) stable.

  (F, k) as input channels — broadcasting the two scalar physics parameters
  to full grids and concatenating them with (U, V) is the thing that makes
  inverse design possible at all: it makes the network's output a
  differentiable function of F and k, so d(output)/dF exists.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

N_INPUT_CHANNELS = 4  # U, V, F_grid, k_grid
N_OUTPUT_CHANNELS = 2  # U_next, V_next


class CircularConv2d(nn.Module):
    """A 3x3 conv with manual circular (wrap-around) padding.

    Implemented as explicit F.pad(mode='circular') + a padding=0 Conv2d,
    rather than relying solely on nn.Conv2d(padding_mode='circular'):
    this is more robust across backends with incomplete op coverage (Apple's
    MPS backend historically lags on some padding-mode combinations) and
    exports more predictably to ONNX (the built-in circular padding mode has
    had inconsistent support in the ONNX exporter across torch versions).
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd for symmetric padding")
        self.pad = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self.pad, self.pad, self.pad, self.pad), mode="circular")
        return self.conv(x)


class Surrogate(nn.Module):
    """Input : [B, 4, H, W] = (U, V, F_grid, k_grid)
    Output: [B, 2, H, W] = (U_next, V_next)
    """

    def __init__(self, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            CircularConv2d(N_INPUT_CHANNELS, hidden),
            nn.GELU(),
            CircularConv2d(hidden, hidden),
            nn.GELU(),
            CircularConv2d(hidden, hidden),
            nn.GELU(),
            CircularConv2d(hidden, N_OUTPUT_CHANNELS),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # residual: predict the DELTA, add it to the current (U, V)
        return x[:, :N_OUTPUT_CHANNELS] + self.net(x)


def make_input(state: torch.Tensor, F_val: torch.Tensor, k_val: torch.Tensor) -> torch.Tensor:
    """Build the [B, 4, H, W] surrogate input from a [B, 2, H, W] state and
    scalar (or [B]-shaped) F, k by broadcasting them to constant-valued
    grids and concatenating. Shared by training, validation, and the inverse-
    design loop so the broadcasting logic exists in exactly one place.
    """
    B, _, H, W = state.shape
    F_grid = F_val.reshape(B, 1, 1, 1).expand(B, 1, H, W).to(state.dtype)
    k_grid = k_val.reshape(B, 1, 1, 1).expand(B, 1, H, W).to(state.dtype)
    return torch.cat([state, F_grid, k_grid], dim=1)
