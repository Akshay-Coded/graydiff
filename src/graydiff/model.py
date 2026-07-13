"""Phase 2: the neural surrogate architecture.

Three choices here do real work, per the spec, and all three are load-bearing
for the inverse-design result later:

  circular padding — Gray-Scott uses wrap-around (periodic) boundaries, so the
  convolutions must wrap too, exactly matching graydiff.solver.laplacian's
  np.roll boundary handling. Mismatching the boundary condition teaches the
  model wrong edge physics.

  residual output — the state barely changes per step (confirmed empirically
  in notebook 01: max |delta V| across the entire training set never
  exceeds ~0.015), so predicting the small delta and adding it to the input
  is far easier and far more stable than predicting the whole next frame
  from scratch.

  BOUNDED residual delta — notebook 04's first attempt at long autoregressive
  rollouts revealed that an *unbounded* residual delta is not enough on its
  own: fed its own output for hundreds of steps, the raw architecture above
  drifted onto inputs unlike anything in training and started emitting large,
  runaway per-step corrections there, diverging exponentially within a few
  hundred steps. Because the true physical per-step change is always small
  (empirically bounded, see above), the delta is passed through a
  `delta_scale * tanh(...)` before being added — a hard, physically-motivated
  cap on how much the state can change in one step, chosen generously above
  the observed data (see DEFAULT_DELTA_SCALE below) so it doesn't constrain
  normal dynamics, but making runaway exponential growth mathematically
  impossible: each step can move the state by at most `delta_scale`,
  regardless of what the input looks like.

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

# The largest per-step |delta V| observed anywhere in the generated training
# set (graydiff.data) is ~0.015. This default is set well above that (~13x)
# so the cap doesn't bind during normal dynamics, while still making runaway
# per-step growth mathematically impossible during long autoregressive
# rollouts (notebook 04).
DEFAULT_DELTA_SCALE = 0.2


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

    def __init__(self, hidden: int = 64, delta_scale: float = DEFAULT_DELTA_SCALE):
        super().__init__()
        self.delta_scale = delta_scale
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
        # residual: predict a BOUNDED delta (see DEFAULT_DELTA_SCALE above),
        # add it to the current (U, V). tanh(0) == 0, so an untrained
        # (zero-init) network still reduces exactly to the identity map.
        delta = self.delta_scale * torch.tanh(self.net(x))
        return x[:, :N_OUTPUT_CHANNELS] + delta


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
