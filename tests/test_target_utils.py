import numpy as np
import torch

from graydiff.target_utils import gaussian_blur, preprocess_target


def test_preprocess_target_shape_and_range():
    mask = np.zeros((100, 100), dtype=np.float32)
    mask[30:70, 30:70] = 255.0
    field = preprocess_target(mask, grid_size=32)
    assert field.shape == (32, 32)
    assert field.min() >= 0.0 and field.max() <= 1.0 + 1e-5


def test_preprocess_target_normalizes_01_input():
    mask = np.zeros((20, 20), dtype=np.float32)
    mask[5:15, 5:15] = 1.0
    field = preprocess_target(mask, grid_size=20)
    assert field.max() > 0.5


def test_gaussian_blur_reduces_high_frequency_content():
    field = torch.zeros(32, 32)
    field[::2, :] = 1.0  # sharp alternating rows -- lots of high-frequency content
    blurred = gaussian_blur(field, sigma=2.0)
    sharp_variation = (field[1:] - field[:-1]).abs().mean()
    blurred_variation = (blurred[1:] - blurred[:-1]).abs().mean()
    assert blurred_variation < sharp_variation


def test_gaussian_blur_is_periodic():
    field = torch.zeros(16, 16)
    field[0, 0] = 1.0
    blurred = gaussian_blur(field, sigma=1.5)
    # energy from the spike at [0,0] should leak into the wrapped neighbour [15,0]
    assert blurred[15, 0] > 1e-4
