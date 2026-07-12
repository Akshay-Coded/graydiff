import numpy as np

from graydiff.phase_classify import classify_pattern


def test_dead_field():
    assert classify_pattern(np.zeros((32, 32))) == "dead"


def test_uniform_field():
    assert classify_pattern(np.full((32, 32), 0.5)) == "uniform"


def test_synthetic_stripes():
    H = W = 64
    _, x = np.mgrid[0:H, 0:W]
    field = (np.sin(2 * np.pi * x / 8) > 0).astype(float) * 0.3
    assert classify_pattern(field) == "stripes"


def test_synthetic_spots():
    H = W = 64
    y, x = np.mgrid[0:H, 0:W]
    field = np.zeros((H, W))
    rng = np.random.default_rng(0)
    for _ in range(12):
        cy, cx = rng.integers(0, H), rng.integers(0, W)
        r2 = (y - cy) ** 2 + (x - cx) ** 2
        field += np.exp(-r2 / (2 * 3.0**2))
    assert classify_pattern(field) == "spots"
