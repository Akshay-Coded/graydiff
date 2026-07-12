import numpy as np
import pytest

from graydiff.constants import MAZES_CHECKPOINT, SPOTS_CHECKPOINT
from graydiff.phase_classify import classify_pattern
from graydiff.solver import gray_scott_step, laplacian, random_seed, rollout, standard_seed


def test_laplacian_zero_on_uniform_field():
    Z = np.full((8, 8), 3.7)
    assert np.allclose(laplacian(Z), 0.0)


def test_laplacian_hand_computed_case():
    # A single 1 at the center of a zero field: periodic 5-point stencil
    # gives -4 at the center and +1 at each of its four neighbours.
    Z = np.zeros((5, 5))
    Z[2, 2] = 1.0
    lap = laplacian(Z)
    assert lap[2, 2] == pytest.approx(-4.0)
    assert lap[1, 2] == pytest.approx(1.0)
    assert lap[3, 2] == pytest.approx(1.0)
    assert lap[2, 1] == pytest.approx(1.0)
    assert lap[2, 3] == pytest.approx(1.0)
    assert lap.sum() == pytest.approx(0.0)  # conservation: periodic Laplacian sums to zero


def test_laplacian_wraps_across_boundary():
    Z = np.zeros((5, 5))
    Z[0, 0] = 1.0
    lap = laplacian(Z)
    # neighbour "above" row 0 wraps to row -1 == row 4
    assert lap[4, 0] == pytest.approx(1.0)
    assert lap[0, 4] == pytest.approx(1.0)


def test_gray_scott_step_deterministic(rng):
    U0, V0 = random_seed(16, 16, rng=np.random.default_rng(42))
    U1, V1 = gray_scott_step(U0.copy(), V0.copy(), F=0.035, k=0.065)
    U2, V2 = gray_scott_step(U0.copy(), V0.copy(), F=0.035, k=0.065)
    assert np.array_equal(U1, U2)
    assert np.array_equal(V1, V2)


def test_gray_scott_step_bounded_for_stable_params():
    U, V = standard_seed(32, 32)
    U, V, _ = rollout(U, V, n_steps=500, F=0.035, k=0.065)
    assert np.all(np.isfinite(U))
    assert np.all(np.isfinite(V))
    assert U.min() > -1.0 and U.max() < 2.0
    assert V.min() > -1.0 and V.max() < 2.0


def test_standard_seed_is_deterministic():
    U1, V1 = standard_seed(32, 32)
    U2, V2 = standard_seed(32, 32)
    assert np.array_equal(U1, U2)
    assert np.array_equal(V1, V2)


@pytest.mark.slow
def test_spots_checkpoint_matches_phase_diagram():
    U, V = standard_seed(64, 64)
    _, Vf, _ = rollout(U, V, n_steps=10000, **{"F": SPOTS_CHECKPOINT["F"], "k": SPOTS_CHECKPOINT["k"]})
    assert classify_pattern(Vf) == "spots"


@pytest.mark.slow
def test_mazes_checkpoint_matches_phase_diagram():
    U, V = standard_seed(64, 64)
    _, Vf, _ = rollout(U, V, n_steps=10000, **{"F": MAZES_CHECKPOINT["F"], "k": MAZES_CHECKPOINT["k"]})
    assert classify_pattern(Vf) == "mazes"
