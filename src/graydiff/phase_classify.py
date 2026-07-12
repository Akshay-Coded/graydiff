"""Heuristic pattern-type classifier for a Gray-Scott V-field snapshot.

This is intentionally coarse: it exists to put a label on a (F, k) grid cell
for bookkeeping (phase-space coverage histograms in notebook 01, automated
"does the surrogate land in the same regime as the solver" checks in
notebook 04), not to be a rigorous pattern-recognition model. The real
validation of pattern type is always the visual thumbnail grid a human looks
at in notebook 00 — this classifier's job is to agree with that visual
judgement often enough to be useful as a summary statistic.

Method, in two stages:
  1. Spots vs. everything else: threshold V at its own mean, label connected
     components under periodic boundaries. Spots show up as many small,
     roughly-square-bounding-box, well-filled components. Stripes and mazes
     instead tend to form one (or a few) components whose bounding box spans
     most of the domain — a single winding path is highly connected, so its
     GLOBAL shape covariance is a bad elongation signal (a labyrinth that
     turns in all directions looks "isotropic" in aggregate even though it
     is locally thin and elongated everywhere). Component compactness sidesteps
     that: it only asks "is this blob small and roughly round", not "what is
     its overall aspect ratio".
  2. Stripes vs. mazes, once we know it isn't spots: compute the LOCAL
     gradient orientation at every foreground pixel (via Sobel filters) and
     measure how consistent that orientation is across the whole field
     (circular mean resultant length, angle doubled since orientation is
     only defined mod pi). Stripes point the same way everywhere -> high
     consistency. Mazes wind and branch -> low consistency. This is a local
     measurement, so it doesn't suffer from the same "global shape hides
     local structure" problem as stage 1 would if applied here.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

PatternLabel = str  # one of: "dead", "uniform", "spots", "stripes", "mazes"

_DEAD_MEAN_THRESHOLD = 0.02
_LOW_VARIANCE_STD = 1e-4
_COVERAGE_RANGE = (0.03, 0.90)
_MIN_COMPONENT_PIXELS = 3
_WRAP_PAD = 4

# A component counts as "compact" (spot-like) if its bounding box's larger
# side is under this fraction of the grid size AND it fills more than
# _MIN_FILL_FRACTION of that bounding box (rules out thin diagonal slivers).
_COMPACT_BBOX_FRACTION = 0.40
_MIN_FILL_FRACTION = 0.35
# If at least this fraction of foreground-component count is compact, and
# there are multiple of them, call the whole field "spots".
_MIN_COMPACT_COMPONENTS = 3

_ORIENTATION_CONSISTENCY_STRIPES_CUTOFF = 0.5


def _periodic_components(mask: np.ndarray) -> np.ndarray:
    """Label connected components of a binary mask under periodic boundaries
    by wrap-padding before labeling, then cropping back to the original
    footprint (a component that straddles the domain edge gets merged
    correctly instead of being cut into two separate labels)."""
    padded = np.pad(mask, _WRAP_PAD, mode="wrap")
    labeled, _ = ndimage.label(padded)
    return labeled[_WRAP_PAD:-_WRAP_PAD, _WRAP_PAD:-_WRAP_PAD]


def _is_compact(coords: np.ndarray, H: int, W: int) -> bool:
    bbox_h = coords[:, 0].max() - coords[:, 0].min() + 1
    bbox_w = coords[:, 1].max() - coords[:, 1].min() + 1
    if max(bbox_h, bbox_w) > _COMPACT_BBOX_FRACTION * max(H, W):
        return False
    fill_fraction = coords.shape[0] / (bbox_h * bbox_w)
    return fill_fraction > _MIN_FILL_FRACTION


def _orientation_consistency(V: np.ndarray, mask: np.ndarray) -> float:
    """Circular consistency of local gradient orientation over the
    foreground: 1.0 means every foreground pixel's edge points the same
    direction (a stripe field); near 0.0 means orientations are scattered
    (a maze)."""
    gy = ndimage.sobel(V, axis=0, mode="wrap")
    gx = ndimage.sobel(V, axis=1, mode="wrap")
    angle = np.arctan2(gy, gx)
    magnitude = np.hypot(gy, gx)

    weights = magnitude * mask
    if weights.sum() < 1e-9:
        return 0.0
    doubled = 2 * angle
    resultant = np.sum(weights * np.exp(1j * doubled)) / weights.sum()
    return float(np.abs(resultant))


def classify_pattern(V: np.ndarray) -> PatternLabel:
    """Classify a single V-field snapshot into a coarse Gray-Scott pattern
    regime. See module docstring for method and caveats."""
    V = np.asarray(V)
    H, W = V.shape

    if V.std() < _LOW_VARIANCE_STD:
        return "dead" if V.mean() < _DEAD_MEAN_THRESHOLD else "uniform"

    mask = V > V.mean()
    coverage = mask.mean()
    if coverage < _COVERAGE_RANGE[0] or coverage > _COVERAGE_RANGE[1]:
        return "uniform"

    labeled = _periodic_components(mask)
    n_labels = labeled.max()
    if n_labels == 0:
        return "uniform"

    n_compact = 0
    compact_area = 0
    total_area = 0
    for label_id in range(1, n_labels + 1):
        coords = np.argwhere(labeled == label_id)
        if coords.shape[0] < _MIN_COMPONENT_PIXELS:
            continue
        total_area += coords.shape[0]
        if _is_compact(coords, H, W):
            n_compact += 1
            compact_area += coords.shape[0]

    if total_area == 0:
        return "uniform"

    if n_compact >= _MIN_COMPACT_COMPONENTS and compact_area / total_area > 0.5:
        return "spots"

    consistency = _orientation_consistency(V, mask)
    return "stripes" if consistency > _ORIENTATION_CONSISTENCY_STRIPES_CUTOFF else "mazes"
