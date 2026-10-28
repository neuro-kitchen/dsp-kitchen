"""`dspk.ops.peaks.candidates` against a NumPy reference."""

import numpy as np
import pytest

import dspk
from conftest import RUNTIMES


def extrema(signed, height):
    mid = signed[1:-1]
    return (signed[:-2] < mid) & (mid >= signed[2:]) & (mid >= height)


def reference(x, heights, polarity):
    rows = []
    for c, row in enumerate(x.astype(np.float64)):
        maxima, minima = extrema(row, heights[c]), extrema(-row, heights[c])
        is_peak = {"positive": maxima, "negative": minima, "both": maxima | minima}[polarity]
        rows += [(c, t + 1) for t in np.flatnonzero(is_peak)]
    return rows


@pytest.mark.parametrize("runtime", RUNTIMES)
@pytest.mark.parametrize("polarity", ["positive", "negative", "both"])
def test_candidates_match_reference(runtime, polarity):
    rng = np.random.default_rng(9)
    x = rng.standard_normal((4, 5_001)).astype(np.float32)
    heights = np.array([1.5, 2.0, 1.0, 2.5], dtype=np.float32)
    got = dspk.ops.peaks.candidates(x, heights, polarity=polarity, runtime=runtime)
    assert list(zip(got["channel"].tolist(), got["sample"].tolist())) == reference(x, heights, polarity)
    np.testing.assert_array_equal(got["value"], x[got["channel"], got["sample"]])
