"""`dspk.ops.filter`: median and Teager–Kaiser against scipy / NumPy."""

import numpy as np
import pytest
from scipy import signal as sp

import dspk
from conftest import RUNTIMES


@pytest.fixture(scope="module")
def data() -> np.ndarray:
    rng = np.random.default_rng(4)
    x = rng.standard_normal((4, 2_003)).astype(np.float32)
    x[:, ::97] += 40.0  # impulses the median removes
    return x


@pytest.mark.parametrize("runtime", RUNTIMES)
@pytest.mark.parametrize("width", [3, 9, 15])
def test_median_matches_medfilt(data, runtime, width):
    expected = np.stack([sp.medfilt(row.astype(np.float64), width) for row in data])
    np.testing.assert_array_equal(dspk.ops.filter.median(data, width, runtime=runtime), expected.astype(np.float32))


@pytest.mark.parametrize("runtime", RUNTIMES)
def test_teager_kaiser_with_mirrored_ends(data, runtime):
    x = np.pad(data.astype(np.float64), ((0, 0), (1, 1)), mode="symmetric")  # x0 x0 x1 ... repeats the edge
    expected = x[:, 1:-1] ** 2 - x[:, :-2] * x[:, 2:]
    got = dspk.ops.filter.teager_kaiser(data, runtime=runtime)
    assert np.max(np.abs(got - expected)) / np.max(np.abs(expected)) < 1e-6


def test_median_width_is_checked(data):
    with pytest.raises(ValueError, match="odd"):
        dspk.ops.filter.median(data, 4)
    with pytest.raises(ValueError, match="at most"):
        dspk.ops.filter.median(data, dspk.ops.filter.MAX_MEDIAN_WIDTH + 2)
