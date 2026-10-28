"""`dspk.ops.resample` against scipy.signal."""

import numpy as np
import pytest
from scipy import signal as sp

import dspk
from conftest import RUNTIMES


@pytest.fixture(scope="module")
def data() -> np.ndarray:
    t = np.arange(12_000) / 30_000.0
    return np.stack([np.sin(2 * np.pi * 300 * (c + 1) * t) + 0.2 * np.cos(2 * np.pi * 37 * t) for c in range(3)]).astype(np.float32)


@pytest.mark.parametrize("runtime", RUNTIMES)
@pytest.mark.parametrize(("up", "down"), [(1, 3), (2, 3), (3, 2)])
def test_resample_poly_matches_scipy(data, runtime, up, down):
    expected = sp.resample_poly(data.astype(np.float64), up, down, axis=1)
    got = dspk.ops.resample.resample_poly(data, up, down, runtime=runtime)
    assert got.shape == expected.shape
    assert np.max(np.abs(got - expected)) < 1e-5


@pytest.mark.parametrize("runtime", RUNTIMES)
def test_decimate_fir_matches_scipy(data, runtime):
    expected = sp.decimate(data.astype(np.float64), 4, ftype="fir", axis=1)
    got = dspk.ops.resample.decimate(data, 4, ftype="fir", runtime=runtime)
    assert got.shape == expected.shape and np.max(np.abs(got - expected)) < 1e-5


@pytest.mark.parametrize("runtime", RUNTIMES)
def test_decimate_iir_matches_scipy_away_from_the_ends(data, runtime):
    expected = sp.decimate(data.astype(np.float64), 4, ftype="iir", axis=1)
    got = dspk.ops.resample.decimate(data, 4, runtime=runtime)
    assert got.shape == expected.shape
    interior = slice(200, -200)  # the padding length differs from scipy's (documented)
    assert np.max(np.abs(got[:, interior] - expected[:, interior])) < 1e-4


def test_defaults_come_from_dspk_ops():
    r = dspk.ops.resample
    assert (r.DECIMATE_DEFAULT_ORDER, r.DECIMATE_DEFAULT_RIPPLE_DB, r.RESAMPLE_POLY_BETA) == (8, 0.05, 5.0)
