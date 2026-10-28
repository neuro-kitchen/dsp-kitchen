"""`dspk.ops.filter`: FIR filtering, `firwin` and Gaussian smoothing against scipy."""

import numpy as np
import pytest
from scipy import ndimage
from scipy import signal as sp

import dspk
from conftest import RUNTIMES

FS = 30_000.0
TOLERANCE = 1e-5  # of the signal peak (float32 sums on the device vs float64)


@pytest.fixture(scope="module")
def data() -> np.ndarray:
    rng = np.random.default_rng(3)
    return (50 * rng.standard_normal((3, 4_001))).astype(np.float32)


def rel(got, expected) -> float:
    return float(np.max(np.abs(got - expected)) / np.max(np.abs(expected)))


def test_firwin_matches_scipy():
    np.testing.assert_allclose(dspk.ops.filter.firwin(61, 0.2), sp.firwin(61, 0.2), atol=1e-12)
    np.testing.assert_allclose(dspk.ops.filter.firwin(61, 3_000.0, sample_rate=FS), sp.firwin(61, 3_000.0, fs=FS), atol=1e-12)
    np.testing.assert_allclose(
        dspk.ops.filter.firwin(41, 0.3, window="kaiser", beta=5.0), sp.firwin(41, 0.3, window=("kaiser", 5.0)), atol=1e-12
    )
    with pytest.raises(ValueError, match="needs beta"):
        dspk.ops.filter.firwin(41, 0.3, window="kaiser")
    with pytest.raises(ValueError, match="Nyquist"):
        dspk.ops.filter.firwin(41, 20_000.0, sample_rate=FS)


@pytest.mark.parametrize("runtime", RUNTIMES)
def test_causal_fir_matches_lfilter(data, runtime):
    taps = np.array([0.5, -0.25, 0.125, 0.3, -0.1], dtype=np.float32)  # asymmetric: checks the direction
    expected = sp.lfilter(taps.astype(np.float64), 1.0, data.astype(np.float64), axis=1)
    assert rel(dspk.ops.filter.fir(data, taps, runtime=runtime), expected) < TOLERANCE


@pytest.mark.parametrize("runtime", RUNTIMES)
def test_centered_fir_matches_correlate1d(data, runtime):
    taps = np.array([0.1, -0.2, 0.6, 0.3, 0.05], dtype=np.float32)  # asymmetric
    expected = ndimage.correlate1d(data.astype(np.float64), taps.astype(np.float64), axis=1, mode="constant", cval=0.0)
    got = dspk.ops.filter.fir(data, taps, centered=True, edge="zeros", runtime=runtime)
    assert rel(got, expected) < TOLERANCE


@pytest.mark.parametrize("runtime", RUNTIMES)
def test_gaussian_smooth_matches_gaussian_filter1d(data, runtime):
    sigma = 3.5
    expected = ndimage.gaussian_filter1d(
        data.astype(np.float64), sigma, axis=1, mode="reflect", truncate=dspk.ops.filter.GAUSSIAN_TRUNCATE
    )
    assert rel(dspk.ops.filter.gaussian_smooth(data, sigma, runtime=runtime), expected) < TOLERANCE


def test_fir_arguments_are_checked(data):
    with pytest.raises(ValueError, match="odd number"):
        dspk.ops.filter.fir(data, np.ones(4), centered=True)
    with pytest.raises(ValueError, match="positive"):
        dspk.ops.filter.gaussian_smooth(data, 0.0)
