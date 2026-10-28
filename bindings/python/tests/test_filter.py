"""`dspk.ops.filter` against `scipy.signal`: designs (float64) and device filtering (float32).

Limits as in the Rust tests (`crates/dspk-ops/tests/filter_scipy.rs`): device results within
2e-5 of the signal peak for mid-band cutoffs.
"""

import numpy as np
import pytest
from scipy import signal as sp

import dspk

FS = 30_000.0
DEVICE_TOLERANCE = 2e-5  # of the signal peak
DESIGN_RTOL = 1e-8

RUNTIMES = dspk.runtime.available()


@pytest.fixture(scope="module")
def data() -> np.ndarray:
    """3 channels × 20 000 samples: tones at 50 Hz, 1 kHz, 8 kHz plus noise and a DC offset."""
    rng = np.random.default_rng(7)
    t = np.arange(20_000) / FS
    tones = sum(np.sin(2 * np.pi * f * t) for f in (50.0, 1_000.0, 8_000.0))
    x = np.stack([tones * (c + 1) + 0.3 * rng.standard_normal(t.size) + 2.0 for c in range(3)])
    return x.astype(np.float32)


def max_rel_err(got, expected) -> float:
    return float(np.max(np.abs(got - expected)) / np.max(np.abs(expected)))


@pytest.mark.parametrize(
    ("band", "cutoff", "design", "extra", "scipy_design"),
    [
        ("bandpass", (300.0, 6_000.0), "butterworth", {}, lambda: sp.butter(5, (300.0, 6_000.0), "bandpass", fs=FS, output="sos")),
        ("highpass", 300.0, "butterworth", {}, lambda: sp.butter(5, 300.0, "highpass", fs=FS, output="sos")),
        ("lowpass", 2_000.0, "chebyshev1", {"ripple_db": 0.5}, lambda: sp.cheby1(5, 0.5, 2_000.0, "lowpass", fs=FS, output="sos")),
        ("bandstop", (45.0, 55.0), "bessel", {}, lambda: sp.bessel(5, (45.0, 55.0), "bandstop", fs=FS, output="sos")),
    ],
)
def test_design_matches_scipy(band, cutoff, design, extra, scipy_design):
    got = dspk.ops.filter.sos(band, cutoff, FS, design=design, **extra)
    np.testing.assert_allclose(got, scipy_design(), rtol=DESIGN_RTOL, atol=1e-12)


@pytest.mark.parametrize("runtime", RUNTIMES)
def test_zero_phase_bandpass_matches_sosfiltfilt(data, runtime):
    sos = sp.butter(5, (300.0, 6_000.0), "bandpass", fs=FS, output="sos")
    pad = min(dspk.ops.filter.settling_samples(sos), data.shape[1] - 1)
    expected = sp.sosfiltfilt(sos, data.astype(np.float64), axis=1, padtype="odd", padlen=pad)
    got = dspk.ops.filter.iir(data, FS, "bandpass", (300.0, 6_000.0), runtime=runtime)
    assert got.dtype == np.float32 and got.shape == data.shape
    assert max_rel_err(got, expected) < DEVICE_TOLERANCE


@pytest.mark.parametrize("runtime", RUNTIMES)
@pytest.mark.parametrize("start", ["rest", "steady_state"])
def test_causal_highpass_matches_sosfilt(data, runtime, start):
    sos = sp.butter(5, 300.0, "highpass", fs=FS, output="sos")
    x = data.astype(np.float64)
    if start == "rest":
        expected = sp.sosfilt(sos, x, axis=1)
    else:
        zi = sp.sosfilt_zi(sos)[:, None, :] * x[:, 0][None, :, None]
        expected, _ = sp.sosfilt(sos, x, axis=1, zi=zi)
    got = dspk.ops.filter.iir(data, FS, "highpass", 300.0, mode="forward", start=start, runtime=runtime)
    assert max_rel_err(got, expected) < DEVICE_TOLERANCE


@pytest.mark.parametrize("runtime", RUNTIMES)
def test_notch_and_explicit_sos(data, runtime):
    b, a = sp.iirnotch(60.0, 30.0, fs=FS)
    sos = sp.tf2sos(b, a)
    pad = min(dspk.ops.filter.settling_samples(sos), data.shape[1] - 1)
    expected = sp.sosfiltfilt(sos, data.astype(np.float64), axis=1, padtype="odd", padlen=pad)
    assert max_rel_err(dspk.ops.filter.notch(data, FS, 60.0, 30.0, runtime=runtime), expected) < DEVICE_TOLERANCE
    assert max_rel_err(dspk.ops.filter.sosfilt(data, sos, runtime=runtime), expected) < DEVICE_TOLERANCE


def test_bad_arguments(data):
    with pytest.raises(ValueError, match="two cutoff"):
        dspk.ops.filter.iir(data, FS, "bandpass", 300.0)
    with pytest.raises(ValueError, match="needs ripple_db"):
        dspk.ops.filter.sos("lowpass", 1_000.0, FS, design="chebyshev1")
    with pytest.raises(ValueError, match="unknown mode"):
        dspk.ops.filter.iir(data, FS, "lowpass", 1_000.0, mode="both")
    with pytest.raises(ValueError):
        dspk.ops.filter.iir(data, FS, "lowpass", 20_000.0)  # above Nyquist
    with pytest.raises(ValueError, match="shape"):
        dspk.ops.filter.sosfilt(data, np.ones((2, 5)))
