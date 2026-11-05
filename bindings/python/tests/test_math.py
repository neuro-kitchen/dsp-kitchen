"""`dspk.ops.math` against NumPy / scipy."""

import numpy as np
import pytest
from scipy import signal as sp

import dspk
from conftest import RUNTIMES


@pytest.mark.parametrize("runtime", RUNTIMES)
def test_channel_mean_std(runtime):
    rng = np.random.default_rng(7)
    x = (rng.standard_normal((5, 10_001)) * np.arange(1, 6)[:, None] + 3.0).astype(np.float32)
    mean, std = dspk.ops.math.channel_mean_std(x, runtime=runtime)
    np.testing.assert_allclose(mean, x.astype(np.float64).mean(axis=1), rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(std, x.astype(np.float64).std(axis=1), rtol=1e-4)


@pytest.mark.parametrize("runtime", RUNTIMES)
def test_channel_noise_std_is_the_mad_estimate(runtime):
    rng = np.random.default_rng(8)
    x = (rng.standard_normal((4, 20_001)) * 5.0).astype(np.float32)  # odd length: one middle element
    x[:, ::500] += 200.0  # spikes barely move a median
    expected = np.sort(np.abs(x), axis=1)[:, x.shape[1] // 2] / dspk.ops.math.MAD_TO_SIGMA
    np.testing.assert_allclose(dspk.ops.math.channel_noise_std(x, runtime=runtime), expected, rtol=1e-6)


@pytest.mark.parametrize("runtime", RUNTIMES)
@pytest.mark.parametrize("samples", [20_001, 20_000])  # odd and even: NumPy's median of each
def test_channel_mad_matches_numpy(runtime, samples):
    rng = np.random.default_rng(9)
    x = (rng.standard_normal((4, samples)) * 3.0 + 1.5).astype(np.float32)
    x64 = x.astype(np.float64)
    expected = np.median(np.abs(x64 - np.median(x64, axis=1, keepdims=True)), axis=1)
    np.testing.assert_allclose(dspk.ops.math.channel_mad(x, runtime=runtime), expected, rtol=1e-5)


def test_windows_match_scipy():
    w = dspk.ops.math.window
    np.testing.assert_allclose(w("hann", 64), sp.get_window("hann", 64, fftbins=False), atol=1e-6)
    np.testing.assert_allclose(w("hamming", 65), sp.get_window("hamming", 65, fftbins=False), atol=1e-6)
    np.testing.assert_allclose(w("kaiser", 51, beta=6.0), sp.get_window(("kaiser", 6.0), 51, fftbins=False), atol=1e-10)
    with pytest.raises(ValueError, match="needs sigma"):
        w("gaussian", 31)
