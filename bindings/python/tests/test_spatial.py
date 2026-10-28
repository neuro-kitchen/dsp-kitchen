"""`dspk.ops.spatial`: common average and median reference against NumPy."""

import numpy as np
import pytest

import dspk
from conftest import RUNTIMES


@pytest.mark.parametrize("runtime", RUNTIMES)
@pytest.mark.parametrize("channels", [5, 32])  # odd and even channel counts for the median
def test_car_and_cmr(runtime, channels):
    rng = np.random.default_rng(5)
    x = (10 * rng.standard_normal((channels, 3_000)) + 7.0).astype(np.float32)
    car = x.astype(np.float64) - x.astype(np.float64).mean(axis=0, keepdims=True)
    cmr = x.astype(np.float64) - np.median(x.astype(np.float64), axis=0, keepdims=True)
    np.testing.assert_allclose(dspk.ops.spatial.car(x, runtime=runtime), car, atol=1e-4)
    np.testing.assert_allclose(dspk.ops.spatial.cmr(x, runtime=runtime), cmr, atol=1e-4)
