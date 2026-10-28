"""Integer arrays cross to the device as their integers and are converted there: every op gives
exactly the result of the same values as float32 (int16 → float32 is exact)."""

import numpy as np
import pytest

import dspk
from conftest import RUNTIMES
from dspk.ops import Pipeline, stage

FS = 30_000.0
ops = dspk.ops


@pytest.fixture(scope="module", params=["int8", "int16", "uint16", "int32"])
def stored(request) -> np.ndarray:
    info = np.iinfo(request.param)
    rng = np.random.default_rng(6)
    lo, hi = max(info.min, -20_000), min(info.max, 20_000)
    return rng.integers(lo, hi, size=(4, 6_000), endpoint=True).astype(request.param)


CASES = {
    "iir": lambda x, rt: ops.filter.iir(x, FS, "highpass", 300.0, runtime=rt),
    "median": lambda x, rt: ops.filter.median(x, 5, runtime=rt),
    "car": lambda x, rt: ops.spatial.car(x, runtime=rt),
    "resample_poly": lambda x, rt: ops.resample.resample_poly(x, 1, 3, runtime=rt),
    "mean_std": lambda x, rt: np.stack(ops.math.channel_mean_std(x, runtime=rt)),
    "pipeline": lambda x, rt: Pipeline([stage.iir("highpass", 300.0, mode="forward"), stage.car()], FS, runtime=rt).process(x),
    "pca": lambda x, rt: ops.linalg.PCA.fit(x, 2, runtime=rt).transform(x, runtime=rt),
}


@pytest.mark.parametrize("runtime", RUNTIMES)
@pytest.mark.parametrize("name", CASES)
def test_integer_input_equals_float32_input(stored, runtime, name):
    run = CASES[name]
    got = run(stored, runtime)
    assert got.dtype == np.float32
    np.testing.assert_array_equal(got, run(stored.astype(np.float32), runtime))


def test_other_dtypes_become_float32_on_the_host():
    x = np.random.default_rng(7).standard_normal((2, 500))
    np.testing.assert_array_equal(ops.spatial.car(x), ops.spatial.car(x.astype(np.float32)))  # float64
    np.testing.assert_array_equal(ops.spatial.car(x.tolist()), ops.spatial.car(x.astype(np.float32)))  # list
