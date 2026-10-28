"""`dspk.ops.spatial` whitening and Laplacian, alone and as pipeline stages."""

import numpy as np
import pytest

import dspk
from conftest import RUNTIMES
from dspk.ops import Pipeline, stage

FS = 30_000.0


@pytest.fixture(scope="module")
def data() -> np.ndarray:
    rng = np.random.default_rng(11)
    mixing = np.eye(6) + 0.6 * rng.standard_normal((6, 6))
    return (mixing @ rng.standard_normal((6, 30_000))).astype(np.float32)


@pytest.mark.parametrize("runtime", RUNTIMES)
def test_zca_whitens_and_matches_numpy(data, runtime):
    w = dspk.ops.spatial.Whitening.fit_zca(data, runtime=runtime)
    values, vectors = np.linalg.eigh(np.cov(data.astype(np.float64), bias=True))
    expected = vectors @ np.diag(1.0 / np.sqrt(values + dspk.ops.spatial.DEFAULT_WHITENING_EPSILON)) @ vectors.T
    np.testing.assert_allclose(w.matrix, expected, atol=1e-3 * np.abs(expected).max())
    white = w.apply(data, runtime=runtime)
    np.testing.assert_allclose(np.cov(white.astype(np.float64), bias=True), np.eye(6), atol=1e-3)


@pytest.mark.parametrize("runtime", RUNTIMES)
def test_laplacian_grid_and_as_a_stage(data, runtime):
    lap = dspk.ops.spatial.Laplacian.from_grid(2, 3)
    m = lap.matrix.astype(np.float64)
    np.testing.assert_allclose(m.sum(axis=1), 0.0, atol=1e-6)  # each channel minus its neighbours' mean
    expected = m @ data.astype(np.float64)
    assert np.max(np.abs(lap.apply(data, runtime=runtime) - expected)) < 1e-4
    out = Pipeline([stage.laplacian(lap)], FS, runtime=runtime).process(data)
    assert np.max(np.abs(out - expected)) < 1e-4


@pytest.mark.parametrize("runtime", RUNTIMES)
def test_local_knn_whitening_as_a_stage(data, runtime):
    positions = np.stack([np.arange(6) * 20.0, np.zeros(6)], axis=1)
    w = dspk.ops.spatial.Whitening.fit_local_knn(data, positions, 3, runtime=runtime)
    out = Pipeline([stage.whiten(w)], FS, runtime=runtime).process(data)
    assert np.max(np.abs(out - w.matrix.astype(np.float64) @ data.astype(np.float64))) < 1e-3


def test_channel_mismatch_is_a_value_error(data):
    lap = dspk.ops.spatial.Laplacian.from_grid(2, 3)
    with pytest.raises(ValueError, match="for 6 channels"):
        Pipeline([stage.laplacian(lap)], FS).process(data[:4])
    with pytest.raises(ValueError, match="positions must have shape"):
        dspk.ops.spatial.Whitening.fit_local_knn(data, np.zeros((5, 2)), 3)
