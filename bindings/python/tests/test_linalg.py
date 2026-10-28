"""`dspk.ops.linalg` PCA and PPCA against NumPy (covariance `/ samples`, as np.cov(bias=True))."""

import numpy as np
import pytest

import dspk
from conftest import RUNTIMES


@pytest.fixture(scope="module")
def data() -> np.ndarray:
    rng = np.random.default_rng(10)
    latent = rng.standard_normal((3, 20_000))
    mixing = rng.standard_normal((8, 3)) * np.array([5.0, 2.0, 1.0])
    return (mixing @ latent + 0.3 * rng.standard_normal((8, 20_000)) + 4.0).astype(np.float32)


def eig_desc(x):
    values, vectors = np.linalg.eigh(np.cov(x.astype(np.float64), bias=True))
    return values[::-1], vectors[:, ::-1]


@pytest.mark.parametrize("runtime", RUNTIMES)
def test_pca_matches_numpy(data, runtime):
    pca = dspk.ops.linalg.PCA.fit(data, 3, runtime=runtime)
    values, vectors = eig_desc(data)
    np.testing.assert_allclose(pca.explained_variance, values[:3], rtol=1e-3)
    np.testing.assert_allclose(pca.explained_variance_ratio, values[:3] / values.sum(), rtol=1e-3)
    signs = np.sign(np.sum(pca.components * vectors[:, :3], axis=0))  # components up to sign
    np.testing.assert_allclose(pca.components * signs, vectors[:, :3], atol=1e-3)
    np.testing.assert_allclose(pca.mean, data.mean(axis=1), rtol=1e-5)
    z = pca.transform(data, runtime=runtime)
    expected = pca.components.T.astype(np.float64) @ (data - pca.mean[:, None]).astype(np.float64)
    assert z.shape == (3, data.shape[1])
    assert np.max(np.abs(z - expected)) / np.max(np.abs(expected)) < 1e-4


@pytest.mark.parametrize("runtime", RUNTIMES)
def test_ppca_noise_is_the_mean_discarded_eigenvalue(data, runtime):
    ppca = dspk.ops.linalg.PPCA.fit(data, 3, runtime=runtime)
    values, _ = eig_desc(data)
    assert ppca.noise_variance == pytest.approx(values[3:].mean(), rel=1e-3)
    z = ppca.transform(data, runtime=runtime)
    w, s2 = ppca.weights.astype(np.float64), float(ppca.noise_variance)
    posterior = np.linalg.solve(w.T @ w + s2 * np.eye(3), w.T)  # (WᵀW + σ²I)⁻¹ Wᵀ
    expected = posterior @ (data - ppca.mean[:, None]).astype(np.float64)
    assert np.max(np.abs(z - expected)) / np.max(np.abs(expected)) < 1e-3


def test_fit_arguments_are_checked(data):
    with pytest.raises(ValueError, match="n_components"):
        dspk.ops.linalg.PCA.fit(data, 9)
    pca = dspk.ops.linalg.PCA.fit(data, 2)
    with pytest.raises(ValueError, match="fitted on 8 channels"):
        pca.transform(data[:4])
