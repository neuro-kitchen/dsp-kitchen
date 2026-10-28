"""Models fitted on the device: PCA and probabilistic PCA, scikit-learn style.

Data is ``(channels, samples)``: channels are the features, samples the observations. FastICA
follows once its iterations run on the device.
"""

from __future__ import annotations

import numpy as np

from dspk._native import ops as _ops
from dspk.ops.filter._common import _channels

_linalg = _ops.linalg


class PCA:
    """Principal component analysis; covariance and eigendecomposition on the device."""

    def __init__(self, native) -> None:
        self._native = native

    @classmethod
    def fit(cls, data, n_components: int, *, runtime: str | None = None) -> PCA:
        """Fits ``n_components`` components on ``data`` ``(channels, samples)``."""
        return cls(_linalg.Pca.fit(_channels(data), n_components, runtime=runtime))

    @property
    def components(self) -> np.ndarray:
        """``(channels, n_components)``: one component per column."""
        return self._native.components

    @property
    def mean(self) -> np.ndarray:
        return self._native.mean

    @property
    def explained_variance(self) -> np.ndarray:
        return self._native.explained_variance

    @property
    def explained_variance_ratio(self) -> np.ndarray:
        return self._native.explained_variance_ratio

    def transform(self, data, *, runtime: str | None = None) -> np.ndarray:
        """Projects ``data`` ``(channels, samples)`` onto the components: ``(n_components, samples)``."""
        return self._native.transform(_channels(data), runtime=runtime)


class PPCA:
    """Probabilistic PCA (Tipping & Bishop), closed-form maximum likelihood."""

    def __init__(self, native) -> None:
        self._native = native

    @classmethod
    def fit(cls, data, n_components: int, *, runtime: str | None = None) -> PPCA:
        """Fits ``n_components`` latent dimensions on ``data`` ``(channels, samples)``."""
        return cls(_linalg.Ppca.fit(_channels(data), n_components, runtime=runtime))

    @property
    def weights(self) -> np.ndarray:
        """Loadings ``W`` ``(channels, n_components)``."""
        return self._native.weights

    @property
    def mean(self) -> np.ndarray:
        return self._native.mean

    @property
    def noise_variance(self) -> float:
        """Isotropic noise variance sigma squared."""
        return self._native.noise_variance

    def transform(self, data, *, runtime: str | None = None) -> np.ndarray:
        """Posterior mean ``E[z | x]`` of ``data`` ``(channels, samples)``: ``(n_components, samples)``."""
        return self._native.transform(_channels(data), runtime=runtime)


__all__ = ["PCA", "PPCA"]
