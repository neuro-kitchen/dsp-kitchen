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
    """Principal component analysis, fitted on the device.

    Channels are the features and samples the observations. Make one with :meth:`fit`.

    Examples
    --------
    >>> import numpy as np
    >>> import dspk
    >>> x = np.random.default_rng(0).standard_normal((8, 5_000)).astype(np.float32)
    >>> pca = dspk.ops.linalg.PCA.fit(x, 3)
    >>> pca.transform(x).shape
    (3, 5000)
    """

    def __init__(self, native) -> None:
        self._native = native

    @classmethod
    def fit(cls, data, n_components: int, *, runtime: str | None = None) -> PCA:
        """Fits ``n_components`` principal components.

        Parameters
        ----------
        data : array_like
            Signal ``(channels, samples)``, converted to float32 on the host
            first (these models read host data).
        n_components : int
            Components to keep, ``1`` to ``channels``.
        runtime : str, optional
            A runtime from :func:`dspk.runtime.available` (``"cuda"``, ``"hip"``, ``"wgpu"``,
            ``"cpu"``) for this call. By default: the one chosen with :func:`dspk.runtime.set`, else
            ``DSPK_RUNTIME``, else the first available (GPUs first).

        Returns
        -------
        PCA

        Raises
        ------
        ValueError
            For ``n_components`` outside ``1..channels``, or fewer than two samples.
        RuntimeError
            When the runtime has no device.
        """
        return cls(_linalg.Pca.fit(_channels(data), n_components, runtime=runtime))

    @property
    def components(self) -> np.ndarray:
        """The components ``(channels, n_components)``, one per column.
        """
        return self._native.components

    @property
    def mean(self) -> np.ndarray:
        """Mean of each channel ``(channels,)``, removed before projecting.
        """
        return self._native.mean

    @property
    def explained_variance(self) -> np.ndarray:
        """Variance along each component ``(n_components,)``.
        """
        return self._native.explained_variance

    @property
    def explained_variance_ratio(self) -> np.ndarray:
        """Each component's share of the total variance ``(n_components,)``.
        """
        return self._native.explained_variance_ratio

    def transform(self, data, *, runtime: str | None = None) -> np.ndarray:
        """Projects ``data`` onto the components.

        Parameters
        ----------
        data : array_like
            Signal ``(channels, samples)``, converted to float32 on the host
            first (these models read host data).
        runtime : str, optional
            A runtime from :func:`dspk.runtime.available` (``"cuda"``, ``"hip"``, ``"wgpu"``,
            ``"cpu"``) for this call. By default: the one chosen with :func:`dspk.runtime.set`, else
            ``DSPK_RUNTIME``, else the first available (GPUs first).

        Returns
        -------
        numpy.ndarray
            ``(n_components, samples)``.

        Raises
        ------
        ValueError
            When ``data`` has another channel count than the fit.
        """
        return self._native.transform(_channels(data), runtime=runtime)


class PPCA:
    """Probabilistic PCA (Tipping & Bishop): principal components plus an isotropic noise level.

    Closed-form maximum likelihood, fitted on the device. Make one with :meth:`fit`.
    """

    def __init__(self, native) -> None:
        self._native = native

    @classmethod
    def fit(cls, data, n_components: int, *, runtime: str | None = None) -> PPCA:
        """Fits ``n_components`` latent dimensions.

        Parameters
        ----------
        data : array_like
            Signal ``(channels, samples)``, converted to float32 on the host
            first (these models read host data).
        n_components : int
            Latent dimensions, ``1`` to ``channels``.
        runtime : str, optional
            A runtime from :func:`dspk.runtime.available` (``"cuda"``, ``"hip"``, ``"wgpu"``,
            ``"cpu"``) for this call. By default: the one chosen with :func:`dspk.runtime.set`, else
            ``DSPK_RUNTIME``, else the first available (GPUs first).

        Returns
        -------
        PPCA

        Raises
        ------
        ValueError
            For ``n_components`` outside ``1..channels``, or fewer than two samples.
        RuntimeError
            When the runtime has no device.
        """
        return cls(_linalg.Ppca.fit(_channels(data), n_components, runtime=runtime))

    @property
    def weights(self) -> np.ndarray:
        """The loadings ``W`` ``(channels, n_components)``.
        """
        return self._native.weights

    @property
    def mean(self) -> np.ndarray:
        """Mean of each channel ``(channels,)``.
        """
        return self._native.mean

    @property
    def noise_variance(self) -> float:
        """The isotropic noise variance, sigma squared.
        """
        return self._native.noise_variance

    def transform(self, data, *, runtime: str | None = None) -> np.ndarray:
        """The posterior mean of the latent variables, ``E[z | x]``, for ``data``.

        Parameters
        ----------
        data : array_like
            Signal ``(channels, samples)``, converted to float32 on the host
            first (these models read host data).
        runtime : str, optional
            A runtime from :func:`dspk.runtime.available` (``"cuda"``, ``"hip"``, ``"wgpu"``,
            ``"cpu"``) for this call. By default: the one chosen with :func:`dspk.runtime.set`, else
            ``DSPK_RUNTIME``, else the first available (GPUs first).

        Returns
        -------
        numpy.ndarray
            ``(n_components, samples)``.

        Raises
        ------
        ValueError
            When ``data`` has another channel count than the fit.
        """
        return self._native.transform(_channels(data), runtime=runtime)


__all__ = ["PCA", "PPCA"]
