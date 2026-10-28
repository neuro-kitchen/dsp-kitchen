"""Operations across channels on the device."""

from __future__ import annotations

import numpy as np

from dspk._native import ops as _ops
from dspk.ops.filter._common import _channels

_spatial = _ops.spatial

DEFAULT_WHITENING_EPSILON: float = _spatial.DEFAULT_WHITENING_EPSILON


def car(data, *, runtime: str | None = None) -> np.ndarray:
    """Common average reference: the mean over channels subtracted at every sample."""
    return _spatial.car(_channels(data), runtime=runtime)


def cmr(data, *, runtime: str | None = None) -> np.ndarray:
    """Common median reference: the median over channels subtracted at every sample."""
    return _spatial.cmr(_channels(data), runtime=runtime)


class Whitening:
    """Spatial whitening ``(channels, channels)``, fitted and applied on the device."""

    def __init__(self, native) -> None:
        self._native = native

    @classmethod
    def fit_zca(cls, data, *, epsilon: float = DEFAULT_WHITENING_EPSILON, runtime: str | None = None) -> Whitening:
        """Global ZCA ``U (L + epsilon)^-1/2 U^T`` of ``data`` ``(channels, samples)``."""
        return cls(_spatial.Whitening.fit_zca(_channels(data), epsilon=epsilon, runtime=runtime))

    @classmethod
    def fit_local_knn(
        cls, data, positions, k: int, *, epsilon: float = DEFAULT_WHITENING_EPSILON, runtime: str | None = None
    ) -> Whitening:
        """Local ZCA: each channel whitened over its ``k`` nearest channels (``positions``
        ``(channels, 2)``), as Kilosort4."""
        pos = np.ascontiguousarray(positions, dtype=np.float32)
        return cls(_spatial.Whitening.fit_local_knn(_channels(data), pos, k, epsilon=epsilon, runtime=runtime))

    @classmethod
    def from_matrix(cls, matrix) -> Whitening:
        """A whitening from an explicit ``(channels, channels)`` matrix."""
        return cls(_spatial.Whitening.from_matrix(np.ascontiguousarray(matrix, dtype=np.float32)))

    @property
    def matrix(self) -> np.ndarray:
        return self._native.matrix

    def apply(self, data, *, runtime: str | None = None) -> np.ndarray:
        """``matrix @ data`` on the device."""
        return self._native.apply(_channels(data), runtime=runtime)


class Laplacian:
    """Surface Laplacian ``(channels, channels)``, applied on the device."""

    def __init__(self, native) -> None:
        self._native = native

    @classmethod
    def from_grid(cls, rows: int, cols: int) -> Laplacian:
        """5-point Laplacian on a ``rows x cols`` grid (channel ``row * cols + col``)."""
        return cls(_spatial.Laplacian.from_grid(rows, cols))

    @classmethod
    def from_positions(cls, positions, k: int) -> Laplacian:
        """Distance-weighted Hjorth Laplacian over each channel's ``k`` nearest."""
        return cls(_spatial.Laplacian.from_positions(np.ascontiguousarray(positions, dtype=np.float32), k))

    @property
    def matrix(self) -> np.ndarray:
        return self._native.matrix

    def apply(self, data, *, runtime: str | None = None) -> np.ndarray:
        """``matrix @ data`` on the device."""
        return self._native.apply(_channels(data), runtime=runtime)


__all__ = ["DEFAULT_WHITENING_EPSILON", "Laplacian", "Whitening", "car", "cmr"]
