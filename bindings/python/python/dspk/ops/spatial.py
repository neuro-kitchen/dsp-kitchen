"""Operations across channels on the device."""

from __future__ import annotations

import numpy as np

from dspk._native import ops as _ops
from dspk.ops.filter._common import _channels

_spatial = _ops.spatial

DEFAULT_WHITENING_EPSILON: float = _spatial.DEFAULT_WHITENING_EPSILON


def car(data, *, runtime: str | None = None) -> np.ndarray:
    """Common average reference: subtracts the mean over channels at every sample.

    Parameters
    ----------
    data : array_like
        Signal ``(channels, samples)``. float32, int8, int16, uint16 and int32 arrays go to the
        device as they are and are converted there; other dtypes and lists become float32 first.
    runtime : str, optional
        A runtime from :func:`dspk.runtime.available` (``"cuda"``, ``"hip"``, ``"wgpu"``,
        ``"cpu"``) for this call. By default: the one chosen with :func:`dspk.runtime.set`, else
        ``DSPK_RUNTIME``, else the first available (GPUs first).

    Returns
    -------
    numpy.ndarray
        ``(channels, samples)``, float32.

    Raises
    ------
    ValueError
        When ``data`` is not 2-D.
    RuntimeError
        When the runtime has no device.

    Examples
    --------
    >>> import numpy as np
    >>> import dspk
    >>> x = np.random.default_rng(0).standard_normal((4, 30_000)).astype(np.float32)
    >>> referenced = dspk.ops.spatial.car(x)
    """
    return _spatial.car(_channels(data), runtime=runtime)


def cmr(data, *, runtime: str | None = None) -> np.ndarray:
    """Common median reference: subtracts the median over channels at every sample.

    The median ignores a single bad channel where the mean would spread it to all. With an even
    number of channels it is the mean of the two middle values, as NumPy's.

    Parameters
    ----------
    data : array_like
        Signal ``(channels, samples)``. float32, int8, int16, uint16 and int32 arrays go to the
        device as they are and are converted there; other dtypes and lists become float32 first.
    runtime : str, optional
        A runtime from :func:`dspk.runtime.available` (``"cuda"``, ``"hip"``, ``"wgpu"``,
        ``"cpu"``) for this call. By default: the one chosen with :func:`dspk.runtime.set`, else
        ``DSPK_RUNTIME``, else the first available (GPUs first).

    Returns
    -------
    numpy.ndarray
        ``(channels, samples)``, float32.

    Raises
    ------
    ValueError
        When ``data`` is not 2-D.
    RuntimeError
        When the runtime has no device.

    Examples
    --------
    >>> import numpy as np
    >>> import dspk
    >>> x = np.random.default_rng(0).standard_normal((4, 30_000)).astype(np.float32)
    >>> referenced = dspk.ops.spatial.cmr(x)
    """
    return _spatial.cmr(_channels(data), runtime=runtime)


class Whitening:
    """Spatial whitening: a ``(channels, channels)`` matrix that decorrelates the channels.

    Make one with :meth:`fit_zca`, :meth:`fit_local_knn` or :meth:`from_matrix`, then
    :meth:`apply` it.

    Examples
    --------
    >>> import numpy as np
    >>> import dspk
    >>> x = np.random.default_rng(0).standard_normal((4, 30_000)).astype(np.float32)
    >>> w = dspk.ops.spatial.Whitening.fit_zca(x)
    >>> white = w.apply(x)
    """

    def __init__(self, native) -> None:
        self._native = native

    @classmethod
    def fit_zca(cls, data, *, epsilon: float = DEFAULT_WHITENING_EPSILON, runtime: str | None = None) -> Whitening:
        """Fits a global ZCA whitening, ``U (L + epsilon)^-1/2 U^T`` of the channel covariance.

        Parameters
        ----------
        data : array_like
            Signal ``(channels, samples)``, converted to float32 on the host
            first (these models read host data).
        epsilon : float
            Added to every eigenvalue: keeps near-silent directions from being amplified.
        runtime : str, optional
            A runtime from :func:`dspk.runtime.available` (``"cuda"``, ``"hip"``, ``"wgpu"``,
            ``"cpu"``) for this call. By default: the one chosen with :func:`dspk.runtime.set`, else
            ``DSPK_RUNTIME``, else the first available (GPUs first).

        Returns
        -------
        Whitening

        Raises
        ------
        ValueError
            When ``data`` is not 2-D.
        RuntimeError
            When the runtime has no device.
        """
        return cls(_spatial.Whitening.fit_zca(_channels(data), epsilon=epsilon, runtime=runtime))

    @classmethod
    def fit_local_knn(
        cls, data, positions, k: int, *, epsilon: float = DEFAULT_WHITENING_EPSILON, runtime: str | None = None
    ) -> Whitening:
        """Fits a local ZCA whitening: each channel whitened over its ``k`` nearest channels.

        As Kilosort4: for each channel, the ``k`` closest channels (itself included) give a local
        covariance whose ZCA row becomes that channel's row of the matrix.

        Parameters
        ----------
        data : array_like
            Signal ``(channels, samples)``, converted to float32 on the host
            first (these models read host data).
        positions : array_like
            Electrode positions ``(channels, 2)``, ``(x, y)``.
        k : int
            Channels per neighbourhood, ``1`` to ``channels``.
        epsilon : float
            Added to every eigenvalue of each local covariance.
        runtime : str, optional
            A runtime from :func:`dspk.runtime.available` (``"cuda"``, ``"hip"``, ``"wgpu"``,
            ``"cpu"``) for this call. By default: the one chosen with :func:`dspk.runtime.set`, else
            ``DSPK_RUNTIME``, else the first available (GPUs first).

        Returns
        -------
        Whitening

        Raises
        ------
        ValueError
            For ``k`` outside ``1..channels``, ``positions`` not ``(channels, 2)``, or ``data`` not
            2-D.
        RuntimeError
            When the runtime has no device.
        """
        pos = np.ascontiguousarray(positions, dtype=np.float32)
        return cls(_spatial.Whitening.fit_local_knn(_channels(data), pos, k, epsilon=epsilon, runtime=runtime))

    @classmethod
    def from_matrix(cls, matrix) -> Whitening:
        """A whitening from a matrix you already have.

        Parameters
        ----------
        matrix : array_like
            ``(channels, channels)``.

        Returns
        -------
        Whitening

        Raises
        ------
        ValueError
            When ``matrix`` is not square.
        """
        return cls(_spatial.Whitening.from_matrix(np.ascontiguousarray(matrix, dtype=np.float32)))

    @property
    def matrix(self) -> np.ndarray:
        """The whitening matrix ``(channels, channels)``.
        """
        return self._native.matrix

    def apply(self, data, *, runtime: str | None = None) -> np.ndarray:
        """Applies the whitening, ``matrix @ data``, on the device.

        Parameters
        ----------
        data : array_like
            Signal ``(channels, samples)``. float32, int8, int16, uint16 and int32 arrays go to the
            device as they are and are converted there; other dtypes and lists become float32 first.
        runtime : str, optional
            A runtime from :func:`dspk.runtime.available` (``"cuda"``, ``"hip"``, ``"wgpu"``,
            ``"cpu"``) for this call. By default: the one chosen with :func:`dspk.runtime.set`, else
            ``DSPK_RUNTIME``, else the first available (GPUs first).

        Returns
        -------
        numpy.ndarray
            ``(channels, samples)``, float32.
        """
        return self._native.apply(_channels(data), runtime=runtime)


class Laplacian:
    """Surface Laplacian: a ``(channels, channels)`` matrix that keeps what is local to each
    electrode.

    Each channel minus the mean of its neighbours, which removes what spreads over the whole
    array (volume-conducted cross-talk). Make one with :meth:`from_grid` or
    :meth:`from_positions`, then :meth:`apply` it.

    Examples
    --------
    >>> import numpy as np
    >>> import dspk
    >>> x = np.random.default_rng(0).standard_normal((64, 30_000)).astype(np.float32)
    >>> lap = dspk.ops.spatial.Laplacian.from_grid(8, 8)
    >>> local = lap.apply(x)
    """

    def __init__(self, native) -> None:
        self._native = native

    @classmethod
    def from_grid(cls, rows: int, cols: int) -> Laplacian:
        """The 5-point Laplacian of a ``rows x cols`` electrode grid.

        Channel ``row * cols + col`` sits at ``(row, col)``. Each channel's neighbours are the ones
        above, below, left and right: 4 inside the grid, 3 on an edge, 2 in a corner.

        Parameters
        ----------
        rows : int
            Grid rows.
        cols : int
            Grid columns.

        Returns
        -------
        Laplacian

        Raises
        ------
        ValueError
            When ``rows`` or ``cols`` is 0.
        """
        return cls(_spatial.Laplacian.from_grid(rows, cols))

    @classmethod
    def from_positions(cls, positions, k: int) -> Laplacian:
        """A distance-weighted (Hjorth) Laplacian from electrode positions.

        Each channel minus a distance-weighted mean of its ``k`` nearest channels.

        Parameters
        ----------
        positions : array_like
            Electrode positions ``(channels, 2)``, ``(x, y)``.
        k : int
            Neighbours per channel.

        Returns
        -------
        Laplacian
        """
        return cls(_spatial.Laplacian.from_positions(np.ascontiguousarray(positions, dtype=np.float32), k))

    @property
    def matrix(self) -> np.ndarray:
        """The Laplacian matrix ``(channels, channels)``.
        """
        return self._native.matrix

    def apply(self, data, *, runtime: str | None = None) -> np.ndarray:
        """Applies the Laplacian, ``matrix @ data``, on the device.

        Parameters
        ----------
        data : array_like
            Signal ``(channels, samples)``. float32, int8, int16, uint16 and int32 arrays go to the
            device as they are and are converted there; other dtypes and lists become float32 first.
        runtime : str, optional
            A runtime from :func:`dspk.runtime.available` (``"cuda"``, ``"hip"``, ``"wgpu"``,
            ``"cpu"``) for this call. By default: the one chosen with :func:`dspk.runtime.set`, else
            ``DSPK_RUNTIME``, else the first available (GPUs first).

        Returns
        -------
        numpy.ndarray
            ``(channels, samples)``, float32.
        """
        return self._native.apply(_channels(data), runtime=runtime)


__all__ = ["DEFAULT_WHITENING_EPSILON", "Laplacian", "Whitening", "car", "cmr"]
