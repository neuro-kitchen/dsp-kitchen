"""Clustering of feature points ``(n, d)`` (spike features, PCs, positions): k-means, HDBSCAN and
Gaussian mixtures on the runtime's device; density peaks and KDE valley merging on the host; and
the random choice of peaks sorters cluster on. Labels are ``int64`` (``-1``: noise / outlier).
"""

from __future__ import annotations

import numpy as np

from dspk._native import synapse as _synapse

_c = _synapse.clustering

DEFAULT_N_INIT: int = _c.DEFAULT_N_INIT
DEFAULT_MAX_ITER: int = _c.DEFAULT_MAX_ITER
DEFAULT_TOL: float = _c.DEFAULT_TOL
DEFAULT_MIN_CLUSTER_SIZE: int = _c.DEFAULT_MIN_CLUSTER_SIZE
DEFAULT_GMM_K_MIN: int = _c.DEFAULT_GMM_K_MIN
DEFAULT_GMM_K_MAX: int = _c.DEFAULT_GMM_K_MAX
DEFAULT_GMM_MAX_ITERATIONS: int = _c.DEFAULT_GMM_MAX_ITERATIONS
DEFAULT_GMM_TOLERANCE: float = _c.DEFAULT_GMM_TOLERANCE
DEFAULT_GMM_REGULARIZATION: float = _c.DEFAULT_GMM_REGULARIZATION
DEFAULT_DIP_THRESHOLD: float = _c.DEFAULT_DIP_THRESHOLD
DEFAULT_KDE_MIN_CLUSTER_SIZE: int = _c.DEFAULT_KDE_MIN_CLUSTER_SIZE


def _points(x) -> np.ndarray:
    x = np.ascontiguousarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2:
        raise ValueError(f"points must be (n, d), got shape {x.shape}")
    return x


def kmeans(points, k: int, *, n_init: int = DEFAULT_N_INIT, max_iter: int = DEFAULT_MAX_ITER, tol: float = DEFAULT_TOL, seed: int = 0, runtime: str | None = None) -> dict:
    """k-means (k-means++ starts; the best of ``n_init``): ``labels``, ``centers`` ``(k, d)``,
    ``inertia``. ``tol`` is relative to the data's mean per-feature variance (as scikit-learn)."""
    return _c.kmeans(_points(points), k, n_init=n_init, max_iter=max_iter, tol=tol, seed=seed, runtime=runtime)


def hdbscan(points, *, min_cluster_size: int = DEFAULT_MIN_CLUSTER_SIZE, allow_single_cluster: bool = False, runtime: str | None = None) -> np.ndarray:
    """HDBSCAN labels (as the ``hdbscan`` package: excess of mass, ``-1`` for noise)."""
    return _c.hdbscan(_points(points), min_cluster_size=min_cluster_size, allow_single_cluster=allow_single_cluster, runtime=runtime)


def gmm(
    points,
    *,
    k_min: int = DEFAULT_GMM_K_MIN,
    k_max: int = DEFAULT_GMM_K_MAX,
    covariance: str = "full",
    max_iterations: int = DEFAULT_GMM_MAX_ITERATIONS,
    tolerance: float = DEFAULT_GMM_TOLERANCE,
    regularization: float = DEFAULT_GMM_REGULARIZATION,
    outlier_mahalanobis_sq: float | None = None,
    mask=None,
    runtime: str | None = None,
) -> dict:
    """A Gaussian mixture with its number of components chosen by BIC in ``k_min..=k_max``.
    ``covariance``: ``full``, ``diagonal`` or ``masked`` (KlustaKwik's masked EM, with ``mask``
    ``(n, d)``). Returns ``labels``, ``weights``, ``means``, ``covariances``,
    ``responsibilities``, ``mahalanobis_sq``, ``log_likelihood``, ``bic``."""
    m = None if mask is None else _points(mask)
    return _c.gmm(_points(points), k_min=k_min, k_max=k_max, covariance=covariance, max_iterations=max_iterations, tolerance=tolerance,
                  regularization=regularization, outlier_mahalanobis_sq=outlier_mahalanobis_sq, mask=m, runtime=runtime)


def density_peaks(points, *, cutoff_distance: float, num_clusters: int, max_points: int | None = None) -> dict:
    """Rodriguez–Laio density peaks: ``labels``, ``centers`` (exemplar indices), ``densities``,
    ``deltas``. ``max_points`` bounds the O(n²) work with an evenly strided subsample."""
    return _c.density_peaks(_points(points), cutoff_distance=cutoff_distance, num_clusters=num_clusters, max_points=max_points)


def kde_merge(points, *, initial_k: int, dip_threshold: float = DEFAULT_DIP_THRESHOLD, min_cluster_size: int = DEFAULT_KDE_MIN_CLUSTER_SIZE) -> dict:
    """k-means over-clustering (``initial_k``), then merges of clusters with no density valley
    between them: ``labels``, ``centroids``."""
    return _c.kde_merge(_points(points), initial_k=initial_k, dip_threshold=dip_threshold, min_cluster_size=min_cluster_size)


def subsample_peaks(samples, channels, *, n_peaks: int, per_channel: bool = False, seed: int = 0) -> np.ndarray:
    """Indices of ``n_peaks`` peaks chosen at random (in total, or of each channel), by sample."""
    return _c.subsample_peaks(np.ascontiguousarray(samples, dtype=np.uint64), np.ascontiguousarray(channels, dtype=np.uint32), n_peaks=n_peaks, per_channel=per_channel, seed=seed)


__all__ = [
    "DEFAULT_DIP_THRESHOLD", "DEFAULT_GMM_K_MAX", "DEFAULT_GMM_K_MIN", "DEFAULT_GMM_MAX_ITERATIONS", "DEFAULT_GMM_REGULARIZATION",
    "DEFAULT_GMM_TOLERANCE", "DEFAULT_KDE_MIN_CLUSTER_SIZE", "DEFAULT_MAX_ITER", "DEFAULT_MIN_CLUSTER_SIZE", "DEFAULT_N_INIT", "DEFAULT_TOL",
    "density_peaks", "gmm", "hdbscan", "kde_merge", "kmeans", "subsample_peaks",
]
