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
    """k-means clustering on the device: k-means++ starts, the best of ``n_init`` runs.

    Parameters
    ----------
    points : array_like
        Feature points ``(n, d)`` (spike features, principal components, positions), converted to
        float32 on the host; 1-D is ``(n, 1)``.
    k : int
        Clusters.
    n_init : int
        Runs from different starts; the one with the lowest inertia is kept.
    max_iter : int
        Iterations per run.
    tol : float
        Convergence tolerance, relative to the data's mean per-feature variance (as scikit-learn).
    seed : int
        Seed of the starts.
    runtime : str, optional
        A runtime from :func:`dspk.runtime.available`; by default the current one.

    Returns
    -------
    dict
        ``labels`` ``(n,)`` int64, ``centers`` ``(k, d)``, ``inertia``.

    Examples
    --------
    >>> import numpy as np
    >>> import dspk
    >>> rng = np.random.default_rng(0)
    >>> pts = np.concatenate([rng.normal(0, 1, (100, 2)), rng.normal(8, 1, (100, 2))])
    >>> out = dspk.synapse.clustering.kmeans(pts, 2)
    >>> len(set(out["labels"][:100])), len(set(out["labels"][100:]))
    (1, 1)
    """
    return _c.kmeans(_points(points), k, n_init=n_init, max_iter=max_iter, tol=tol, seed=seed, runtime=runtime)


def hdbscan(points, *, min_cluster_size: int = DEFAULT_MIN_CLUSTER_SIZE, allow_single_cluster: bool = False, runtime: str | None = None) -> np.ndarray:
    """HDBSCAN clustering on the device, as the ``hdbscan`` package (excess of mass).

    Parameters
    ----------
    points : array_like
        Feature points ``(n, d)`` (spike features, principal components, positions), converted to
        float32 on the host; 1-D is ``(n, 1)``.
    min_cluster_size : int
        Smallest group of points that counts as a cluster.
    allow_single_cluster : bool
        Allow all points to form one cluster.
    runtime : str, optional
        A runtime from :func:`dspk.runtime.available`; by default the current one.

    Returns
    -------
    numpy.ndarray
        ``labels`` ``(n,)`` int64; ``-1`` for noise.
    """
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
    """A Gaussian mixture on the device, its number of components chosen by BIC.

    Parameters
    ----------
    points : array_like
        Feature points ``(n, d)`` (spike features, principal components, positions), converted to
        float32 on the host; 1-D is ``(n, 1)``.
    k_min, k_max : int
        Components tried, ``k_min`` to ``k_max`` included.
    covariance : {"full", "diagonal", "masked"}
        Covariance model; ``"masked"`` is KlustaKwik's masked EM and needs ``mask``.
    max_iterations : int
        EM iterations.
    tolerance : float
        Convergence tolerance of the log-likelihood.
    regularization : float
        Added to each covariance's diagonal.
    outlier_mahalanobis_sq : float, optional
        Points beyond this squared Mahalanobis distance get the label ``-1`` (``25``: beyond 5
        sigma).
    mask : array_like, optional
        ``(n, d)``: a weight for each feature of each point (``"masked"`` only).
    runtime : str, optional
        A runtime from :func:`dspk.runtime.available`; by default the current one.

    Returns
    -------
    dict
        ``labels``, ``weights``, ``means``, ``covariances``, ``responsibilities``,
        ``mahalanobis_sq``, ``log_likelihood``, ``bic``.
    """
    m = None if mask is None else _points(mask)
    return _c.gmm(_points(points), k_min=k_min, k_max=k_max, covariance=covariance, max_iterations=max_iterations, tolerance=tolerance,
                  regularization=regularization, outlier_mahalanobis_sq=outlier_mahalanobis_sq, mask=m, runtime=runtime)


def density_peaks(points, *, cutoff_distance: float, num_clusters: int, max_points: int | None = None) -> dict:
    """Density-peaks clustering (Rodriguez & Laio, 2014), on the host.

    Parameters
    ----------
    points : array_like
        Feature points ``(n, d)`` (spike features, principal components, positions), converted to
        float32 on the host; 1-D is ``(n, 1)``.
    cutoff_distance : float
        Neighbourhood radius of the density.
    num_clusters : int
        Clusters (the points of highest density times distance).
    max_points : int, optional
        Bounds the O(n²) work with an evenly strided subsample.

    Returns
    -------
    dict
        ``labels``, ``centers`` (exemplar indices), ``densities``, ``deltas``.
    """
    return _c.density_peaks(_points(points), cutoff_distance=cutoff_distance, num_clusters=num_clusters, max_points=max_points)


def kde_merge(points, *, initial_k: int, dip_threshold: float = DEFAULT_DIP_THRESHOLD, min_cluster_size: int = DEFAULT_KDE_MIN_CLUSTER_SIZE) -> dict:
    """Over-clusters with k-means, then merges clusters with no density valley between them (host).

    Parameters
    ----------
    points : array_like
        Feature points ``(n, d)`` (spike features, principal components, positions), converted to
        float32 on the host; 1-D is ``(n, 1)``.
    initial_k : int
        Clusters of the first k-means.
    dip_threshold : float
        Pairs whose dip score (``0`` to ``3``: the depth of the density valley between them) is
        below this merge.
    min_cluster_size : int
        Smaller clusters are merged into their nearest.

    Returns
    -------
    dict
        ``labels``, ``centroids``.
    """
    return _c.kde_merge(_points(points), initial_k=initial_k, dip_threshold=dip_threshold, min_cluster_size=min_cluster_size)


def subsample_peaks(samples, channels, *, n_peaks: int, per_channel: bool = False, seed: int = 0) -> np.ndarray:
    """Chooses peaks at random, as sorters do before clustering.

    Parameters
    ----------
    samples : array_like
        Sample of every peak.
    channels : array_like
        Channel of every peak.
    n_peaks : int
        Peaks to choose: in total, or on each channel with ``per_channel``.
    per_channel : bool
        Choose ``n_peaks`` on every channel.
    seed : int
        Seed of the choice.

    Returns
    -------
    numpy.ndarray
        Indices of the chosen peaks, in sample order.
    """
    return _c.subsample_peaks(np.ascontiguousarray(samples, dtype=np.uint64), np.ascontiguousarray(channels, dtype=np.uint32), n_peaks=n_peaks, per_channel=per_channel, seed=seed)


__all__ = [
    "DEFAULT_DIP_THRESHOLD", "DEFAULT_GMM_K_MAX", "DEFAULT_GMM_K_MIN", "DEFAULT_GMM_MAX_ITERATIONS", "DEFAULT_GMM_REGULARIZATION",
    "DEFAULT_GMM_TOLERANCE", "DEFAULT_KDE_MIN_CLUSTER_SIZE", "DEFAULT_MAX_ITER", "DEFAULT_MIN_CLUSTER_SIZE", "DEFAULT_N_INIT", "DEFAULT_TOL",
    "density_peaks", "gmm", "hdbscan", "kde_merge", "kmeans", "subsample_peaks",
]
