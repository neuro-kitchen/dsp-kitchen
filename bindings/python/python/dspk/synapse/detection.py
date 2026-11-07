"""Spike detection: threshold crossings, Teager–Kaiser energy (NEO) and matched filtering on data
in memory, spatial de-duplication across neighbouring channels, and the streaming detector over a
whole recording (:func:`detect_recording`, as ``dspk detect``).

Detections are dicts of arrays sorted by time: ``samples`` (sample index), ``channels``,
``amplitudes`` (µV at the extremum).
"""

from __future__ import annotations

import numpy as np

from dspk._native import synapse as _synapse
from dspk.synapse.io import Probe, Sorting

_d = _synapse.detection

#: In-memory detectors (``threshold``, ``matched_filter``), as dspk-synapse's detectors.
DEFAULT_THRESHOLD: float = _d.DEFAULT_THRESHOLD
#: :func:`detect_recording`, as ``dspk detect``.
DEFAULT_RECORDING_THRESHOLD: float = _d.DEFAULT_RECORDING_THRESHOLD
DEFAULT_NEO_THRESHOLD: float = _d.DEFAULT_NEO_THRESHOLD
DEFAULT_REFRACTORY_MS: float = _d.DEFAULT_REFRACTORY_MS
DEFAULT_RADIUS_UM: float = _d.DEFAULT_RADIUS_UM
DEFAULT_K_NEIGHBORS: int = _d.DEFAULT_K_NEIGHBORS
DEFAULT_PRE_MS: float = _d.DEFAULT_PRE_MS
DEFAULT_POST_MS: float = _d.DEFAULT_POST_MS
DEFAULT_BATCH_S: float = _d.DEFAULT_BATCH_S
DEFAULT_CALIBRATION_S: float = _d.DEFAULT_CALIBRATION_S
DEFAULT_CALIBRATION_CHUNKS: int = _d.DEFAULT_CALIBRATION_CHUNKS
#: Spike band of :func:`detect_recording` (Hz), as ``dspk detect``.
DEFAULT_BAND: tuple[float, float] = (300.0, 6_000.0)


def _f32(data) -> np.ndarray:
    return np.ascontiguousarray(np.atleast_2d(data), dtype=np.float32)


def threshold(data, sample_rate: float, *, threshold: float = DEFAULT_THRESHOLD, polarity: str = "negative", refractory_ms: float = DEFAULT_REFRACTORY_MS,
              sigmas=None, spacing: str = "locally_exclusive") -> dict:  # noqa: A002
    """Local extrema beyond ``threshold · σ`` per channel of ``data`` ``(channels, samples)``
    (``polarity``: negative, positive, both). ``sigmas``: each channel's noise (default: estimated,
    median-based). Spikes on a channel stay more than ``refractory_ms`` apart (``spacing``:
    ``locally_exclusive``, which chunked detection reproduces exactly, or ``scipy``)."""
    s = None if sigmas is None else np.ascontiguousarray(sigmas, dtype=np.float32)
    return _d.threshold(_f32(data), sample_rate, threshold, polarity, refractory_ms, spacing, s)


def neo(data, sample_rate: float, *, threshold: float = DEFAULT_NEO_THRESHOLD, refractory_ms: float = DEFAULT_REFRACTORY_MS, spacing: str = "locally_exclusive") -> dict:  # noqa: A002
    """Teager–Kaiser energy peaks beyond ``threshold · σ(energy)`` per channel."""
    return _d.neo(_f32(data), sample_rate, threshold, refractory_ms, spacing)


def prototype(samples: int) -> np.ndarray:
    """A zero-mean, unit-norm biphasic spike shape of ``samples`` samples."""
    return _d.prototype(samples)


def matched_filter(data, sample_rate: float, *, prototype_ms: float = 1.0, prototype_=None, threshold: float = DEFAULT_THRESHOLD,
                   refractory_ms: float = DEFAULT_REFRACTORY_MS, spacing: str = "locally_exclusive") -> dict:  # noqa: A002
    """Correlation with a spike shape (``prototype_``, default :func:`prototype` of ``prototype_ms``)
    peaking beyond ``threshold · σ(score)`` where the signal is negative."""
    proto = prototype(max(3, round(prototype_ms * sample_rate / 1e3))) if prototype_ is None else np.ascontiguousarray(prototype_, dtype=np.float32)
    return _d.matched_filter(_f32(data), sample_rate, proto, threshold, refractory_ms, spacing)


def heights(sigmas, threshold: float = DEFAULT_THRESHOLD) -> np.ndarray:  # noqa: A002
    """Detection height per channel: ``threshold · σ`` (``inf`` where σ is not positive)."""
    return _d.heights(np.ascontiguousarray(sigmas, dtype=np.float32), threshold)


def deduplicate(spikes: dict, probe: Probe, sample_rate: float, *, radius_um: float = DEFAULT_RADIUS_UM, window_ms: float = 0.5) -> dict:
    """One spike per event seen on several channels: the largest within ``radius_um`` and
    ``window_ms`` stays. Adds ``participating`` (the channels each event was seen on)."""
    return _d.deduplicate(np.ascontiguousarray(spikes["samples"], dtype=np.uint64), np.ascontiguousarray(spikes["channels"], dtype=np.uint32),
                          np.ascontiguousarray(spikes["amplitudes"], dtype=np.float32), probe._native, radius_um, max(1, round(window_ms * sample_rate / 1e3)))


def detect_recording(recording, probe: Probe, *, band: tuple[float, float] = DEFAULT_BAND, car: bool = False, threshold: float = DEFAULT_RECORDING_THRESHOLD,  # noqa: A002
                     refractory_ms: float = DEFAULT_REFRACTORY_MS, polarity: str = "negative", spacing: str = "locally_exclusive",
                     radius_um: float = DEFAULT_RADIUS_UM, k_neighbors: int = DEFAULT_K_NEIGHBORS, pre_ms: float = DEFAULT_PRE_MS,
                     post_ms: float = DEFAULT_POST_MS, sinc_shift: bool = True, locate: bool = True, batch_s: float = DEFAULT_BATCH_S,
                     calibration_s: float = DEFAULT_CALIBRATION_S, calibration_chunks: int = DEFAULT_CALIBRATION_CHUNKS,
                     runtime: str | None = None) -> tuple[Sorting, dict]:
    """Detects spikes over a whole :class:`dspk.io.Recording`, read in place in ``batch_s`` batches
    (optional CAR, zero-phase band-pass, noise calibrated on spread-out chunks, crossings,
    spatial de-duplication, snippets, per-channel templates) as ``dspk detect``; with ``locate``,
    every spike located (centre of mass of its snippet's heights over its ``k_neighbors``: the
    sorting's ``spikes()["locations"]``, else NaN). Returns a :class:`Sorting` (one unit per
    channel) and a summary (``sigmas_uv``, ``raw_crossings``, ``spikes``, ``channel_counts``)."""
    native, summary = _d.detect_recording(
        recording, probe._native, band=tuple(band), car=car, threshold=threshold, refractory_ms=refractory_ms, polarity=polarity,
        spacing=spacing, radius_um=radius_um, k_neighbors=k_neighbors, pre_ms=pre_ms, post_ms=post_ms, sinc_shift=sinc_shift,
        locate=locate, batch_s=batch_s, calibration_s=calibration_s, calibration_chunks=calibration_chunks, runtime=runtime,
    )
    return Sorting(native), summary


__all__ = [
    "DEFAULT_BAND", "DEFAULT_BATCH_S", "DEFAULT_CALIBRATION_CHUNKS", "DEFAULT_CALIBRATION_S", "DEFAULT_K_NEIGHBORS", "DEFAULT_NEO_THRESHOLD",
    "DEFAULT_POST_MS", "DEFAULT_PRE_MS", "DEFAULT_RADIUS_UM", "DEFAULT_RECORDING_THRESHOLD", "DEFAULT_REFRACTORY_MS", "DEFAULT_THRESHOLD",
    "deduplicate", "detect_recording", "heights", "matched_filter", "neo", "prototype", "threshold",
]
