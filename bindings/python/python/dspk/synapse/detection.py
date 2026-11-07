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
    """Threshold crossings: local extrema beyond ``threshold * sigma`` on each channel.

    Parameters
    ----------
    data : array_like
        Signal ``(channels, samples)`` in µV, converted to float32 on the host.
    sample_rate : float
        Sample rate in Hz.
    threshold : float
        Height in units of each channel's noise.
    polarity : {"negative", "positive", "both"}
        Which extrema count.
    refractory_ms : float
        Spikes on one channel stay more than this apart.
    sigmas : array_like, optional
        Each channel's noise ``(channels,)``; by default estimated as ``median(|x|) / 0.6745``
        (:func:`dspk.ops.math.channel_noise_std`).
    spacing : {"locally_exclusive", "scipy"}
        Which of two nearer spikes stays. ``"locally_exclusive"``: a spike stays unless a larger one
        is nearer, which chunked detection reproduces exactly. ``"scipy"``: as
        ``scipy.signal.find_peaks`` with ``distance``.

    Returns
    -------
    dict of numpy.ndarray
        ``samples`` (sample index), ``channels`` and ``amplitudes`` (µV at the extremum), sorted by
        time.

    Raises
    ------
    ValueError
        For an unknown ``polarity`` or ``spacing``, or ``sigmas`` of another length.

    Examples
    --------
    >>> import numpy as np
    >>> import dspk
    >>> x = np.random.default_rng(0).standard_normal((4, 30_000)).astype(np.float32) * 10
    >>> x[1, 15_000:15_010] -= 120  # one spike
    >>> spikes = dspk.synapse.detection.threshold(x, 30_000.0)
    >>> int(spikes["channels"][0]), int(spikes["samples"][0])  # the spike's most negative sample
    (1, 15004)
    """
    s = None if sigmas is None else np.ascontiguousarray(sigmas, dtype=np.float32)
    return _d.threshold(_f32(data), sample_rate, threshold, polarity, refractory_ms, spacing, s)


def neo(data, sample_rate: float, *, threshold: float = DEFAULT_NEO_THRESHOLD, refractory_ms: float = DEFAULT_REFRACTORY_MS, spacing: str = "locally_exclusive") -> dict:  # noqa: A002
    """Peaks of the Teager–Kaiser energy (NEO) beyond ``threshold * sigma(energy)`` on each channel.

    The energy is large where the signal is both large and fast, so it favours spikes over slow
    waves.

    Parameters
    ----------
    data : array_like
        Signal ``(channels, samples)`` in µV, converted to float32 on the host.
    sample_rate : float
        Sample rate in Hz.
    threshold : float
        Height in units of each channel's energy noise.
    refractory_ms : float
        Spikes on one channel stay more than this apart.
    spacing : {"locally_exclusive", "scipy"}
        Which of two nearer spikes stays. ``"locally_exclusive"``: a spike stays unless a larger one
        is nearer, which chunked detection reproduces exactly. ``"scipy"``: as
        ``scipy.signal.find_peaks`` with ``distance``.

    Returns
    -------
    dict of numpy.ndarray
        ``samples`` (sample index), ``channels`` and ``amplitudes`` (µV at the extremum), sorted by
        time.
    """
    return _d.neo(_f32(data), sample_rate, threshold, refractory_ms, spacing)


def prototype(samples: int) -> np.ndarray:
    """A biphasic spike shape: zero mean, unit norm.

    Parameters
    ----------
    samples : int
        Length.

    Returns
    -------
    numpy.ndarray
        ``(samples,)``, float32.
    """
    return _d.prototype(samples)


def matched_filter(data, sample_rate: float, *, prototype_ms: float = 1.0, prototype_=None, threshold: float = DEFAULT_THRESHOLD,
                   refractory_ms: float = DEFAULT_REFRACTORY_MS, spacing: str = "locally_exclusive") -> dict:  # noqa: A002
    """Matched-filter detection: correlation with a spike shape, peaks beyond ``threshold * sigma``.

    Only peaks where the signal is negative count.

    Parameters
    ----------
    data : array_like
        Signal ``(channels, samples)`` in µV, converted to float32 on the host.
    sample_rate : float
        Sample rate in Hz.
    prototype_ms : float
        Length of the default shape (:func:`prototype`), in ms.
    prototype_ : array_like, optional
        Your own shape, instead of the default.
    threshold : float
        Height in units of each channel's correlation noise.
    refractory_ms : float
        Spikes on one channel stay more than this apart.
    spacing : {"locally_exclusive", "scipy"}
        Which of two nearer spikes stays. ``"locally_exclusive"``: a spike stays unless a larger one
        is nearer, which chunked detection reproduces exactly. ``"scipy"``: as
        ``scipy.signal.find_peaks`` with ``distance``.

    Returns
    -------
    dict of numpy.ndarray
        ``samples`` (sample index), ``channels`` and ``amplitudes`` (µV at the extremum), sorted by
        time.
    """
    proto = prototype(max(3, round(prototype_ms * sample_rate / 1e3))) if prototype_ is None else np.ascontiguousarray(prototype_, dtype=np.float32)
    return _d.matched_filter(_f32(data), sample_rate, proto, threshold, refractory_ms, spacing)


def heights(sigmas, threshold: float = DEFAULT_THRESHOLD) -> np.ndarray:  # noqa: A002
    """Detection height of every channel: ``threshold * sigma``.

    Parameters
    ----------
    sigmas : array_like
        Each channel's noise ``(channels,)``.
    threshold : float
        Height in units of noise.

    Returns
    -------
    numpy.ndarray
        ``(channels,)``; ``inf`` where ``sigma`` is not positive (the channel never detects).
    """
    return _d.heights(np.ascontiguousarray(sigmas, dtype=np.float32), threshold)


def deduplicate(spikes: dict, probe: Probe, sample_rate: float, *, radius_um: float = DEFAULT_RADIUS_UM, window_ms: float = 0.5) -> dict:
    """One spike per event seen on several neighbouring channels: the largest stays.

    Parameters
    ----------
    spikes : dict of numpy.ndarray
        ``samples``, ``channels`` and ``amplitudes``, as the detectors return.
    probe : Probe
        The probe, for distances between channels.
    sample_rate : float
        Sample rate in Hz.
    radius_um : float
        Spikes on channels closer than this...
    window_ms : float
        ...and closer in time than this are one event.

    Returns
    -------
    dict of numpy.ndarray
        The spikes kept, plus ``participating``: the channels each event was seen on.
    """
    return _d.deduplicate(np.ascontiguousarray(spikes["samples"], dtype=np.uint64), np.ascontiguousarray(spikes["channels"], dtype=np.uint32),
                          np.ascontiguousarray(spikes["amplitudes"], dtype=np.float32), probe._native, radius_um, max(1, round(window_ms * sample_rate / 1e3)))


def detect_recording(recording, probe: Probe, *, band: tuple[float, float] = DEFAULT_BAND, car: bool = False, threshold: float = DEFAULT_RECORDING_THRESHOLD,  # noqa: A002
                     refractory_ms: float = DEFAULT_REFRACTORY_MS, polarity: str = "negative", spacing: str = "locally_exclusive",
                     radius_um: float = DEFAULT_RADIUS_UM, k_neighbors: int = DEFAULT_K_NEIGHBORS, pre_ms: float = DEFAULT_PRE_MS,
                     post_ms: float = DEFAULT_POST_MS, sinc_shift: bool = True, locate: bool = True, batch_s: float = DEFAULT_BATCH_S,
                     calibration_s: float = DEFAULT_CALIBRATION_S, calibration_chunks: int = DEFAULT_CALIBRATION_CHUNKS,
                     runtime: str | None = None) -> tuple[Sorting, dict]:
    """Detects spikes over a whole recording, as ``dspk detect`` does.

    The recording is read in place in ``batch_s`` batches: optional CAR, zero-phase band-pass,
    noise calibrated on chunks spread over the recording, threshold crossings, spatial
    de-duplication, snippets and per-channel templates.

    Parameters
    ----------
    recording : dspk.io.Recording
        The recording.
    probe : Probe
        Its probe.
    band : (float, float)
        Band-pass ``(low, high)`` in Hz.
    car : bool
        Subtract the common average first.
    threshold : float
        Height in units of each channel's noise.
    refractory_ms : float
        Spikes on one channel stay more than this apart.
    polarity : {"negative", "positive", "both"}
        Which extrema count.
    spacing : {"locally_exclusive", "scipy"}
        Which of two nearer spikes stays (see :func:`threshold`).
    radius_um : float
        De-duplication radius.
    k_neighbors : int
        Channels per snippet (the spike's nearest).
    pre_ms, post_ms : float
        Snippet span around each spike.
    sinc_shift : bool
        Realign each snippet to its sub-sample trough.
    locate : bool
        Locate every spike: the centre of mass of its snippet's heights over its
        ``k_neighbors``.
    batch_s : float
        Seconds read and processed at a time.
    calibration_s : float
        Seconds of signal the noise is measured on.
    calibration_chunks : int
        Pieces those seconds are spread over.
    runtime : str, optional
        A runtime from :func:`dspk.runtime.available`; by default the current one.

    Returns
    -------
    sorting : Sorting
        One unit per channel; ``spikes()["locations"]`` holds the locations (NaN without
        ``locate``).
    summary : dict
        ``sigmas_uv`` (noise per channel), ``raw_crossings``, ``spikes``, ``channel_counts``.

    Raises
    ------
    ValueError
        For an invalid option.
    OSError
        When the recording cannot be read.
    RuntimeError
        When the runtime has no device.

    Examples
    --------
    >>> import dspk
    >>> rec = dspk.io.open("session.bin")  # doctest: +SKIP
    >>> probe = dspk.synapse.io.Probe.preset("neuropixels1")  # doctest: +SKIP
    >>> sorting, summary = dspk.synapse.detection.detect_recording(rec, probe)  # doctest: +SKIP
    """
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
