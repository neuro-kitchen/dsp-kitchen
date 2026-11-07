"""Waveform features: shape (morphology), Haar wavelet coefficients, and the conduction velocity of
an HD-EMG motor unit. Principal components are :class:`dspk.ops.linalg.PCA`."""

from __future__ import annotations

import numpy as np

from dspk._native import synapse as _synapse

_f = _synapse.features


def morphology(waveforms) -> dict:
    """Shape features of single-channel waveforms.

    Parameters
    ----------
    waveforms : array_like
        ``(spikes, samples)`` in µV, one channel.

    Returns
    -------
    dict of numpy.ndarray
        Per waveform: ``trough_uv``, ``trough_to_peak_samples``, ``half_width_samples``,
        ``repolarization_slope``.
    """
    return _f.morphology(np.ascontiguousarray(np.atleast_2d(waveforms), dtype=np.float32))


def haar_wavelet(signal, levels: int = 4) -> np.ndarray:
    """Multi-level Haar wavelet transform of a signal.

    Parameters
    ----------
    signal : array_like
        1-D.
    levels : int
        Decomposition levels.

    Returns
    -------
    numpy.ndarray
        The coefficients concatenated, approximation first.
    """
    return _f.haar_wavelet(np.ascontiguousarray(signal, dtype=np.float32), levels)


def conduction_velocity(template, rows: int, cols: int, pitch_um: float, sample_rate: float) -> dict:
    """Muscle-fibre conduction velocity of an HD-EMG motor unit, from its template.

    Parameters
    ----------
    template : array_like
        ``(rows * cols, samples)`` on a row-major grid.
    rows, cols : int
        The grid.
    pitch_um : float
        Electrode spacing, in µm.
    sample_rate : float
        Sample rate in Hz.

    Returns
    -------
    dict
        ``velocity_m_per_s``, ``mean_delay_ms``, ``correlation``.
    """
    return _f.conduction_velocity(np.ascontiguousarray(template, dtype=np.float32), rows, cols, pitch_um, sample_rate)


__all__ = ["conduction_velocity", "haar_wavelet", "morphology"]
