"""Waveform features: shape (morphology), Haar wavelet coefficients, and the conduction velocity of
an HD-EMG motor unit. Principal components are :class:`dspk.ops.linalg.PCA`."""

from __future__ import annotations

import numpy as np

from dspk._native import synapse as _synapse

_f = _synapse.features


def morphology(waveforms) -> dict:
    """Per waveform (rows of ``(spikes, samples)``, one channel): ``trough_uv``,
    ``trough_to_peak_samples``, ``half_width_samples``, ``repolarization_slope``."""
    return _f.morphology(np.ascontiguousarray(np.atleast_2d(waveforms), dtype=np.float32))


def haar_wavelet(signal, levels: int = 4) -> np.ndarray:
    """Multi-level Haar wavelet transform (coefficients concatenated, approximation first)."""
    return _f.haar_wavelet(np.ascontiguousarray(signal, dtype=np.float32), levels)


def conduction_velocity(template, rows: int, cols: int, pitch_um: float, sample_rate: float) -> dict:
    """Muscle-fibre conduction velocity from an HD-EMG template ``(rows · cols, samples)`` (row-major
    grid, ``pitch_um`` apart): ``velocity_m_per_s``, ``mean_delay_ms``, ``correlation``."""
    return _f.conduction_velocity(np.ascontiguousarray(template, dtype=np.float32), rows, cols, pitch_um, sample_rate)


__all__ = ["conduction_velocity", "haar_wavelet", "morphology"]
