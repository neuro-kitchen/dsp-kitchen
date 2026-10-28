"""Shared by the filter modules: the native submodule and the input check."""

import numpy as np

from dspk._native import ops as _ops

_filter = _ops.filter


def _channels(data) -> np.ndarray:
    """``data`` as a C-contiguous float32 ``(channels, samples)`` array."""
    array = np.ascontiguousarray(data, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"data must be 2-D (channels, samples), got shape {array.shape}")
    return array
