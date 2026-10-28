"""Shared by the filter modules: the native submodule and the input check."""

import numpy as np

from dspk._native import ops as _ops

_filter = _ops.filter


#: NumPy dtypes that cross to the device as they are (integers are converted there); other dtypes
#: and Python lists become float32 on the host first.
DEVICE_DTYPES = ("float32", "int8", "int16", "uint16", "int32")


def _channels(data) -> np.ndarray:
    """``data`` as a C-contiguous ``(channels, samples)`` array: a NumPy array of a
    :data:`DEVICE_DTYPES` dtype stays in it (little-endian), anything else becomes float32."""
    array = np.asarray(data)
    keep = isinstance(data, np.ndarray) and array.dtype.name in DEVICE_DTYPES
    array = np.ascontiguousarray(array, dtype=array.dtype.newbyteorder("<") if keep else np.float32)
    if array.ndim != 2:
        raise ValueError(f"data must be 2-D (channels, samples), got shape {array.shape}")
    return array
