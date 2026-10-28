"""Recordings: open files and read them into NumPy; write NumPy arrays as raw binary or Zarr.

Reads return scaled ``float32`` values in each channel's unit, shape ``(channels, samples)``.
Generic formats only (raw binary with a JSON sidecar, Zarr ``/traces``); neural formats come with
the synapse bindings.
"""

from __future__ import annotations

from os import PathLike

import numpy as np

from dspk._native import io as _io

Recording = _io.Recording
Source = _io.Source
open = _io.open  # noqa: A001 (mirrors the Rust name)
sources = _io.sources
formats = _io.formats
WRITE_CHUNK_SAMPLES: int = _io.WRITE_CHUNK_SAMPLES


#: NumPy array dtypes written as they are; other dtypes and Python lists become float32 first.
STORED_DTYPES = ("int8", "int16", "uint16", "int32", "float32", "float64")


def _as_channels(data) -> np.ndarray:
    """``data`` as a C-contiguous ``(channels, samples)`` array: a NumPy array in its own dtype when
    dspk stores that dtype (little-endian), anything else (lists, other dtypes) as float32."""
    array = np.asarray(data)
    keep = isinstance(data, np.ndarray) and array.dtype.name in STORED_DTYPES
    dtype = array.dtype.newbyteorder("<") if keep else np.dtype(np.float32)
    array = np.ascontiguousarray(array, dtype=dtype)
    if array.ndim != 2:
        raise ValueError(f"data must be 2-D (channels, samples), got shape {array.shape}")
    return array


def write_raw(
    data,
    path: str | PathLike,
    sample_rate: float,
    *,
    format: str | None = None,  # noqa: A002
    order: str = "channel_major",
    gain: float = 1.0,
    unit: str = "",
    chunk_samples: int = WRITE_CHUNK_SAMPLES,
) -> None:
    """Writes ``data`` ``(channels, samples)`` as raw binary at ``path`` with its JSON sidecar.

    Integer ``data`` is the stored integers, ``value = stored * gain`` (an int16 amplifier array and
    its volts per bit); float ``data`` is the values. ``format`` defaults to ``data``'s dtype, so
    integers are written as they are, with no conversion; float values stored in an integer
    ``format`` become ``round(value / gain)``. ``order`` is ``"channel_major"`` or ``"time_major"``;
    ``unit`` is the symbol of the values (``""``: dimensionless).
    """
    _io.write_raw(
        _as_channels(data), path, sample_rate,
        format=format, order=order, gain=gain, unit=unit, chunk_samples=chunk_samples,
    )


def write_zarr(
    data,
    path: str | PathLike,
    sample_rate: float,
    *,
    unit: str = "",
    chunk_samples: int = WRITE_CHUNK_SAMPLES,
) -> None:
    """Writes ``data`` ``(channels, samples)`` as a Zarr store at ``path`` (``/traces``, float32)."""
    _io.write_zarr(_as_channels(data), path, sample_rate, unit=unit, chunk_samples=chunk_samples)


__all__ = [
    "STORED_DTYPES", "Recording", "Source", "WRITE_CHUNK_SAMPLES",
    "formats", "open", "sources", "write_raw", "write_zarr",
]
