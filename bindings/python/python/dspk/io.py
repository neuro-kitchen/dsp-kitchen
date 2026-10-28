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


def _as_channels(data) -> np.ndarray:
    array = np.ascontiguousarray(data, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"data must be 2-D (channels, samples), got shape {array.shape}")
    return array


def write_raw(
    data,
    path: str | PathLike,
    sample_rate: float,
    *,
    format: str = "float32",  # noqa: A002
    order: str = "channel_major",
    gain: float = 1.0,
    unit: str = "",
    chunk_samples: int = WRITE_CHUNK_SAMPLES,
) -> None:
    """Writes ``data`` ``(channels, samples)`` as raw binary at ``path`` with its JSON sidecar.

    Integer ``format``\\ s store ``round(value / gain)``; ``order`` is ``"channel_major"`` or
    ``"time_major"``; ``unit`` is the symbol of the values (``""``: dimensionless).
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
    "Recording", "Source", "WRITE_CHUNK_SAMPLES",
    "formats", "open", "sources", "write_raw", "write_zarr",
]
