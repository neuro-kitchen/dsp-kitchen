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
ZarrWriter = _io.ZarrWriter
Source = _io.Source
open = _io.open  # noqa: A001 (mirrors the Rust name)
sources = _io.sources
formats = _io.formats
WRITE_CHUNK_SAMPLES: int = _io.WRITE_CHUNK_SAMPLES
ZARR_CHUNK_SAMPLES: int = _io.ZARR_CHUNK_SAMPLES


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
    """Writes a recording as raw binary with a JSON sidecar.

    Integers are written as they are: an int16 array from an amplifier is stored as int16, its
    ``gain`` (units per bit) in the sidecar. Float values stored in an integer ``format`` become
    ``round(value / gain)``.

    Parameters
    ----------
    data : array_like
        Samples ``(channels, samples)``: integer arrays are the stored integers
        (``value = stored * gain``), float arrays the values. NumPy arrays of a
        :data:`STORED_DTYPES` dtype are written as they are; anything else becomes float32.
    path : str or os.PathLike
        The binary file to write; the sidecar is written next to it.
    sample_rate : float
        Sample rate in Hz.
    format : str, optional
        Stored type (``"int16"``, ``"float32"``, ...); by default ``data``'s dtype.
    order : {"channel_major", "time_major"}
        Sample layout in the file: each channel's samples together, or each time step's channels
        together.
    gain : float
        Units per stored integer.
    unit : str
        Unit symbol of the values: ``"V"``, ``"mV"``, ``"µV"`` (or ``"uV"``), ``"A"``, ...;
        ``""`` for dimensionless.
    chunk_samples : int
        Samples per channel written at a time (bounds memory, not the result).

    Raises
    ------
    ValueError
        When ``data`` is not 2-D or an option is invalid.
    OSError
        When the file cannot be written.

    Examples
    --------
    >>> import numpy as np
    >>> import dspk
    >>> counts = np.zeros((4, 30_000), dtype=np.int16)
    >>> dspk.io.write_raw(counts, "session.bin", 30_000.0, gain=0.195, unit="µV")  # doctest: +SKIP
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
    chunk_samples: int = ZARR_CHUNK_SAMPLES,
    compression: str = "zstd",
    level: int | None = None,
    parallel: bool = True,
) -> None:
    """Writes a recording as a Zarr store (``/traces``, float32).

    Chunks of ``chunk_samples`` (all channels) are compressed and stored on several threads.

    Parameters
    ----------
    data : array_like
        Values ``(channels, samples)``.
    path : str or os.PathLike
        The store (a folder) to create.
    sample_rate : float
        Sample rate in Hz.
    unit : str
        Unit symbol of the values (``""`` for dimensionless).
    chunk_samples : int
        Samples per Zarr chunk.
    compression : {"zstd", "blosc_lz4", "none"}
        How chunks are compressed.
    level : int, optional
        Compression level: zstd's (default 3) or blosc's 0–9 (default 5); ``"none"`` takes none.
    parallel : bool
        Compress and store on several threads; ``False`` uses one.

    Raises
    ------
    ValueError
        When ``data`` is not 2-D or an option is invalid.
    OSError
        When the store cannot be written.

    Examples
    --------
    >>> import numpy as np
    >>> import dspk
    >>> x = np.random.default_rng(0).standard_normal((4, 30_000)).astype(np.float32)
    >>> dspk.io.write_zarr(x, "session.zarr", 30_000.0, unit="µV")  # doctest: +SKIP
    """
    _io.write_zarr(
        _as_channels(data), path, sample_rate,
        unit=unit, chunk_samples=chunk_samples, compression=compression, level=level, parallel=parallel,
    )


__all__ = [
    "STORED_DTYPES", "Recording", "Source", "WRITE_CHUNK_SAMPLES", "ZARR_CHUNK_SAMPLES", "ZarrWriter",
    "formats", "open", "sources", "write_raw", "write_zarr",
]
