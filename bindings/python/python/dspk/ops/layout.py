"""Channels in a new order on the device, and channel maps.

A channel map says which input channel becomes which output channel, as pairs (like a hardware
channel mapper). ``base`` is 0, or 1 for 1-based numbers (TDT, MATLAB):

TOML::

    [remap]
    base = 0
    pairs = [
      { from = 4, to = 0, name = "A1" },
      { from = 2, to = 1, name = "A2" },
    ]

JSON::

    {
      "remap": {
        "base": 0,
        "pairs": [
          {"from": 4, "to": 0, "name": "A1"},
          {"from": 2, "to": 1, "name": "A2"}
        ]
      }
    }

Every output channel ``0..n`` gets exactly one input; inputs left out are dropped; the same input
twice is refused; names are optional. A map is a dict ``{"base": 0, "pairs": [...]}`` (or the
file's ``{"remap": {...}}``), a ``.toml`` / ``.json`` path, or a list of pairs. In a pipeline spec
the same map is a ``{"op": "remap", "pairs": [...]}`` (or ``"path": "map.toml"``) node.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np

from dspk._native import ops as _ops
from dspk.ops.filter._common import _channels

_layout = _ops.layout

DEFAULT_BASE: int = 0

ChannelMap = dict[str, Any]
MapLike = ChannelMap | Sequence[dict[str, Any]] | str | PathLike


def read_map(path: str | PathLike) -> ChannelMap:
    """Reads and checks a channel map file.

    Parameters
    ----------
    path : str or os.PathLike
        A ``.toml`` or ``.json`` file with the pairs under ``remap``.

    Returns
    -------
    dict
        The map, checked: ``{"base": ..., "pairs": [...]}``.

    Raises
    ------
    ValueError
        For another suffix or an invalid map.
    OSError
        When the file cannot be read.
    """
    path = Path(path)
    if path.suffix.lower() not in (".toml", ".json"):
        raise ValueError(f"{path}: a channel map is a .toml or .json file")
    return json.loads(_layout.parse_map(path.read_text(), toml=path.suffix.lower() == ".toml"))


def write_map(map: MapLike, path: str | PathLike) -> Path:  # noqa: A002
    """Writes a channel map file, TOML or JSON by the suffix of ``path``.

    Parameters
    ----------
    map : dict, list of dict, str or os.PathLike
        A channel map: a ``{"base", "pairs"}`` dict (or a file's ``{"remap": {...}}``), a list of
        ``{"from", "to", "name"}`` pairs, or the path of a ``.toml`` / ``.json`` map file.
    path : str or os.PathLike
        The ``.toml`` or ``.json`` file to write.

    Returns
    -------
    pathlib.Path
        The path written.

    Raises
    ------
    ValueError
        For another suffix or an invalid map.

    Examples
    --------
    >>> import dspk
    >>> pairs = [{"from": 1, "to": 0, "name": "A1"}, {"from": 0, "to": 1, "name": "A2"}]
    >>> dspk.ops.layout.write_map(pairs, "map.toml")  # doctest: +SKIP
    """
    path = Path(path)
    canonical = as_map(map)
    if path.suffix.lower() == ".toml":
        path.write_text(_layout.map_toml(json.dumps(canonical)))
    elif path.suffix.lower() == ".json":
        path.write_text(json.dumps({"remap": canonical}, indent=2) + "\n")
    else:
        raise ValueError(f"{path}: a channel map is a .toml or .json file")
    return path


def as_map(map: MapLike, *, base: int = DEFAULT_BASE) -> ChannelMap:  # noqa: A002
    """Any form of channel map as one checked dict.

    Parameters
    ----------
    map : dict, list of dict, str or os.PathLike
        A channel map: a ``{"base", "pairs"}`` dict (or a file's ``{"remap": {...}}``), a list of
        ``{"from", "to", "name"}`` pairs, or the path of a ``.toml`` / ``.json`` map file.
    base : int
        Channel numbering of a list of pairs: 0, or 1 for 1-based numbers (TDT, MATLAB).

    Returns
    -------
    dict
        ``{"base": ..., "pairs": [...]}``.

    Raises
    ------
    ValueError
        When the map is invalid: an output without an input, an input used twice, ...
    """
    if isinstance(map, (str, PathLike)):
        return read_map(map)
    if isinstance(map, dict):
        inner = map.get("remap", map)
    else:
        inner = {"base": base, "pairs": list(map)}
    return json.loads(_layout.parse_map(json.dumps({"remap": inner})))


def map_order(map: MapLike, channels: int | None = None) -> tuple[list[int], list[str | None]]:  # noqa: A002
    """The input channel and the name of every output channel of a map.

    Parameters
    ----------
    map : dict, list of dict, str or os.PathLike
        A channel map: a ``{"base", "pairs"}`` dict (or a file's ``{"remap": {...}}``), a list of
        ``{"from", "to", "name"}`` pairs, or the path of a ``.toml`` / ``.json`` map file.
    channels : int, optional
        Number of input channels, to check the map against.

    Returns
    -------
    order : list of int
        The input channel (0-based) of each output channel.
    names : list of str or None
        Each output channel's name (``None`` when unnamed).

    Raises
    ------
    ValueError
        When the map is invalid or reads a channel past ``channels``.

    Examples
    --------
    >>> import dspk
    >>> dspk.ops.layout.map_order([{"from": 2, "to": 0}, {"from": 0, "to": 1}])
    ([2, 0], [None, None])
    """
    return _layout.map_order(json.dumps(as_map(map)), channels)


def gather_channels(data, order: Sequence[int], *, runtime: str | None = None) -> np.ndarray:
    """Rows of ``data`` in a new order, on the device: row ``i`` is channel ``order[i]``.

    Channels may be left out or repeated.

    Parameters
    ----------
    data : array_like
        Signal ``(channels, samples)``. float32, int8, int16, uint16 and int32 arrays go to the
        device as they are and are converted there; other dtypes and lists become float32 first.
    order : sequence of int
        Input channel (0-based) of each output row.
    runtime : str, optional
        A runtime from :func:`dspk.runtime.available` (``"cuda"``, ``"hip"``, ``"wgpu"``,
        ``"cpu"``) for this call. By default: the one chosen with :func:`dspk.runtime.set`, else
        ``DSPK_RUNTIME``, else the first available (GPUs first).

    Returns
    -------
    numpy.ndarray
        ``(len(order), samples)``, float32.

    Raises
    ------
    ValueError
        When ``order`` names a channel ``data`` does not have.
    RuntimeError
        When the runtime has no device.

    Examples
    --------
    >>> import numpy as np
    >>> import dspk
    >>> x = np.random.default_rng(0).standard_normal((4, 30_000)).astype(np.float32)
    >>> dspk.ops.layout.gather_channels(x, [3, 0]).shape
    (2, 30000)
    """
    return _layout.gather_channels(_channels(data), [int(c) for c in order], runtime=runtime)


def remap(data, map: MapLike, *, runtime: str | None = None) -> np.ndarray:  # noqa: A002
    """``data`` through a channel map, on the device.

    Inputs the map leaves out are dropped; :func:`map_order` gives the outputs' names.

    Parameters
    ----------
    data : array_like
        Signal ``(channels, samples)``. float32, int8, int16, uint16 and int32 arrays go to the
        device as they are and are converted there; other dtypes and lists become float32 first.
    map : dict, list of dict, str or os.PathLike
        A channel map: a ``{"base", "pairs"}`` dict (or a file's ``{"remap": {...}}``), a list of
        ``{"from", "to", "name"}`` pairs, or the path of a ``.toml`` / ``.json`` map file.
    runtime : str, optional
        A runtime from :func:`dspk.runtime.available` (``"cuda"``, ``"hip"``, ``"wgpu"``,
        ``"cpu"``) for this call. By default: the one chosen with :func:`dspk.runtime.set`, else
        ``DSPK_RUNTIME``, else the first available (GPUs first).

    Returns
    -------
    numpy.ndarray
        ``(outputs, samples)``, float32.

    Raises
    ------
    ValueError
        When the map is invalid or reads a channel ``data`` does not have.
    RuntimeError
        When the runtime has no device.

    Examples
    --------
    >>> import numpy as np
    >>> import dspk
    >>> x = np.random.default_rng(0).standard_normal((4, 30_000)).astype(np.float32)
    >>> dspk.ops.layout.remap(x, [{"from": 3, "to": 0}, {"from": 1, "to": 1}]).shape
    (2, 30000)
    """
    data = _channels(data)
    order, _ = map_order(map, data.shape[0])
    return gather_channels(data, order, runtime=runtime)


__all__ = ["DEFAULT_BASE", "ChannelMap", "MapLike", "as_map", "gather_channels", "map_order", "read_map", "remap", "write_map"]
