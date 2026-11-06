"""Channels in a new order on the device, and channel maps.

A channel map says which input channel becomes which output channel, as pairs (like a hardware
channel mapper)::

    [remap]
    base = 0            # 1 for 1-based numbers (TDT, MATLAB)
    pairs = [
      { from = 4, to = 0, name = "A1" },
      { from = 2, to = 1, name = "A2" },
    ]

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
    """A map file (``.toml`` or ``.json``, the pairs under ``remap``), checked: ``{"base", "pairs"}``."""
    path = Path(path)
    if path.suffix.lower() not in (".toml", ".json"):
        raise ValueError(f"{path}: a channel map is a .toml or .json file")
    return json.loads(_layout.parse_map(path.read_text(), toml=path.suffix.lower() == ".toml"))


def write_map(map: MapLike, path: str | PathLike) -> Path:  # noqa: A002
    """Writes a map file (``.toml`` or ``.json`` by the suffix); returns its path."""
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
    """Any map form as a checked ``{"base", "pairs"}`` dict (``base`` applies to a list of pairs)."""
    if isinstance(map, (str, PathLike)):
        return read_map(map)
    if isinstance(map, dict):
        inner = map.get("remap", map)
    else:
        inner = {"base": base, "pairs": list(map)}
    return json.loads(_layout.parse_map(json.dumps({"remap": inner})))


def map_order(map: MapLike, channels: int | None = None) -> tuple[list[int], list[str | None]]:  # noqa: A002
    """The input channel (0-based) of every output channel, and each output's name (``None``:
    unnamed), checked against ``channels`` input channels when given."""
    return _layout.map_order(json.dumps(as_map(map)), channels)


def gather_channels(data, order: Sequence[int], *, runtime: str | None = None) -> np.ndarray:
    """``data`` ``(channels, samples)`` with row ``i`` = channel ``order[i]`` (0-based; channels may
    be left out or repeated), on the device. Returns float32 ``(len(order), samples)``."""
    return _layout.gather_channels(_channels(data), [int(c) for c in order], runtime=runtime)


def remap(data, map: MapLike, *, runtime: str | None = None) -> np.ndarray:  # noqa: A002
    """``data`` ``(channels, samples)`` through a channel map (see the module docs), on the device.
    Returns float32 ``(outputs, samples)``; :func:`map_order` gives the outputs' names."""
    data = _channels(data)
    order, _ = map_order(map, data.shape[0])
    return gather_channels(data, order, runtime=runtime)


__all__ = ["DEFAULT_BASE", "ChannelMap", "MapLike", "as_map", "gather_channels", "map_order", "read_map", "remap", "write_map"]
