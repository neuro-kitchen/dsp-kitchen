"""Probes (site geometry: presets, probe files, channel maps) and sortings (Phy / Kilosort folders,
``.sorting.zarr``, NWB ``/units``).

A :class:`Probe` follows a recording's channels: :meth:`Probe.remap` applies a channel map (see
:mod:`dspk.ops.layout`), :meth:`Probe.depth_order` gives the map that sorts channels by depth.
A :class:`Sorting` holds units with their spike trains, templates and quality metrics.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from os import PathLike

import numpy as np

from dspk._native import synapse as _synapse

_io = _synapse.io

#: Electrode pitch of the HD-EMG grid presets (µm), as the CLI's ``--pitch-um``.
DEFAULT_HDEMG_PITCH_UM: float = 8_000.0
PRESETS = ("neuropixels1", "neuropixels2", "tetrode", "utah", "hdemg4x8", "hdemg8x8")


class Probe:
    """A probe's geometry: one site per recording channel (µm; ``y`` is depth)."""

    def __init__(self, native: _io.Probe) -> None:
        self._native = native

    @classmethod
    def preset(cls, name: str, *, pitch_um: float = DEFAULT_HDEMG_PITCH_UM) -> Probe:
        """A nominal layout, one of :data:`PRESETS` (``pitch_um``: HD-EMG grids only)."""
        return cls(_io.Probe.preset(name, pitch_um=pitch_um))

    @classmethod
    def grid(cls, rows: int, cols: int, pitch_um: float) -> Probe:
        """A ``rows × cols`` grid, channel ``row * cols + col``."""
        return cls(_io.Probe.grid(rows, cols, pitch_um))

    @classmethod
    def from_positions(cls, positions, *, shanks: Sequence[int] | None = None, name: str = "probe") -> Probe:
        """Sites at ``positions`` (``(channels, 2)`` or ``(channels, 3)`` µm), channel ``i`` at row ``i``."""
        return cls(_io.Probe.from_positions(np.ascontiguousarray(positions, dtype=np.float32), shanks=None if shanks is None else list(shanks), name=name))

    @classmethod
    def from_json(cls, text: str) -> Probe:
        return cls(_io.Probe.from_json(text))

    @property
    def name(self) -> str:
        return self._native.name

    @property
    def channels(self) -> list[int]:
        """The recording channel of every site, in site order."""
        return self._native.channels

    @property
    def positions(self) -> np.ndarray:
        """Site positions ``(sites, 3)`` in µm (x, y = depth, z)."""
        return self._native.positions

    @property
    def shanks(self) -> list[int]:
        return self._native.shanks

    def remap(self, map) -> Probe:  # noqa: A002
        """The geometry after a channel map (any form :func:`dspk.ops.layout.as_map` takes)."""
        from dspk.ops.layout import as_map

        return Probe(self._native.remap(json.dumps(as_map(map))))

    def depth_order(self) -> dict:
        """A channel map sorting the channels by shank, depth, then x (``{"base", "pairs"}``)."""
        return json.loads(self._native.depth_order())

    def select(self, channels: Sequence[int]) -> Probe:
        """The sites of ``channels`` (in that order), renumbered ``0..``."""
        return Probe(self._native.select([int(c) for c in channels]))

    def neighbours(self, k: int) -> np.ndarray:
        """The ``k`` nearest sites of every channel (itself first): ``(channels, k)``."""
        return self._native.neighbours(k)

    def to_json(self) -> str:
        return self._native.to_json()

    def __len__(self) -> int:
        return len(self._native)

    def __repr__(self) -> str:
        return repr(self._native)


def read_probe_file(path: str | PathLike) -> tuple[Probe, dict]:
    """A probe file (``.json``: probeinterface or Kilosort 4): the probe in contact order and the
    channel map that reads a recording in that order (``.mat`` files are refused)."""
    native, map_json = _io.read_probe_file(path)
    return Probe(native), json.loads(map_json)


def probe_of(path: str | PathLike, source: str | None = None) -> Probe | None:
    """The probe a recording file carries (SpikeGLX), or ``None``."""
    native = _io.probe_of(path, source)
    return None if native is None else Probe(native)


class Sorting:
    """Units with their spike trains (sample indices), templates and quality metrics."""

    def __init__(self, native: _io.Sorting) -> None:
        self._native = native

    @classmethod
    def load(cls, path: str | PathLike) -> Sorting:
        """A Phy / Kilosort folder, a ``.sorting.zarr`` store or an NWB ``.nwb.zarr`` (``/units``)."""
        return cls(_io.Sorting.load(path))

    @classmethod
    def from_spikes(cls, spike_samples, spike_units, sample_rate: float, total_samples: int, *, sorter_name: str = "dspk", probe: Probe | None = None) -> Sorting:
        """From flat arrays: the sample of every spike and its unit; metrics computed from the trains."""
        return cls(_io.Sorting.from_spikes(
            [int(s) for s in np.asarray(spike_samples, dtype=np.uint64)], [int(u) for u in np.asarray(spike_units)],
            sample_rate, int(total_samples), sorter_name=sorter_name, probe=None if probe is None else probe._native,
        ))

    def save(self, path: str | PathLike, format: str | None = None) -> None:  # noqa: A002
        """As ``"phy"``, ``"sorting_zarr"`` or ``"nwb_units"`` (default: from the path's name)."""
        self._native.save(path, format)

    @property
    def sorter_name(self) -> str:
        return self._native.sorter_name

    @property
    def sample_rate(self) -> float:
        return self._native.sample_rate

    @property
    def total_samples(self) -> int:
        return self._native.total_samples

    @property
    def unit_ids(self) -> list[int]:
        return self._native.unit_ids

    @property
    def probe(self) -> Probe | None:
        native = self._native.probe
        return None if native is None else Probe(native)

    def spikes(self) -> dict[str, np.ndarray]:
        """Every spike, by time: ``samples``, ``units``, ``amplitudes`` (µV), ``locations`` (µm)."""
        return self._native.spikes()

    def spike_train(self, unit_id: int) -> np.ndarray:
        """Sample indices of ``unit_id``'s spikes (sorted, ``uint64``)."""
        return self._native.spike_train(unit_id)

    def template(self, unit_id: int) -> dict | None:
        """``mean`` / ``std`` / ``se`` ``(channels, samples)`` µV on ``channels``, ``count``, ``trough_index``."""
        return self._native.template(unit_id)

    def units(self) -> dict:
        """One row per unit (columns): ``unit_id``, ``primary_channel``, ``spikes``,
        ``firing_rate_hz``, ``snr``, ``isi_violation_ratio``, ``presence_ratio``,
        ``amplitude_cutoff``, ``composite_score``, ``label``."""
        return self._native.units()

    def __len__(self) -> int:
        return len(self._native)

    def __repr__(self) -> str:
        return repr(self._native)


__all__ = ["DEFAULT_HDEMG_PITCH_UM", "PRESETS", "Probe", "Sorting", "probe_of", "read_probe_file"]
