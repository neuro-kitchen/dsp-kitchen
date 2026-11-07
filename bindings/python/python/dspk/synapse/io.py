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
    """A probe's geometry: the position of the site under every recording channel.

    Positions are in µm, ``y`` is depth. Make one with :meth:`preset`, :meth:`grid`,
    :meth:`from_positions`, :func:`read_probe_file` or :func:`probe_of`.

    Examples
    --------
    >>> import dspk
    >>> probe = dspk.synapse.io.Probe.preset("tetrode")
    >>> len(probe)
    4
    """

    def __init__(self, native: _io.Probe) -> None:
        self._native = native

    @classmethod
    def preset(cls, name: str, *, pitch_um: float = DEFAULT_HDEMG_PITCH_UM) -> Probe:
        """A nominal layout by name.

        Parameters
        ----------
        name : {"neuropixels1", "neuropixels2", "tetrode", "utah", "hdemg4x8", "hdemg8x8"}
            One of :data:`PRESETS`.
        pitch_um : float
            Electrode spacing of the HD-EMG grids, in µm; the other presets ignore it.

        Returns
        -------
        Probe

        Raises
        ------
        ValueError
            For an unknown name.
        """
        return cls(_io.Probe.preset(name, pitch_um=pitch_um))

    @classmethod
    def grid(cls, rows: int, cols: int, pitch_um: float) -> Probe:
        """A ``rows x cols`` grid; channel ``row * cols + col`` sits at ``(col, row) * pitch_um``.

        Parameters
        ----------
        rows : int
            Grid rows.
        cols : int
            Grid columns.
        pitch_um : float
            Electrode spacing, in µm.

        Returns
        -------
        Probe
        """
        return cls(_io.Probe.grid(rows, cols, pitch_um))

    @classmethod
    def from_positions(cls, positions, *, shanks: Sequence[int] | None = None, name: str = "probe") -> Probe:
        """Sites at given positions, channel ``i`` at row ``i``.

        Parameters
        ----------
        positions : array_like
            ``(channels, 2)`` or ``(channels, 3)`` in µm.
        shanks : sequence of int, optional
            The shank of each site; by default all on one.
        name : str
            A name for the probe.

        Returns
        -------
        Probe
        """
        return cls(_io.Probe.from_positions(np.ascontiguousarray(positions, dtype=np.float32), shanks=None if shanks is None else list(shanks), name=name))

    @classmethod
    def from_json(cls, text: str) -> Probe:
        """A probe from the JSON :meth:`to_json` writes.

        Parameters
        ----------
        text : str
            The JSON.

        Returns
        -------
        Probe

        Raises
        ------
        ValueError
            When the JSON is not a probe.
        """
        return cls(_io.Probe.from_json(text))

    @property
    def name(self) -> str:
        """The probe's name.
        """
        return self._native.name

    @property
    def channels(self) -> list[int]:
        """The recording channel of every site, in site order.
        """
        return self._native.channels

    @property
    def positions(self) -> np.ndarray:
        """Site positions ``(sites, 3)`` in µm: ``x``, ``y`` (depth), ``z``.
        """
        return self._native.positions

    @property
    def shanks(self) -> list[int]:
        """The shank of every site.
        """
        return self._native.shanks

    def remap(self, map) -> Probe:  # noqa: A002
        """The geometry after a channel map, so it follows a remapped recording.

        Parameters
        ----------
        map : dict, list of dict, str or os.PathLike
            Any form :func:`dspk.ops.layout.as_map` takes.

        Returns
        -------
        Probe

        Raises
        ------
        ValueError
            When the map is invalid for this probe.
        """
        from dspk.ops.layout import as_map

        return Probe(self._native.remap(json.dumps(as_map(map))))

    def depth_order(self) -> dict:
        """A channel map that sorts the channels by shank, then depth, then ``x``.

        Returns
        -------
        dict
            ``{"base", "pairs"}``, for :func:`dspk.ops.layout.remap` and :meth:`remap`.
        """
        return json.loads(self._native.depth_order())

    def select(self, channels: Sequence[int]) -> Probe:
        """The sites of some channels, renumbered from 0 in the order given.

        Parameters
        ----------
        channels : sequence of int
            Recording channels to keep.

        Returns
        -------
        Probe
        """
        return Probe(self._native.select([int(c) for c in channels]))

    def neighbours(self, k: int) -> np.ndarray:
        """The ``k`` nearest sites of every channel, itself first.

        Parameters
        ----------
        k : int
            Sites per channel.

        Returns
        -------
        numpy.ndarray
            ``(channels, k)`` channel indices.
        """
        return self._native.neighbours(k)

    def to_json(self) -> str:
        """The probe as JSON (read back with :meth:`from_json`).
        """
        return self._native.to_json()

    def __len__(self) -> int:
        return len(self._native)

    def __repr__(self) -> str:
        return repr(self._native)


def read_probe_file(path: str | PathLike) -> tuple[Probe, dict]:
    """Reads a probe file and the channel map that puts a recording in its contact order.

    Parameters
    ----------
    path : str or os.PathLike
        A ``.json`` probe: probeinterface or Kilosort 4. ``.mat`` files are refused.

    Returns
    -------
    probe : Probe
        The probe, in contact order.
    map : dict
        The channel map that reads a recording in that order.

    Raises
    ------
    OSError
        When the file cannot be read.
    ValueError
        When it is not a probe file dspk reads.
    """
    native, map_json = _io.read_probe_file(path)
    return Probe(native), json.loads(map_json)


def probe_of(path: str | PathLike, source: str | None = None) -> Probe | None:
    """The probe a recording file carries, if any (SpikeGLX metadata).

    Parameters
    ----------
    path : str or os.PathLike
        The recording.
    source : str, optional
        One of the file's sources.

    Returns
    -------
    Probe or None
    """
    native = _io.probe_of(path, source)
    return None if native is None else Probe(native)


class Sorting:
    """A spike sorting: units with their spike trains (sample indices), templates and quality
    metrics.

    Make one with :meth:`load`, :meth:`from_spikes` or :func:`dspk.synapse.detection.detect_recording`.

    Examples
    --------
    >>> import numpy as np
    >>> import dspk
    >>> s = dspk.synapse.io.Sorting.from_spikes([100, 900, 1500], [0, 1, 0], 30_000.0, 30_000)
    >>> s.unit_ids
    [0, 1]
    """

    def __init__(self, native: _io.Sorting) -> None:
        self._native = native

    @classmethod
    def load(cls, path: str | PathLike) -> Sorting:
        """Loads a sorting, its format detected from its contents.

        Parameters
        ----------
        path : str or os.PathLike
            A Phy / Kilosort folder, a ``.sorting.zarr`` store, or an NWB ``.nwb.zarr`` (``/units``).

        Returns
        -------
        Sorting

        Raises
        ------
        OSError
            When ``path`` does not exist or cannot be read.
        ValueError
            When it is not a sorting dspk reads.
        """
        return cls(_io.Sorting.load(path))

    @classmethod
    def from_spikes(cls, spike_samples, spike_units, sample_rate: float, total_samples: int, *, sorter_name: str = "dspk", probe: Probe | None = None) -> Sorting:
        """A sorting from flat arrays: the sample of every spike and its unit.

        Metrics are computed from the spike trains.

        Parameters
        ----------
        spike_samples : array_like
            Sample index of every spike.
        spike_units : array_like
            Unit of every spike.
        sample_rate : float
            Sample rate in Hz.
        total_samples : int
            Samples in the recording (for rates and presence).
        sorter_name : str
            Who made it.
        probe : Probe, optional
            The probe the recording was made with.

        Returns
        -------
        Sorting
        """
        return cls(_io.Sorting.from_spikes(
            [int(s) for s in np.asarray(spike_samples, dtype=np.uint64)], [int(u) for u in np.asarray(spike_units)],
            sample_rate, int(total_samples), sorter_name=sorter_name, probe=None if probe is None else probe._native,
        ))

    def save(self, path: str | PathLike, format: str | None = None) -> None:  # noqa: A002
        """Saves the sorting.

        Parameters
        ----------
        path : str or os.PathLike
            Where to write it.
        format : {"phy", "sorting_zarr", "nwb_units"}, optional
            By default from the path's name: ``.nwb.zarr`` → ``"nwb_units"``, other ``.zarr`` →
            ``"sorting_zarr"``, anything else → ``"phy"`` (a folder).

        Raises
        ------
        OSError
            When it cannot be written.
        """
        self._native.save(path, format)

    @property
    def sorter_name(self) -> str:
        """Who made the sorting.
        """
        return self._native.sorter_name

    @property
    def sample_rate(self) -> float:
        """Sample rate of the recording, in Hz.
        """
        return self._native.sample_rate

    @property
    def total_samples(self) -> int:
        """Samples in the recording.
        """
        return self._native.total_samples

    @property
    def unit_ids(self) -> list[int]:
        """The unit ids.
        """
        return self._native.unit_ids

    @property
    def probe(self) -> Probe | None:
        """The probe, when the sorting has one.
        """
        native = self._native.probe
        return None if native is None else Probe(native)

    def spikes(self) -> dict[str, np.ndarray]:
        """Every spike, by time.

        Returns
        -------
        dict of numpy.ndarray
            ``samples``, ``units``, ``amplitudes`` (µV) and ``locations`` (µm; NaN where not located).
        """
        return self._native.spikes()

    def spike_train(self, unit_id: int) -> np.ndarray:
        """The spike times of one unit.

        Parameters
        ----------
        unit_id : int
            The unit.

        Returns
        -------
        numpy.ndarray
            Sample indices, sorted, uint64.
        """
        return self._native.spike_train(unit_id)

    def template(self, unit_id: int) -> dict | None:
        """The template of one unit, when the sorting has templates.

        Parameters
        ----------
        unit_id : int
            The unit.

        Returns
        -------
        dict or None
            ``mean``, ``std`` and ``se`` ``(channels, samples)`` in µV on ``channels``, ``count`` (spikes
            averaged) and ``trough_index``.
        """
        return self._native.template(unit_id)

    def units(self) -> dict:
        """One row per unit, as columns.

        Returns
        -------
        dict of list
            ``unit_id``, ``primary_channel``, ``spikes``, ``firing_rate_hz``, ``snr``,
            ``isi_violation_ratio``, ``presence_ratio``, ``amplitude_cutoff``, ``composite_score``,
            ``label``.
        """
        return self._native.units()

    def __len__(self) -> int:
        return len(self._native)

    def __repr__(self) -> str:
        return repr(self._native)


__all__ = ["DEFAULT_HDEMG_PITCH_UM", "PRESETS", "Probe", "Sorting", "probe_of", "read_probe_file"]
