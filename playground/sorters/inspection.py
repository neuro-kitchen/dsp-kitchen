"""Interactive inspection of sorted units on one channel of the preprocessed signal.

The top row is the channel's trace with every spike of the units **on that channel** marked in
its unit's colour; underneath, a raster with one row per unit; on the right, each unit's template
on that channel. Several sortings can be shown together (e.g. ours and Kilosort4's saved results),
each in its own block of raster rows and line style.

A spike is on a channel when that channel is the probe contact nearest the spike's location (the
sorting's per-spike `spike_locations`, x and y); the *within radius* toggle shows every spike within
`radius_um` instead. Only numpy and matplotlib (its own widgets: works in any interactive backend).

    from inspection import inspect, preprocessed_segment

    signal, offset = preprocessed_segment(result, recording, start=10.0, end=20.0)
    inspect({"ours": result.to_sorting_output(probe), "Kilosort4": syn.load_sorting(ks4_dir)},
            signal, recording.sample_rate, offset=offset, channel=40)

EMUsort reports spike times in its reference channel's frame: pass `channel_shifts=delays` (from
`result.channel_delays`) so each channel's marks sit where the spike is on that channel.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, CheckButtons, TextBox

#: Unit colours (cycled by unit id, so a unit keeps its colour across channels and pages).
UNIT_COLORMAP = "tab20"
#: Line style of each sorting's marks (cycled).
SORTING_STYLES = ("solid", "dashed", "dotted")
#: Height of a spike mark on the trace, as a fraction of the window's peak-to-peak range.
MARK_FRACTION = 0.10
#: Fraction of the window a ◀ / ▶ step moves.
PAGE_STEP = 0.8
#: Units whose templates are drawn (the ones with most spikes on the channel).
MAX_TEMPLATES = 8


def preprocessed_segment(result, recording, start: float, end: float):
    """`[channels, samples]` of `recording` between `start` and `end` (s) through the run's own
    preprocessing (`result.preprocessing`), read with the run's halos so filters settle outside the
    segment. Returns the signal and the recording sample of its first column."""
    fs = recording.sample_rate
    s0, s1 = int(round(start * fs)), int(round(end * fs))
    s0, s1 = max(0, s0), min(recording.samples, s1)
    left, right = result.halos
    r0, r1 = max(0, s0 - int(left)), min(recording.samples, s1 + int(right))
    out = np.asarray(result.preprocessing.run(recording.read(start_sample=r0, end_sample=r1), fs=fs))
    return out[:, s0 - r0 : s1 - r0], s0


@dataclass
class Unit:
    id: int
    times: np.ndarray  # recording samples
    channel: np.ndarray  # nearest contact of each spike
    location: np.ndarray  # [n, 2] µm
    template: dict | None  # `unit_template`: mean [channels, nt], channel_ids


@dataclass
class SortingSpikes:
    name: str
    style: str
    units: list[Unit] = field(default_factory=list)

    @classmethod
    def of(cls, name, sorting, positions, style):
        units = []
        for uid in sorting.unit_ids():
            times = np.asarray(sorting.spike_train(uid), dtype=np.int64)
            loc = np.asarray(sorting.spike_locations(uid), dtype=float).reshape(len(times), -1)[:, :2]
            if len(loc) != len(times):
                # No per-spike locations: every spike at the unit's primary channel
                primary = sorting.unit_metrics(uid)["primary_channel"] or 0
                loc = np.repeat(positions[primary][None, :], len(times), axis=0)
            d = ((loc[:, None, :] - positions[None, :, :]) ** 2).sum(axis=2)
            units.append(Unit(uid, times, d.argmin(axis=1), loc, sorting.unit_template(uid)))
        return cls(name, style, units)


def _positions(sortings, positions):
    if positions is not None:
        return np.asarray(positions, dtype=float)[:, :2]
    for sorting in sortings.values():
        if sorting.probe is not None:
            return np.asarray(sorting.probe.contact_positions(), dtype=float)[:, :2]
    raise ValueError("no probe geometry: pass positions= or sortings with a probe")


class Inspector:
    """The figure and its state (see the module docs); built by [`inspect`]."""

    def __init__(self, sortings, signal, sample_rate, *, offset, channel, start, end, positions, channel_shifts, radius_um, title):
        self.signal = np.atleast_2d(np.asarray(signal))
        self.fs = float(sample_rate)
        self.offset = int(offset)
        self.positions = _positions(sortings, positions)
        n_ch, n = self.signal.shape
        if len(self.positions) != n_ch:
            raise ValueError(f"{len(self.positions)} contact positions for {n_ch} signal channels")
        self.shifts = np.zeros(n_ch, dtype=np.int64) if channel_shifts is None else np.asarray(channel_shifts, dtype=np.int64)
        spacing = np.sqrt(((self.positions[:, None] - self.positions[None]) ** 2).sum(axis=2))
        self.radius = float(radius_um) if radius_um is not None else float(np.min(spacing[spacing > 0])) if n_ch > 1 else 0.0
        self.sortings = [SortingSpikes.of(name, s, self.positions, SORTING_STYLES[i % len(SORTING_STYLES)]) for i, (name, s) in enumerate(sortings.items())]
        self.cmap = plt.get_cmap(UNIT_COLORMAP)
        self.title = title
        self.first, self.last = self.offset / self.fs, (self.offset + n) / self.fs
        self.state = {"channel": int(np.clip(channel, 0, n_ch - 1)), "near": False}
        self.state["start"] = float(np.clip(start, self.first, self.last))
        self.state["end"] = float(np.clip(self.last if end is None else end, self.state["start"], self.last))
        self._build()
        self.draw()

    # -- data -------------------------------------------------------------------------------------
    def _on_channel(self, unit: Unit):
        c = self.state["channel"]
        if self.state["near"]:
            return np.sqrt(((unit.location - self.positions[c]) ** 2).sum(axis=1)) <= self.radius
        return unit.channel == c

    def _channel_units(self):
        """Per sorting: `(unit, times in s)` of the units with spikes on the channel, most spikes first."""
        shift = self.shifts[self.state["channel"]]
        out = []
        for sorting in self.sortings:
            rows = []
            for unit in sorting.units:
                sel = self._on_channel(unit)
                if sel.any():
                    rows.append((unit, (unit.times[sel] + shift) / self.fs))
            rows.sort(key=lambda r: -len(r[1]))
            out.append(rows)
        return out

    def _color(self, unit: Unit):
        return self.cmap(unit.id % self.cmap.N)

    # -- figure -----------------------------------------------------------------------------------
    def _build(self):
        self.fig = plt.figure(figsize=(15, 8))
        grid = self.fig.add_gridspec(2, 2, width_ratios=[4, 1], height_ratios=[3, 2], left=0.06, right=0.98, top=0.93, bottom=0.17, hspace=0.08, wspace=0.12)
        self.ax_trace = self.fig.add_subplot(grid[0, 0])
        self.ax_raster = self.fig.add_subplot(grid[1, 0], sharex=self.ax_trace)
        self.ax_templates = self.fig.add_subplot(grid[:, 1])
        s = self.state
        self.box_channel = TextBox(self.fig.add_axes([0.07, 0.05, 0.06, 0.045]), "channel ", initial=str(s["channel"]))
        self.box_start = TextBox(self.fig.add_axes([0.20, 0.05, 0.08, 0.045]), "start (s) ", initial=f"{s['start']:.4g}")
        self.box_end = TextBox(self.fig.add_axes([0.35, 0.05, 0.08, 0.045]), "end (s) ", initial=f"{s['end']:.4g}")
        self.button_prev = Button(self.fig.add_axes([0.46, 0.05, 0.04, 0.045]), "◀")
        self.button_next = Button(self.fig.add_axes([0.51, 0.05, 0.04, 0.045]), "▶")
        self.toggle = CheckButtons(self.fig.add_axes([0.58, 0.035, 0.18, 0.075]), [f"within {self.radius:g} µm"], [False])
        self.box_channel.on_submit(self._on_channel_text)
        self.box_start.on_submit(self._on_start)
        self.box_end.on_submit(self._on_end)
        self.button_prev.on_clicked(lambda _: self.page(-1))
        self.button_next.on_clicked(lambda _: self.page(1))
        self.toggle.on_clicked(self._on_toggle)

    def draw(self):
        s, fs = self.state, self.fs
        c = s["channel"]
        i0 = int(np.clip(round(s["start"] * fs) - self.offset, 0, self.signal.shape[1]))
        i1 = int(np.clip(round(s["end"] * fs) - self.offset, i0, self.signal.shape[1]))
        y = self.signal[c, i0:i1]
        for ax in (self.ax_trace, self.ax_raster, self.ax_templates):
            ax.cla()
        self.ax_trace.plot((self.offset + np.arange(i0, i1)) / fs, y, color="0.2", linewidth=0.6)
        height = MARK_FRACTION * max(float(np.ptp(y)) if len(y) else 1.0, np.finfo(float).eps)

        per_sorting = self._channel_units()
        row, ticks, labels, in_window = 0, [], [], 0
        for sorting, rows in zip(self.sortings, per_sorting):
            for unit, times in rows:
                color = self._color(unit)
                shown = times[(times >= s["start"]) & (times < s["end"])]
                in_window += len(shown)
                if len(shown) and len(y):
                    idx = np.clip(np.round(shown * fs).astype(np.int64) - self.offset - i0, 0, len(y) - 1)
                    self.ax_trace.vlines(shown, y[idx] - height, y[idx] + height, colors=[color], linestyles=sorting.style, linewidth=1.6)
                self.ax_raster.vlines(shown, row + 0.1, row + 0.9, colors=[color], linestyles=sorting.style, linewidth=1.0)
                ticks.append(row + 0.5)
                labels.append(f"{sorting.name} {unit.id}" if len(self.sortings) > 1 else str(unit.id))
                row += 1
            if row and sorting is not self.sortings[-1]:
                self.ax_raster.axhline(row, color="0.6", linewidth=0.8)
        self._draw_templates(per_sorting)

        where = f"within {self.radius:g} µm of" if s["near"] else "nearest to"
        self.ax_trace.set_title(self.title or f"Channel {c}: {in_window:,} spikes of units {where} it, {s['end'] - s['start']:.3g} s", fontweight="bold")
        self.ax_trace.set_ylabel(f"channel {c} (preprocessed)")
        self.ax_trace.tick_params(labelbottom=False)
        self.ax_raster.set_ylim(0, max(row, 1))
        self.ax_raster.set_yticks(ticks, labels, fontsize=7)
        self.ax_raster.set_ylabel("unit")
        self.ax_raster.set_xlabel("time (s)")
        self.ax_raster.set_xlim(s["start"], s["end"])
        self.fig.canvas.draw_idle()

    def _draw_templates(self, per_sorting):
        c = self.state["channel"]
        ax = self.ax_templates
        drawn = 0
        for sorting, rows in zip(self.sortings, per_sorting):
            for unit, _ in rows[:MAX_TEMPLATES]:
                t = unit.template
                if t is None or c not in t["channel_ids"]:
                    continue
                wave = np.asarray(t["mean"])[list(t["channel_ids"]).index(c)]
                label = f"{sorting.name} {unit.id}" if len(self.sortings) > 1 else f"unit {unit.id}"
                ax.plot(np.arange(len(wave)) / self.fs * 1e3, wave, color=self._color(unit), linestyle=sorting.style, linewidth=1.4, label=label)
                drawn += 1
        ax.set_title(f"templates on channel {c}", fontsize=9)
        ax.set_xlabel("ms")
        ax.tick_params(labelsize=7)
        if drawn:
            ax.legend(fontsize=6, loc="lower right")
        else:
            ax.text(0.5, 0.5, "no templates", ha="center", va="center", transform=ax.transAxes, color="0.5")

    # -- controls ---------------------------------------------------------------------------------
    def set_window(self, start, end):
        width = max(end - start, 1.0 / self.fs)
        start = float(np.clip(start, self.first, max(self.first, self.last - width)))
        self.state["start"], self.state["end"] = start, min(start + width, self.last)
        self.box_start.set_val(f"{self.state['start']:.4g}")
        self.box_end.set_val(f"{self.state['end']:.4g}")
        self.draw()

    def page(self, direction):
        step = PAGE_STEP * (self.state["end"] - self.state["start"]) * direction
        self.set_window(self.state["start"] + step, self.state["end"] + step)

    def set_channel(self, channel):
        self.state["channel"] = int(np.clip(channel, 0, self.signal.shape[0] - 1))
        self.draw()

    def _on_channel_text(self, text):
        try:
            self.set_channel(int(text))
        except ValueError:
            self.box_channel.set_val(str(self.state["channel"]))

    def _on_start(self, text):
        try:
            self.set_window(float(text), max(float(text) + 1.0 / self.fs, self.state["end"]))
        except ValueError:
            self.box_start.set_val(f"{self.state['start']:.4g}")

    def _on_end(self, text):
        try:
            self.set_window(self.state["start"], float(text))
        except ValueError:
            self.box_end.set_val(f"{self.state['end']:.4g}")

    def _on_toggle(self, _):
        self.state["near"] = not self.state["near"]
        self.draw()


def inspect(sortings, signal, sample_rate, *, offset=0, channel=0, start=None, end=None, positions=None, channel_shifts=None, radius_um=None, title=None, show=True) -> Inspector:
    """Opens the viewer (module docs).

    - `sortings`: a `SortingOutput`, or `{name: SortingOutput}` to compare several.
    - `signal`: `[channels, samples]` (e.g. from [`preprocessed_segment`]); `offset`: its first
      sample in the recording; `start` / `end`: the first window (s, default the first second).
    - `positions`: `[channels, 2]` contact positions (default: the first sorting's probe).
    - `channel_shifts`: samples added to spike times per channel (EMUsort's channel delays).
    - `radius_um`: radius of the *within* toggle (default: the smallest contact spacing).

    Returns the [`Inspector`] (its figure is `.fig`; keep it alive while the window is open).
    """
    if not isinstance(sortings, dict):
        sortings = {"sorting": sortings}
    start = offset / sample_rate if start is None else start
    end = start + 1.0 if end is None else end
    inspector = Inspector(sortings, signal, sample_rate, offset=offset, channel=channel, start=start, end=end, positions=positions, channel_shifts=channel_shifts, radius_um=radius_um, title=title)
    if show:
        plt.show()
    return inspector
