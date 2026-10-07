"""Interactive inspection of one channel of a sorted recording: the preprocessed trace with the
detected spikes coloured by template, and a raster of every template underneath.

Only numpy and matplotlib (already dependencies of the project); the interaction uses
matplotlib's own widgets, so it works in any interactive backend.

    from inspection import inspect_channel
    inspect_channel(signal, fs, spikes["sample"], spikes["template"], channel=5, start=0.0, end=2.0)

`signal` is a `[channels, samples]` array (e.g. the run's preprocessing applied to a stretch of
the recording) whose first sample is recording sample `offset`; spike samples are recording
samples. The window can be moved and resized within that stretch.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, CheckButtons, TextBox

# Colours of templates (cycled when there are more templates than colours).
TEMPLATE_COLORMAP = "tab10"
# Height of a spike marker on the trace, as a fraction of the window's peak-to-peak range.
MARKER_FRACTION = 0.08
# Fraction of the window a ◀ / ▶ step moves.
PAGE_STEP = 0.8


def inspect_channel(
    signal: np.ndarray,
    sample_rate: float,
    spike_samples,
    spike_templates=None,
    *,
    channel: int = 0,
    start: float = 0.0,
    end: float | None = None,
    offset: int = 0,
    channel_shifts=None,
    spike_y_um=None,
    channel_y_um=None,
    near_um: float | None = None,
    title: str | None = None,
    show: bool = True,
):
    """Opens an interactive view of `signal[channel]` between `start` and `end` (seconds of the
    recording) with the spikes marked and a raster per template underneath.

    - `spike_samples`: recording samples of the detected spikes; `spike_templates`: the template
      of each (all 0 when omitted), which sets its colour and raster row.
    - `offset`: recording sample of `signal[:, 0]`.
    - `channel_shifts`: samples added to every spike time when channel `c` is shown (EMUsort reports
      spikes in its reference channel's frame: pass the channel delays).
    - `spike_y_um`, `channel_y_um`, `near_um`: with all three, a toggle shows only spikes within
      `near_um` of the channel's position.

    Returns the figure (the widgets live as long as it does).
    """
    signal = np.asarray(signal)
    if signal.ndim == 1:
        signal = signal[None, :]
    channels, length = signal.shape
    fs = float(sample_rate)
    samples = np.asarray(spike_samples, dtype=np.int64)
    templates = np.zeros(len(samples), dtype=np.int64) if spike_templates is None else np.asarray(spike_templates, dtype=np.int64)
    if len(templates) != len(samples):
        raise ValueError(f"{len(samples)} spike samples but {len(templates)} templates")
    shifts = None if channel_shifts is None else np.asarray(channel_shifts, dtype=np.int64)
    can_filter = spike_y_um is not None and channel_y_um is not None and near_um is not None
    spike_y = None if spike_y_um is None else np.asarray(spike_y_um, dtype=float)
    chan_y = None if channel_y_um is None else np.asarray(channel_y_um, dtype=float)

    n_templates = int(templates.max()) + 1 if len(templates) else 1
    cmap = plt.get_cmap(TEMPLATE_COLORMAP)
    colors = [cmap(k % cmap.N) for k in range(n_templates)]
    first_s, last_s = offset / fs, (offset + length) / fs

    state = {
        "channel": int(np.clip(channel, 0, channels - 1)),
        "start": float(np.clip(start, first_s, last_s)),
        "end": float(np.clip(last_s if end is None else end, first_s, last_s)),
        "near": can_filter,
    }

    fig = plt.figure(figsize=(14, 7))
    grid = fig.add_gridspec(2, 1, height_ratios=[3, 2], left=0.07, right=0.98, top=0.93, bottom=0.2, hspace=0.08)
    ax_trace = fig.add_subplot(grid[0])
    ax_raster = fig.add_subplot(grid[1], sharex=ax_trace)

    def visible_spikes():
        """Spike times (s) and templates in the window for the current channel."""
        times = samples + (shifts[state["channel"]] if shifts is not None else 0)
        keep = (times >= state["start"] * fs) & (times < state["end"] * fs)
        if state["near"] and can_filter:
            keep &= np.abs(spike_y - chan_y[state["channel"]]) <= near_um
        return times[keep] / fs, templates[keep]

    def draw():
        c, s0, s1 = state["channel"], state["start"], state["end"]
        i0 = int(np.clip(round(s0 * fs) - offset, 0, length))
        i1 = int(np.clip(round(s1 * fs) - offset, i0, length))
        t = (offset + np.arange(i0, i1)) / fs
        y = signal[c, i0:i1]
        ax_trace.cla()
        ax_raster.cla()
        ax_trace.plot(t, y, color="0.25", linewidth=0.6)
        times, temps = visible_spikes()
        if len(y):
            lo, hi = float(np.min(y)), float(np.max(y))
            height = MARKER_FRACTION * max(hi - lo, np.finfo(float).eps)
            idx = np.clip(np.round(times * fs).astype(np.int64) - offset, 0, length - 1)
            values = signal[c, idx]
            for k in np.unique(temps):
                sel = temps == k
                ax_trace.vlines(times[sel], values[sel] - height, values[sel] + height, color=colors[k], linewidth=1.5)
                ax_raster.vlines(times[sel], k + 0.1, k + 0.9, color=colors[k], linewidth=1.0)
        ax_trace.set_ylabel(f"channel {c}")
        ax_trace.set_title(title or f"Channel {c}: {len(times):,} spikes in {s1 - s0:.3g} s", fontweight="bold")
        ax_trace.tick_params(labelbottom=False)
        ax_raster.set_ylim(0, n_templates)
        ax_raster.set_yticks(np.arange(n_templates) + 0.5, [str(k) for k in range(n_templates)])
        ax_raster.set_ylabel("template")
        ax_raster.set_xlabel("time (s)")
        ax_raster.set_xlim(s0, s1)
        fig.canvas.draw_idle()

    def set_window(s0, s1):
        width = max(s1 - s0, 1.0 / fs)
        s0 = float(np.clip(s0, first_s, max(first_s, last_s - width)))
        state["start"], state["end"] = s0, min(s0 + width, last_s)
        box_start.set_val(f"{state['start']:.4g}")
        box_end.set_val(f"{state['end']:.4g}")

    def on_channel(text):
        try:
            state["channel"] = int(np.clip(int(text), 0, channels - 1))
        except ValueError:
            # Unreadable input: show the current value again (which redraws)
            box_channel.set_val(str(state["channel"]))
            return
        draw()

    def on_start(text):
        try:
            state["start"] = float(np.clip(float(text), first_s, state["end"]))
        except ValueError:
            box_start.set_val(f"{state['start']:.4g}")
            return
        draw()

    def on_end(text):
        try:
            state["end"] = float(np.clip(float(text), state["start"], last_s))
        except ValueError:
            box_end.set_val(f"{state['end']:.4g}")
            return
        draw()

    def page(direction):
        step = PAGE_STEP * (state["end"] - state["start"]) * direction
        set_window(state["start"] + step, state["end"] + step)

    box_channel = TextBox(fig.add_axes([0.07, 0.06, 0.08, 0.05]), "channel ", initial=str(state["channel"]))
    box_start = TextBox(fig.add_axes([0.24, 0.06, 0.1, 0.05]), "start (s) ", initial=f"{state['start']:.4g}")
    box_end = TextBox(fig.add_axes([0.42, 0.06, 0.1, 0.05]), "end (s) ", initial=f"{state['end']:.4g}")
    button_prev = Button(fig.add_axes([0.56, 0.06, 0.05, 0.05]), "◀")
    button_next = Button(fig.add_axes([0.62, 0.06, 0.05, 0.05]), "▶")
    box_channel.on_submit(on_channel)
    box_start.on_submit(on_start)
    box_end.on_submit(on_end)
    button_prev.on_clicked(lambda _: page(-1))
    button_next.on_clicked(lambda _: page(1))
    widgets = [box_channel, box_start, box_end, button_prev, button_next]
    if can_filter:
        toggle = CheckButtons(fig.add_axes([0.71, 0.04, 0.2, 0.09]), [f"only spikes within {near_um:g} µm"], [state["near"]])

        def on_toggle(_):
            state["near"] = not state["near"]
            draw()

        toggle.on_clicked(on_toggle)
        widgets.append(toggle)
    # Matplotlib widgets stop responding once garbage-collected: keep them with the figure
    fig._inspection_widgets = widgets

    draw()
    if show:
        plt.show()
    return fig


if __name__ == "__main__":
    # Self-contained demo on synthetic data: 4 channels, 3 templates, 10 s at 30 kHz
    rng = np.random.default_rng(0)
    fs, n, channels = 30_000.0, 300_000, 4
    signal = rng.normal(0.0, 1.0, (channels, n)).astype(np.float32)
    spike_samples = np.sort(rng.integers(100, n - 100, 600))
    spike_templates = rng.integers(0, 3, len(spike_samples))
    shape = -np.exp(-(((np.arange(41) - 20) / 4.0) ** 2))
    for s, k in zip(spike_samples, spike_templates):
        signal[k % channels, s - 20 : s + 21] += (6 + 3 * k) * shape
    inspect_channel(signal, fs, spike_samples, spike_templates, channel=0, start=0.0, end=1.0)
