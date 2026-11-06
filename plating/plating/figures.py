"""Ready figures, drawn in the theme of :mod:`plating.theme`.

Panel helpers draw into an axes you give them (``traces``, ``overlay``, ``spectrum``,
``response``, ``raster``, ``waveforms``); figure helpers build a whole figure and return
``(fig, axes)`` for :func:`plating.save` (``before_after``, ``timings``, ``agreement``).

Colour rules: data before an operation (or a reference) is drawn in the muted ink, the result in
categorical slot 1; several results take the slots in order; text stays in the ink colours.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from plating import theme
from plating.results import read

#: Welch segment length of :func:`psd` (samples; shorter signals use their own length).
DEFAULT_NPERSEG = 4096


def _window(data: np.ndarray, sample_rate: float, start: float, duration: float | None) -> tuple[np.ndarray, np.ndarray]:
    data = np.atleast_2d(np.asarray(data))
    a = int(round(start * sample_rate))
    b = data.shape[1] if duration is None else min(data.shape[1], a + int(round(duration * sample_rate)))
    return (a + np.arange(b - a)) / sample_rate, data[:, a:b]


def traces(ax, data, sample_rate: float, *, start: float = 0.0, duration: float | None = None, spacing: float | None = None,
           color: str | None = None, label: str | None = None, names: Sequence[str] | None = None, linewidth: float = 0.8) -> float:
    """Channels stacked top to bottom (channel 0 on top), ``spacing`` apart (default: 6 × the median
    standard deviation). Returns the spacing used, so a second call can share it."""
    t, x = _window(data, sample_rate, start, duration)
    if spacing is None:
        spacing = 6.0 * float(np.median(np.std(x, axis=1))) or 1.0
    color = color or theme.palette().series[0]
    for i, row in enumerate(x):
        ax.plot(t, row - i * spacing, color=color, linewidth=linewidth, label=label if i == 0 else None)
    ax.set_yticks([-i * spacing for i in range(x.shape[0])])
    ax.set_yticklabels(names if names is not None else [f"ch {i}" for i in range(x.shape[0])])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("time (s)")
    ax.set_xlim(t[0], t[-1])
    return spacing


def overlay(ax, t, series: Mapping[str, np.ndarray], *, reference: str | None = None, unit: str = "µV") -> None:
    """Several 1-D signals over the same time axis; ``reference`` (one of the keys) in the muted ink,
    the others in the categorical slots, in order. A legend names them."""
    p = theme.palette()
    slot = 0
    for name, y in series.items():
        if name == reference:
            ax.plot(t, y, color=p.muted, linewidth=1.0, label=name, zorder=1)
        else:
            ax.plot(t, y, color=p.series[slot], linewidth=1.3, label=name, zorder=2 + slot)
            slot += 1
    ax.set_xlabel("time (s)")
    ax.set_ylabel(unit)
    ax.set_xlim(t[0], t[-1])
    ax.legend(loc="upper right", ncols=min(len(series), 4))


def psd(data, sample_rate: float, *, nperseg: int = DEFAULT_NPERSEG) -> tuple[np.ndarray, np.ndarray]:
    """Welch power spectral density (Hann, 50 % overlap), averaged over channels: ``(freqs, power)``
    in unit²/Hz."""
    x = np.atleast_2d(np.asarray(data, dtype=np.float64))
    n = min(nperseg, x.shape[1])
    step = max(1, n // 2)
    w = np.hanning(n)
    scale = 1.0 / (sample_rate * np.sum(w**2))
    starts = range(0, x.shape[1] - n + 1, step)
    acc = np.zeros(n // 2 + 1)
    count = 0
    for s in starts:
        seg = x[:, s : s + n]
        seg = seg - seg.mean(axis=1, keepdims=True)
        spec = np.abs(np.fft.rfft(seg * w, axis=1)) ** 2 * scale
        spec[:, 1:-1] *= 2
        acc += spec.mean(axis=0)
        count += 1
    return np.fft.rfftfreq(n, 1 / sample_rate), acc / max(count, 1)


def spectrum(ax, data, sample_rate: float, *, label: str | None = None, color: str | None = None, nperseg: int = DEFAULT_NPERSEG,
             fmin: float = 1.0, fmax: float | None = None) -> None:
    """Power spectrum (dB re 1 unit²/Hz) on a log frequency axis."""
    f, pxx = psd(data, sample_rate, nperseg=nperseg)
    # The Nyquist bin holds half a bin's power: left out
    keep = (f >= fmin) & (f < (fmax or sample_rate / 2))
    ax.semilogx(f[keep], 10 * np.log10(pxx[keep] + 1e-20), color=color or theme.palette().series[0], label=label, linewidth=1.2)
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("power (dB)")


def response(ax, freqs, magnitude, *, label: str | None = None, color: str | None = None, db: bool = True) -> None:
    """A filter's magnitude response (``magnitude`` linear; drawn in dB unless ``db=False``)."""
    m = np.asarray(magnitude, dtype=np.float64)
    y = 20 * np.log10(np.maximum(np.abs(m), 1e-12)) if db else np.abs(m)
    ax.semilogx(freqs, y, color=color or theme.palette().series[0], label=label, linewidth=1.4)
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("gain (dB)" if db else "gain")


def raster(ax, spike_samples, spike_units, sample_rate: float, *, unit_names: Sequence[str] | None = None, markersize: float = 6.0) -> None:
    """Spike times per unit, one row per unit, each unit in its categorical slot (8 at most)."""
    p = theme.palette()
    units = np.unique(spike_units)
    if units.size > len(p.series):
        raise ValueError(f"{units.size} units: a raster shows at most {len(p.series)} (facet or pick)")
    for row, u in enumerate(units):
        t = np.asarray(spike_samples)[np.asarray(spike_units) == u] / sample_rate
        ax.plot(t, np.full(t.size, row), "|", color=p.series[row], markersize=markersize, markeredgewidth=1.4)
    ax.set_yticks(range(units.size))
    ax.set_yticklabels(unit_names if unit_names is not None else [f"unit {u}" for u in units])
    ax.set_ylim(-0.6, units.size - 0.4)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("time (s)")


def waveforms(ax, snippets, sample_rate: float, *, color: str | None = None, label: str | None = None, show: int = 60) -> None:
    """Up to ``show`` individual waveforms (faint) and their mean (solid); ``snippets`` is
    ``(n, samples)``, time in ms around the centre."""
    s = np.atleast_2d(np.asarray(snippets))
    t = (np.arange(s.shape[1]) - s.shape[1] // 2) / sample_rate * 1e3
    c = color or theme.palette().series[0]
    for row in s[:show]:
        ax.plot(t, row, color=c, alpha=0.12, linewidth=0.7)
    ax.plot(t, s.mean(axis=0), color=c, linewidth=2.0, label=label)
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("µV")


def before_after(before, after, sample_rate: float, *, title: str, channel: int = 0, start: float = 0.0, duration: float = 0.2,
                 labels: tuple[str, str] = ("before", "after"), subtitles: tuple[str, str] = ("one channel", "power spectrum, all channels"),
                 fmax: float | None = None):
    """The usual figure of an operation: one channel over a short window (left) and the power
    spectrum of all channels (right), before (muted) and after (slot 1)."""
    p = theme.palette()
    fig, (left, right) = theme.figure(title, cols=2, size=theme.WIDE)
    t, b = _window(before, sample_rate, start, duration)
    _, a = _window(after, sample_rate, start, duration)
    overlay(left, t, {labels[0]: b[channel], labels[1]: a[channel]}, reference=labels[0])
    left.set_title(subtitles[0])
    spectrum(right, before, sample_rate, label=labels[0], color=p.muted, fmax=fmax)
    spectrum(right, after, sample_rate, label=labels[1], color=p.series[0], fmax=fmax)
    right.set_title(subtitles[1])
    right.legend(loc="upper right")
    return fig, (left, right)


def _record(record) -> dict:
    return read(record) if isinstance(record, (str, Path)) else record


def timings(record, *, title: str | None = None):
    """dspk's time against the reference's per case (horizontal bars, log scale), from a benchmark
    record (a dict or a path). Each bar carries its value; the speed-up is printed per case."""
    r = _record(record)
    p = theme.palette()
    cases = [c for c in r["cases"] if c.get("dspk_ms") is not None]
    ref_name = (r.get("reference") or {}).get("tool", "")
    fig, ax = theme.figure(title or f"{r['benchmark']}: time per case", size=(11.0, 1.2 + 0.55 * len(cases) * (2 if ref_name else 1)))
    h = 0.38
    for i, c in enumerate(cases):
        y = -i
        ax.barh(y + (h / 2 if ref_name else 0), c["dspk_ms"], height=h, color=p.series[0], label=f"dspk ({r['dspk']['runtime']})" if i == 0 else None)
        ax.text(c["dspk_ms"] * 1.06, y + (h / 2 if ref_name else 0), f"{c['dspk_ms']:.3g} ms", va="center", fontsize=8.5, color=p.ink)
        if ref_name and c.get("reference_ms") is not None:
            ax.barh(y - h / 2, c["reference_ms"], height=h, color=p.series[1], label=ref_name if i == 0 else None)
            speed = c["reference_ms"] / c["dspk_ms"] if c["dspk_ms"] else float("nan")
            ax.text(c["reference_ms"] * 1.06, y - h / 2, f"{c['reference_ms']:.3g} ms  ({speed:.1f}× faster)" if speed >= 1 else f"{c['reference_ms']:.3g} ms", va="center", fontsize=8.5, color=p.ink2)
    ax.set_yticks([-i for i in range(len(cases))])
    ax.set_yticklabels([c["name"] for c in cases])
    ax.set_xscale("log")
    ax.set_xlabel("milliseconds (log)")
    ax.grid(axis="y", visible=False)
    lo, hi = ax.get_xlim()
    ax.set_xlim(lo, hi * 4)
    ax.legend(loc="lower right")
    ax.set_title(f"{r['dspk']['version']} · {r['date']}" + (f" · against {ref_name} {r['reference'].get('version', '')}" if ref_name else ""))
    return fig, ax


def agreement(record, *, title: str | None = None):
    """Each case's relative error against its tolerance (log scale): a dot per case, its tolerance as
    a tick; passes and failures named in the label, not by colour alone."""
    r = _record(record)
    p = theme.palette()
    cases = [c for c in r["cases"] if c.get("rel_err") is not None]
    fig, ax = theme.figure(title or f"{r['benchmark']}: agreement with {(r.get('reference') or {}).get('tool', 'the reference')}", size=(11.0, 1.2 + 0.45 * len(cases)))
    for i, c in enumerate(cases):
        y = -i
        err = max(c["rel_err"], 1e-12)
        ok = c.get("pass")
        ax.plot(err, y, "o", color=p.good if ok else p.critical, markersize=7, markeredgecolor=p.surface, markeredgewidth=1.5, zorder=3)
        if c.get("tolerance"):
            ax.plot(c["tolerance"], y, "|", color=p.ink2, markersize=14, markeredgewidth=1.5, zorder=2)
        ax.text(err * 1.4, y, f"{c['rel_err']:.1e}  {'pass' if ok else 'FAIL' if ok is False else ''}", va="center", fontsize=8.5, color=p.ink)
    ax.set_yticks([-i for i in range(len(cases))])
    ax.set_yticklabels([c["name"] for c in cases])
    ax.set_xscale("log")
    ax.set_xlabel("relative error (max |dspk − reference| / max |reference|; tick: tolerance)")
    ax.grid(axis="y", visible=False)
    lo, hi = ax.get_xlim()
    ax.set_xlim(lo, hi * 30)
    return fig, ax


__all__ = ["DEFAULT_NPERSEG", "agreement", "before_after", "overlay", "psd", "raster", "response", "spectrum", "timings", "traces", "waveforms"]
