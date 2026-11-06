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

from matplotlib.ticker import FuncFormatter

from plating import theme
from plating.results import read

#: Times within this ratio of each other read "same speed" in :func:`timings`.
SAME_SPEED = 1.05
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


def matrix(ax, m, *, diverging: bool = True, labels: Sequence[str] | None = None, colorbar_label: str = ""):
    """A square matrix as a heatmap: ``diverging`` (blue ↔ red around a grey zero, symmetric
    limits) for signed values such as covariance; else the sequential ramp (one hue, light to dark)
    for magnitudes. Returns the image (for a colour bar)."""
    from matplotlib.colors import LinearSegmentedColormap

    p = theme.palette()
    m = np.asarray(m, dtype=np.float64)
    if diverging:
        mid = "#f0efec" if p is theme.LIGHT else "#383835"
        cmap = LinearSegmentedColormap.from_list("plating-div", [p.series[0], mid, p.series[7]])
        lim = float(np.max(np.abs(m))) or 1.0
        im = ax.imshow(m, cmap=cmap, vmin=-lim, vmax=lim, interpolation="nearest")
    else:
        cmap = LinearSegmentedColormap.from_list("plating-seq", list(p.sequential))
        im = ax.imshow(m, cmap=cmap, interpolation="nearest")
    ax.grid(False)
    n = m.shape[0]
    ticks = range(0, n, max(1, n // 8))
    ax.set_xticks(list(ticks))
    ax.set_yticks(list(ticks))
    if labels is not None:
        ax.set_xticklabels([labels[i] for i in ticks])
        ax.set_yticklabels([labels[i] for i in ticks])
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)
    bar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    bar.outline.set_visible(False)
    bar.ax.tick_params(colors=p.ink2, labelsize=8)
    if colorbar_label:
        bar.set_label(colorbar_label, color=p.ink2)
    return im


def ms_label(ms: float) -> str:
    """A time in milliseconds as people read it: ``0.82 ms``, ``233 ms``, ``2,460 ms``."""
    return f"{ms:,.0f} ms" if ms >= 100 else f"{ms:.3g} ms"


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
    # Bar height, and the offset of a pair's bars (a small gap of surface between them)
    h, off = 0.36, 0.2
    for i, c in enumerate(cases):
        y = -i
        ax.barh(y + (off if ref_name else 0), c["dspk_ms"], height=h, color=p.series[0], label=f"dspk ({r['dspk']['runtime']})" if i == 0 and ref_name else None)
        # The comparison is dspk's: it goes on dspk's bar ("× faster" / "× slower")
        label = ms_label(c["dspk_ms"])
        has_ref = ref_name and c.get("reference_ms") is not None
        if has_ref and c["dspk_ms"]:
            ratio = c["reference_ms"] / c["dspk_ms"]
            if 1 / SAME_SPEED <= ratio <= SAME_SPEED:
                label += "  ·  same speed"
            else:
                label += f"  ·  {ratio:.1f}× faster" if ratio > 1 else f"  ·  {1 / ratio:.1f}× slower"
        ax.text(c["dspk_ms"] * 1.06, y + (off if ref_name else 0), label, va="center", fontsize=8.5, color=p.ink)
        if has_ref:
            ax.barh(y - off, c["reference_ms"], height=h, color=p.series[1], label=ref_name if i == 0 else None)
            ax.text(c["reference_ms"] * 1.06, y - off, ms_label(c["reference_ms"]), va="center", fontsize=8.5, color=p.ink2)
    ax.set_yticks([-i for i in range(len(cases))])
    ax.set_yticklabels([c["name"] for c in cases])
    ax.grid(axis="y", visible=False)
    times = [c["dspk_ms"] for c in cases] + [c["reference_ms"] for c in cases if ref_name and c.get("reference_ms") is not None]
    if ref_name:
        # Times far apart: a log axis, bars from a whole decade below the shortest
        ax.set_xscale("log")
        ax.set_xlim(10 ** np.floor(np.log10(min(times))), max(times) * 8)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.set_xlabel("milliseconds (log)")
        # Above the bars, in one row, so it never covers a value
        ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncols=2, borderaxespad=0.3)
    else:
        # One series: bars from zero on a linear axis, so lengths compare honestly; no legend
        ax.set_xlim(0, max(times) * 1.25)
        ax.set_xlabel("milliseconds")
    return fig, ax


def agreement(record, *, title: str | None = None):
    """Each case's relative error against its tolerance (log scale): a dot per case, its tolerance as
    a tick; passes and failures named in the label, not by colour alone."""
    r = _record(record)
    p = theme.palette()
    cases = [c for c in r["cases"] if c.get("rel_err") is not None]
    fig, ax = theme.figure(title or f"{r['benchmark']}: agreement with {(r.get('reference') or {}).get('tool', 'the reference')}", size=(11.0, 1.2 + 0.45 * len(cases)))
    # The axis: a decade below the smallest non-zero error or tolerance, to well past the largest
    values = [v for c in cases for v in (c["rel_err"], c.get("tolerance")) if v]
    lo = 10 ** (np.floor(np.log10(min(values))) - 1) if values else 1e-9
    hi = 10 ** (np.ceil(np.log10(max(values))) + 1.5) if values else 1.0
    for i, c in enumerate(cases):
        y = -i
        exact = c["rel_err"] == 0
        err = lo if exact else c["rel_err"]
        ok = c.get("pass")
        ax.plot(err, y, "o", color=p.good if ok else p.critical, markersize=7, markeredgecolor=p.surface, markeredgewidth=1.5, zorder=3, clip_on=False)
        tol = c.get("tolerance")
        if tol:
            ax.plot(tol, y, "|", color=p.ink2, markersize=14, markeredgewidth=1.5, zorder=2)
        verdict = "pass" if ok else "FAIL" if ok is False else ""
        text = f"exact  {verdict}" if exact else f"{c['rel_err']:.1e}  {verdict}"
        # Right of the dot, or of the tolerance tick when the dot is left of it and close
        x = err * 1.5 if not tol or err > tol or tol / err > 30 else tol * 1.5
        ax.text(x, y, text, va="center", fontsize=8.5, color=p.ink)
    ax.set_yticks([-i for i in range(len(cases))])
    ax.set_yticklabels([c["name"] for c in cases])
    ax.set_xscale("log")
    ax.set_xlim(lo, hi)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0e}"))
    ax.set_xlabel("relative error (max |dspk − reference| / max |reference|; tick: tolerance; exact matches at the left)")
    ax.grid(axis="y", visible=False)
    return fig, ax


__all__ = ["DEFAULT_NPERSEG", "agreement", "before_after", "matrix", "ms_label", "overlay", "psd", "raster", "response", "spectrum", "timings", "traces", "waveforms"]
