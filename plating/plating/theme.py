"""The look: a validated categorical palette (fixed order), ink and chrome tokens for a light and a
dark surface, fonts, and the figure frame (left-aligned bold title, a subtitle per panel, a footer
naming the machine and version).

Colours follow the job they do: series (identity) take the categorical slots in order, never
cycled past what the figure needs; "before" / reference data is drawn in the muted ink, the result
in slot 1; status colours are kept for pass / fail.
"""

from __future__ import annotations

import platform
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

#: Figure width presets (inches) and the resolution figures are saved at.
WIDE = (11.0, 4.6)
TALL = (11.0, 7.2)
SQUARE = (6.4, 5.6)
DPI = 200
#: Height of one footer line (inches).
FOOTER_LINE_IN = 0.2
#: Fonts (installed with the repository's dev setup; matplotlib falls back to DejaVu otherwise).
SANS = ["Open Sans", "DejaVu Sans"]
MONO = ["Source Code Pro", "DejaVu Sans Mono"]


@dataclass(frozen=True)
class Palette:
    surface: str
    page: str
    ink: str
    ink2: str
    muted: str
    grid: str
    baseline: str
    #: Categorical slots in their fixed (CVD-validated) order.
    series: tuple[str, ...]
    #: Sequential ramp (one hue, near-zero → large).
    sequential: tuple[str, ...]
    good: str = "#0ca30c"
    warning: str = "#fab219"
    critical: str = "#d03b3b"


LIGHT = Palette(
    surface="#fcfcfb", page="#f9f9f7", ink="#0b0b0b", ink2="#52514e", muted="#898781",
    grid="#e1e0d9", baseline="#c3c2b7",
    series=("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"),
    sequential=("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"),
)
DARK = Palette(
    surface="#1a1a19", page="#0d0d0d", ink="#f0efec", ink2="#c3c2b7", muted="#898781",
    grid="#2c2c2a", baseline="#383835",
    series=("#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"),
    sequential=("#0d366b", "#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4", "#cde2fb"),
)

_current: Palette = LIGHT


def palette() -> Palette:
    """The palette in use (see :func:`use`)."""
    return _current


def use(mode: str = "light") -> Palette:
    """Applies the theme (``"light"`` or ``"dark"``) to matplotlib; returns its palette."""
    global _current
    if mode not in ("light", "dark"):
        raise ValueError(f"mode is 'light' or 'dark', got {mode!r}")
    p = _current = LIGHT if mode == "light" else DARK
    plt.rcParams.update({
        "figure.facecolor": p.surface, "savefig.facecolor": p.surface, "axes.facecolor": p.surface,
        "font.family": "sans-serif", "font.sans-serif": SANS, "font.monospace": MONO, "font.size": 9.5,
        "text.color": p.ink, "axes.labelcolor": p.ink2, "axes.labelsize": 9.5,
        "axes.titlesize": 10.5, "axes.titlelocation": "left", "axes.titlecolor": p.ink, "axes.titlepad": 8,
        "axes.edgecolor": p.baseline, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": p.grid, "grid.linewidth": 0.8, "axes.axisbelow": True,
        "axes.prop_cycle": plt.cycler(color=list(p.series)),
        "xtick.color": p.ink2, "ytick.color": p.ink2, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
        "xtick.major.size": 3, "ytick.major.size": 3,
        "legend.frameon": False, "legend.fontsize": 8.5, "legend.labelcolor": p.ink,
        "lines.linewidth": 1.2, "lines.solid_capstyle": "round",
        "savefig.dpi": DPI, "figure.dpi": 100,
    })
    return p


def figure(title: str, *, rows: int = 1, cols: int = 1, size: tuple[float, float] = WIDE, sharex: bool = False, sharey: bool = False, height_ratios=None):
    """A figure with a bold left-aligned ``title``; returns ``(fig, axes)`` (one axes, or an array).
    Give each panel its own subtitle with ``ax.set_title``."""
    if plt.rcParams["axes.facecolor"] != _current.surface:
        use("light" if _current is LIGHT else "dark")
    fig, axes = plt.subplots(rows, cols, figsize=size, sharex=sharex, sharey=sharey, squeeze=True,
                             gridspec_kw={"height_ratios": height_ratios} if height_ratios else None)
    # tight_layout (in save) leaves room for this title by itself
    fig.suptitle(title, x=0.012, y=1 - 0.1 / size[1], ha="left", va="top", fontsize=13, fontweight="bold", color=_current.ink)
    return fig, axes


def machine() -> str:
    """``"<GPU> · <CPU> (<n> cores) · <OS>"`` for figure footers and benchmark records."""
    import os
    import shutil
    import subprocess

    gpu = ""
    if shutil.which("nvidia-smi"):
        try:
            gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True, timeout=5).stdout.strip().splitlines()[0]
        except (OSError, subprocess.SubprocessError, IndexError):
            gpu = ""
    cpu = platform.processor() or platform.machine()
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    parts = [gpu, f"{cpu} ({os.cpu_count()} threads)", f"{platform.system()} {platform.release()}"]
    return " · ".join(p for p in parts if p)


def _versions() -> str:
    try:
        import dspk

        return f"dspk {dspk.__version__} · runtime {dspk.runtime.current()}"
    except Exception:  # noqa: BLE001 (plots work without dspk)
        return ""


def reference_line(ax, y: float, label: str, *, horizontal: bool = True) -> None:
    """A dashed reference line (real time, a tolerance, a cutoff) with its label in a small box."""
    p = _current
    line = ax.axhline if horizontal else ax.axvline
    line(y, color=p.ink2, linewidth=1, linestyle=(0, (4, 3)), zorder=2)
    box = dict(boxstyle="round,pad=0.25", facecolor=p.surface, edgecolor=p.grid, linewidth=0.8)
    if horizontal:
        ax.text(0.5, y, label, transform=ax.get_yaxis_transform(), color=p.ink2, fontsize=8.5, ha="center", va="center", bbox=box, zorder=5)
    else:
        ax.text(y, 0.95, label, transform=ax.get_xaxis_transform(), color=p.ink2, fontsize=8.5, ha="center", va="top", bbox=box, zorder=5)


def save(fig: Figure, path: str | Path, *, note: str = "", footer: bool = True) -> Path:
    """Saves ``fig`` as a PNG at :data:`DPI`, with a footer: ``note`` (what was done, in one line),
    then the machine and the dspk version / runtime. Creates the folder; returns the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if footer:
        lines = [note] if note else []
        stamp = " · ".join(s for s in (machine(), _versions()) if s)
        if stamp:
            lines.append(stamp)
        if lines:
            fig.text(0.012, 0.004, "\n".join(lines), fontsize=8, color=_current.ink2, ha="left", va="bottom")
    h = fig.get_figheight()
    lines_below = (bool(note) + 1) if footer else 0
    fig.tight_layout(rect=(0, (0.06 + FOOTER_LINE_IN * lines_below) / h, 1, 1), h_pad=1.2, w_pad=2.0)
    fig.savefig(path, facecolor=_current.surface)
    plt.close(fig)
    return path


use("light")
