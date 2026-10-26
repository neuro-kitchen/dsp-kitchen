# %% [markdown]
# # dsp-kitchen benchmark: table of sorter time and memory
# Draws the reports of `sorter_resources.py` (Kilosort4 and EMUsort) as one image: recording, wall
# time, speed against real time, peak GPU memory and host RAM, spikes and units, reproducibility,
# and the warm run's time per stage on one shared scale.
#
#     uv run python playground/benchmarks/sorter_resources_table.py
#
# Reads `playground/output/sorter_resources_{ks4,emusort}.json`, writes
# `playground/output/sorter_resources.png`.

# %% [1] Reports
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUTPUT = Path(__file__).resolve().parents[1] / "output"
COLUMNS = [("ks4", "Kilosort4"), ("emusort", "EMUsort")]
reports = [json.loads((OUTPUT / f"sorter_resources_{key}.json").read_text()) for key, _ in COLUMNS]

# Light surface, text inks and one sequential hue (dataviz reference palette)
SURFACE, PRIMARY, SECONDARY, MUTED, RULE = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
BAR, BAR_TRACK = "#2a78d6", "#eceae6"
SANS, MONO = "Open Sans", "Source Code Pro"
DPI = 300


def duration(sec):
    return f"{sec:.0f} s" if sec < 120 else f"{sec / 60:.1f} min"


def gib(mib):
    return "n/a" if mib is None else f"{mib / 1024:.2f} GiB"


def peak(report, key):
    values = [run[key] for run in report["runs"] if run[key] is not None]
    return max(values) if values else None


def wall(report):
    cold, *warm = report["runs"]
    if not warm:
        return f"{cold['wall_sec']:.1f} s cold"
    return f"{warm[-1]['wall_sec']:.1f} s warm ({cold['wall_sec']:.1f} s cold)"


def last(report):
    return report["runs"][-1]


ROWS = [
    ("Recording", lambda r: r["source"], SANS),
    ("Channels × sample rate", lambda r: f"{r['channels']} × {r['sample_rate'] / 1e3:.3g} kHz", MONO),
    ("Duration sorted", lambda r: duration(r["duration_sec"]), MONO),
    ("Signal size (float32)", lambda r: f"{r['signal_f32_gib']:.2f} GiB", MONO),
    None,
    ("Wall time", wall, MONO),
    ("Speed vs real time", lambda r: f"{r['duration_sec'] / last(r)['wall_sec']:.1f}× faster", MONO),
    ("Peak GPU memory", lambda r: gib(peak(r, "peak_gpu_mib")), MONO),
    ("Peak host RAM (process)", lambda r: gib(r["process_max_rss_mib"]), MONO),
    None,
    ("Spikes / units", lambda r: f"{last(r)['spikes']:,} / {last(r)['units']}", MONO),
    ("Rerun", lambda r: "—" if len(r["runs"]) < 2 else "identical" if r["reproducible"] else "differs", SANS),
]
stages = list(dict.fromkeys(s for r in reports for s in last(r)["stages_sec"]))

# %% [2] Draw (figure units: inches, y down)
W, H = 12.0, 13.7
fig = plt.figure(figsize=(W, H), dpi=DPI, facecolor=SURFACE)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(H, 0)
ax.axis("off")
LEFT, RIGHT = 0.6, W - 0.6
X_COLUMNS = [6.0, 9.1]
BAR_WIDTH, BAR_HEIGHT = 2.1, 0.2


def text(x, y, s, size=13, color=PRIMARY, weight="normal", family=SANS):
    ax.text(x, y, s, fontsize=size, color=color, fontweight=weight, family=family, va="center")


def rule(y, color=RULE, lw=1.0):
    ax.plot([LEFT, RIGHT], [y, y], color=color, lw=lw)


def bar(x, y, width, color):
    ax.add_patch(FancyBboxPatch((x, y - BAR_HEIGHT / 2), width, BAR_HEIGHT, fc=color, ec="none",
                                boxstyle="round,pad=0,rounding_size=0.05"))


y = 0.65
text(LEFT, y, "dsp-kitchen · early benchmarks", 24, weight="bold")
y += 0.5
text(LEFT, y, "GPU spike sorting in Rust (CubeCL) · whole recording, raw file to sorted units", 13, SECONDARY)
y += 0.5
rule(y, PRIMARY, 1.2)
y += 0.42
for x, (_, title) in zip(X_COLUMNS, COLUMNS):
    text(x, y, title, 15, weight="bold")
y += 0.25
rule(y)
for row in ROWS:
    if row is None:
        y += 0.24
        rule(y)
        y -= 0.08
        continue
    label, value, family = row
    y += 0.44
    text(LEFT, y, label, 13, SECONDARY)
    for x, report in zip(X_COLUMNS, reports):
        text(x, y, value(report), 13, family=family)
y += 0.3
rule(y, PRIMARY, 1.2)

# Time per stage: bar length = seconds, scaled to each sorter's longest stage (the runs differ ~10×)
y += 0.6
text(LEFT, y, "Where the time goes (last run, seconds)", 15, weight="bold")
y += 0.2
longest = [max(last(r)["stages_sec"].values()) for r in reports]
for stage in stages:
    y += 0.4
    text(LEFT, y, stage, 12.5, SECONDARY)
    for x, report, scale in zip(X_COLUMNS, reports, longest):
        sec = last(report)["stages_sec"].get(stage, 0.0)
        bar(x, y, BAR_WIDTH, BAR_TRACK)
        bar(x, y, max(BAR_WIDTH * sec / scale, 0.05), BAR)
        text(x + BAR_WIDTH + 0.12, y, f"{sec:.1f}", 12, family=MONO)

y += 0.55
rule(y)
gpu = (reports[0]["gpu"] or "no NVIDIA GPU reading").replace("NVIDIA GeForce ", "NVIDIA ")
y += 0.35
text(LEFT, y, f"{gpu} · {reports[0]['cpu']} · {reports[0]['runtime']} · f32 · release build", 10.5, MUTED)
y += 0.3
text(LEFT, y, "Cold = first run, kernels compile. GPU memory: nvidia-smi peak above idle. "
              "Bars: each column on its own scale.", 10.5, MUTED)
y += 0.3
text(LEFT, y, "Rerun \u2014: one run only. Early, unreleased: no drift correction yet.", 10.5, MUTED)
y += 0.3
text(LEFT, y, "github.com/neuro-kitchen/dsp-kitchen", 10.5, MUTED)

path = OUTPUT / "sorter_resources.png"
fig.savefig(path, facecolor=SURFACE)
print(f"saved {path}")
