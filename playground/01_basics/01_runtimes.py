# %% [markdown]
# # 01 · Basics — 1. Runtimes
# dspk runs every operation on a *runtime*: `cuda` (NVIDIA), `wgpu` (Vulkan / Metal / DirectX, any
# GPU) or `cpu`. The same call gives the same result on each; only the time differs. Every function
# takes `runtime=`; without it, the one `dspk.runtime.set` chose (default: the first available).

# %% Which runtimes this build has, and which one is in use
from pathlib import Path
import time

import numpy as np

import dspk
import plating
from plating import figures, signals

FIG = Path(__file__).parent / "figures"
print("available:", dspk.runtime.available())
print("in use:   ", dspk.runtime.current())

# %% The same operation on every runtime: identical results, different times
s = signals.synthetic(channels=64, seconds=2.0)
results, times = {}, {}
for rt in dspk.runtime.available():
    try:
        dspk.ops.spatial.car(s.data, runtime=rt)  # the first call compiles the kernels (cached on disk)
    except RuntimeError as e:  # compiled in, but no such device on this machine
        print(f"{rt}: not usable here ({e})")
        continue
    start = time.perf_counter()
    results[rt] = dspk.ops.spatial.car(s.data, runtime=rt)
    times[rt] = (time.perf_counter() - start) * 1e3
first = next(iter(results.values()))
for rt, out in results.items():
    print(f"{rt:5s} {times[rt]:7.1f} ms   max difference from {next(iter(results))}: {np.max(np.abs(out - first)):.1e}")

# %% Figure: time per runtime for one call (upload, kernel, download)
fig, ax = plating.figure("One operation, every runtime", size=(9.0, 1.4 + 0.5 * len(times)))
p = plating.theme.palette()
names = list(times)
ax.barh(range(len(names)), [times[n] for n in names], height=0.5, color=p.series[0])
for i, n in enumerate(names):
    ax.text(times[n] * 1.02, i, figures.ms_label(times[n]), va="center", fontsize=8.5, color=p.ink)
ax.set_yticks(range(len(names)))
ax.set_yticklabels(names)
ax.invert_yaxis()
ax.set_xlim(0, max(times.values()) * 1.25)
ax.grid(axis="y", visible=False)
ax.set_xlabel("milliseconds")
ax.set_title(f"common average reference of {s.channels} channels × {s.samples:,} samples, same result on each")
plating.save(fig, FIG / "01_runtimes.png", note="Time of one call from a NumPy array: upload, kernel, download (after a warm-up call).")
