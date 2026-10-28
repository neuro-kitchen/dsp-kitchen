# %% [markdown]
# # dspk: stages chained on the device, and streaming
# A `dspk.ops.Pipeline` runs its stages on the device: one upload, every stage, one download per
# chunk; filter designs and device buffers are kept between chunks. With `stream=True`, filter
# state carries from chunk to chunk, so a signal fed in pieces gives exactly the result of the whole
# signal at once (forward filters).

# %% Imports and a synthetic recording
import time

import numpy as np
from scipy import signal as sp

import dspk
from dspk.ops import Pipeline, stage

FS = 30_000.0
rng = np.random.default_rng(1)
t = np.arange(300_000) / FS  # 10 s
common = 30 * np.sin(2 * np.pi * 50 * t)  # shared by every channel: the CAR stage removes it
x = np.stack([common + 5 * np.sin(2 * np.pi * (300 + 100 * c) * t) + rng.standard_normal(t.size) for c in range(32)]).astype(np.float32)

stages = [stage.scale(0.195), stage.car(), stage.iir("bandpass", (300.0, 6_000.0), mode="forward")]

# %% The same chain with NumPy + scipy, for reference
y = 0.195 * x.astype(np.float64)
y -= y.mean(axis=0, keepdims=True)
reference = sp.sosfilt(sp.butter(5, (300.0, 6_000.0), "bandpass", fs=FS, output="sos"), y, axis=1)
peak = np.max(np.abs(reference))

# %% Stream in uneven chunks on every runtime: equal to the whole signal
# Timings depend on the build: `maturin develop --release` for real numbers (`--profile validate`
# optimizes the dependencies but not dspk itself).
edges = [0, 12_345, 12_346, 100_000, 222_222, x.shape[1]]
for runtime in dspk.runtime.available():
    try:
        pipe = Pipeline(stages, FS, stream=True, runtime=runtime)
        pipe.process(x[:, :1_000])  # first call compiles and tunes (cached on disk for later runs)
        pipe.reset()
        start = time.perf_counter()
        out = np.concatenate([pipe.process(x[:, a:b]) for a, b in zip(edges, edges[1:])], axis=1)
        seconds = time.perf_counter() - start
    except RuntimeError as e:
        print(f"{runtime}: skipped ({e})")
        continue
    err = np.max(np.abs(out - reference)) / peak
    rate = x.size / seconds / 1e6
    print(f"{runtime:5s} max error {err:.1e} of the peak | {seconds * 1e3:7.1f} ms for 32 ch × 10 s ({rate:.0f} M samples/s)")

# %% Independent chunks (stream=False) need context: `settling` says how much
pipe = Pipeline([stage.iir("bandpass", (300.0, 6_000.0))], FS)  # zero-phase
print("settling (left, right) samples:", pipe.settling)
