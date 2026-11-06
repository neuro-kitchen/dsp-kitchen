# %% [markdown]
# # 06 · Pipelines — 2. Streaming: pieces equal the whole
# A live recording arrives in pieces. With `Pipeline(..., stream=True)` each filter carries its
# state from one piece to the next, so the pieces, of any lengths, give exactly the result of the
# whole signal at once. Streaming needs causal filters (`mode="forward"`): a zero-phase filter would
# need samples that have not arrived yet.

# %%
from pathlib import Path

import numpy as np

import plating
from dspk.ops import Pipeline, stage
from plating import signals

FIG = Path(__file__).parent / "figures"
s = signals.synthetic()
stages = [stage.notch(60.0, 30.0, mode="forward"), stage.iir("bandpass", (300.0, 6_000.0), mode="forward"), stage.cmr()]

whole = Pipeline(stages, s.sample_rate, stream=True).process(s.data)
edges = [0, 1_234, 1_235, 20_000, 33_333, 47_000, s.samples]  # uneven, one of a single sample
stream = Pipeline(stages, s.sample_rate, stream=True)
pieces = np.concatenate([stream.process(s.data[:, a:b]) for a, b in zip(edges, edges[1:])], axis=1)
err = np.max(np.abs(pieces - whole)) / np.max(np.abs(whole))
print(f"{len(edges) - 1} pieces vs the whole signal: max difference {err:.1e} of the peak")

# %% Figure: around a piece boundary, the streamed result over the whole-signal one
p = plating.theme.palette()
ch, at = s.loudest_channel, edges[3]
win = slice(at - 600, at + 600)
t = (np.arange(win.start, win.stop) - at) / s.sample_rate * 1e3
fig, (top, bottom) = plating.figure("Streamed in pieces, equal to the whole", rows=2, size=(11.0, 5.6), sharex=True, height_ratios=[2, 1])
top.plot(t, whole[ch, win], color=p.muted, linewidth=2.6, label="the whole signal at once")
top.plot(t, pieces[ch, win], color=p.series[0], linewidth=1.0, label=f"{len(edges) - 1} pieces of uneven length")
top.axvline(0, color=p.ink2, linewidth=0.8, linestyle=(0, (4, 3)))
top.set_ylabel("µV")
top.set_title(f"channel {ch} around a piece boundary (dashed)")
top.legend(loc="lower left", ncols=2)
bottom.plot(t, (pieces[ch, win] - whole[ch, win]) * 1e3, color=p.series[0], linewidth=1.0)
bottom.set_ylabel("nV")
bottom.set_xlabel("time from the boundary (ms)")
bottom.set_title(f"difference: at most {err:.0e} of the peak (float32 rounding)")
plating.save(fig, FIG / "02_streaming.png", note="Pipeline(stream=True): notch 60 Hz, band-pass 300–6000 Hz (both forward), CMR.")
