# %% [markdown]
# # 07 · Channels — 1. Recordings saved out of order
# Acquisition systems store channels in the order of their amplifier inputs, not the order of the
# electrodes, so neighbouring electrodes end up anywhere in the file. A **channel map** puts them
# back, written as pairs (like a hardware channel mapper): "stored channel `from` is electrode
# `to`", with an optional name. Inputs left out are dropped; `base = 1` reads 1-based numbers
# (TDT, MATLAB). Maps are TOML or JSON files (`[remap]`), and the same file works in Python, in
# pipeline specs, in the CLI (`--channel-map`) and in the app.

# %%
from pathlib import Path
import tempfile

import numpy as np

import dspk
import plating
from dspk.ops import layout
from plating import figures, signals

FIG = Path(__file__).parent / "figures"
work = Path(tempfile.mkdtemp(prefix="dspk-playground-"))
s = signals.synthetic(channels=16, lfp_uv=20.0, spread_channels=1.5)
spikes_band = dspk.ops.filter.iir(s.data, s.sample_rate, "highpass", 300.0)

# The amplifier wired electrode e to input wiring[e]: the file stores them in input order
rng = np.random.default_rng(4)
wiring = rng.permutation(s.channels)
stored = np.empty_like(spikes_band)
stored[wiring] = spikes_band

# The map: stored input wiring[e] is electrode e (1-based here, as a TDT export would be)
pairs = [{"from": int(wiring[e]) + 1, "to": e + 1, "name": f"E{e + 1:02d}"} for e in range(s.channels)]
layout.write_map({"base": 1, "pairs": pairs}, work / "map.toml")
print((work / "map.toml").read_text()[:200], "…")

ordered = layout.remap(stored, work / "map.toml")
order, names = layout.map_order(work / "map.toml")
print("back in electrode order:", np.array_equal(ordered, spikes_band))

# %% Figure: the stored order and the electrode order (spikes span neighbouring electrodes)
p = plating.theme.palette()
fig, (left, right) = plating.figure("A channel map puts the electrodes back in order", cols=2, size=(11.0, 7.0))
spacing = figures.traces(left, stored, s.sample_rate, start=0.0, duration=0.15, color=p.muted, names=[f"in {i + 1}" for i in range(s.channels)])
left.set_title("as stored (amplifier inputs): a spike lands on scattered rows")
figures.traces(right, ordered, s.sample_rate, start=0.0, duration=0.15, spacing=spacing, names=names)
right.set_title("after layout.remap(data, 'map.toml'): neighbours together")
right.set_ylim(left.get_ylim())  # the same scale (not sharey: each side keeps its own row names)
plating.save(fig, FIG / "01_remap_with_pairs.png", note="dspk.ops.layout.remap with a 1-based map of pairs (from = input, to = electrode, name).")
