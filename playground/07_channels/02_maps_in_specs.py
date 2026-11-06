# %% [markdown]
# # 07 · Channels — 2. A map in a pipeline
# In a spec, a `remap` node takes the same map: `{"op": "remap", "path": "map.toml"}` or the
# `pairs` written in. Right after `read` it costs nothing (the reader reads the channels in the new
# order) and may drop channels; between processing nodes it reorders on the device; right before
# `write` it may drop channels again. The output file keeps the map's channel names.

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
s = signals.synthetic(channels=16)
rng = np.random.default_rng(4)
wiring = rng.permutation(s.channels)
stored = np.empty_like(s.data)
stored[wiring] = s.data
DEAD = 5  # electrode 5 is broken: the map leaves it out
stored[wiring[DEAD]] = rng.normal(0, 400, s.samples)
dspk.io.write_raw(stored, work / "recording.bin", s.sample_rate, unit="µV")
pairs = [{"from": int(wiring[e]), "to": i, "name": f"E{e:02d}"} for i, e in enumerate(e for e in range(s.channels) if e != DEAD)]
layout.write_map({"base": 0, "pairs": pairs}, work / "map.toml")

spec = {
    "dspk_pipeline": 1,
    "nodes": {
        "raw": {"op": "read", "path": "recording.bin"},
        "map": {"op": "remap", "path": "map.toml"},
        "band": {"op": "iir", "band": "bandpass", "cutoff": [300.0, 6000.0]},
        "cmr": {"op": "cmr"},
        "out": {"op": "write", "path": "ordered.zarr", "format": "zarr"},
    },
    "edges": [["raw", "map"], ["map", "band"], ["band", "cmr"], ["cmr", "out"]],
}
(plan,) = dspk.pipeline.plan(spec, work)
print(f"reads {plan['channels']} channels, writes {plan['output_channels']}")
(report,) = dspk.pipeline.run(spec, work)
out = dspk.io.open(report["output"])
print(out.channel_names)

# %% Figure: the written file, in electrode order, named, without the dead electrode
fig, ax = plating.figure("read → remap → band-pass → CMR → write: ordered, named, the dead electrode out", size=(11.0, 6.4))
figures.traces(ax, out.read(), s.sample_rate, duration=0.2, names=out.channel_names)
ax.set_title(f"ordered.zarr: {len(out.channel_names)} of {s.channels} electrodes (E{DEAD:02d} dropped by the map); the names come from the map")
plating.save(fig, FIG / "02_maps_in_specs.png", note="A remap node right after read (folded into the read); channel names kept in the Zarr output.")
