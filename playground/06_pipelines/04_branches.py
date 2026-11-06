# %% [markdown]
# # 06 · Pipelines — 4. Branches: one read, several outputs
# A spec can branch: one `read`, then paths to several `write`s. The recording is read once, the
# operations every branch starts with run once (on the device, shared), and each branch then runs
# only its own. Here: the spike band and the LFP band from one pass over the file.

# %%
from pathlib import Path
import tempfile

import dspk
import plating
from plating import figures, signals

FIG = Path(__file__).parent / "figures"
work = Path(tempfile.mkdtemp(prefix="dspk-playground-"))
s = signals.synthetic()
dspk.io.write_zarr(s.data, work / "recording.zarr", s.sample_rate, unit="µV")

spec = {
    "dspk_pipeline": 1,
    "name": "spikes and LFP from one read",
    "nodes": {
        "raw": {"op": "read", "path": "recording.zarr"},
        "notch": {"op": "notch", "freq": 60.0, "q": 30.0},  # shared by both branches: computed once
        "spikes_band": {"op": "iir", "band": "bandpass", "cutoff": [300.0, 6000.0]},
        "cmr": {"op": "cmr"},
        "lfp_band": {"op": "iir", "band": "lowpass", "cutoff": [300.0]},
        "spikes": {"op": "write", "path": "spikes.zarr", "format": "zarr"},
        "lfp": {"op": "write", "path": "lfp.zarr", "format": "zarr"},
    },
    "edges": [["raw", "notch"], ["notch", "spikes_band"], ["spikes_band", "cmr"], ["cmr", "spikes"], ["notch", "lfp_band"], ["lfp_band", "lfp"]],
}
reports = dspk.pipeline.run(spec, work)
for r in reports:
    print(r["output"].name, {k: r[k] for k in ("windows", "seconds")})
spikes, lfp = dspk.io.open(work / "spikes.zarr").read(), dspk.io.open(work / "lfp.zarr").read()

# %% Figure: the two outputs of one pass
p = plating.theme.palette()
fig, (left, right) = plating.figure("One read, two outputs: the spike band and the LFP", cols=2, size=(11.0, 5.4))
figures.traces(left, spikes, s.sample_rate, duration=0.5, color=p.series[0])
left.set_title("spikes.zarr: notch → band-pass → CMR")
figures.traces(right, lfp, s.sample_rate, duration=0.5, color=p.series[1])
right.set_title("lfp.zarr: notch → low-pass 300 Hz")
plating.save(fig, FIG / "04_branches.png", note="A branched spec: the notch (shared) runs once per window; each branch runs its own stages.")
