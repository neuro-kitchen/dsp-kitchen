# %% [markdown]
# # 06 · Pipelines — 3. Pipelines as data (JSON or TOML)
# A *spec* names a recording, the operations and the output, as nodes and edges. `dspk.pipeline.run`
# streams the recording window by window through the device and writes the output, plus the spec
# itself next to it (`<output>.pipeline.json`, every default written out and the runtime that ran),
# so every result says how it was made. The same spec runs from Python, from the CLI
# (`dspk run spec.toml`) and in the app's Pipeline workspace.

# %%
from pathlib import Path
import json
import tempfile

import numpy as np

import dspk
import plating
from plating import figures, signals

FIG = Path(__file__).parent / "figures"
work = Path(tempfile.mkdtemp(prefix="dspk-playground-"))
s = signals.synthetic()
dspk.io.write_raw(s.data, work / "recording.bin", s.sample_rate, format="int16", gain=0.195, unit="µV")

spec = {
    "dspk_pipeline": 1,
    "name": "notch + band-pass + CMR",
    "nodes": {
        "raw": {"op": "read", "path": "recording.bin"},
        "notch": {"op": "notch", "freq": 60.0, "q": 30.0},
        "band": {"op": "iir", "band": "bandpass", "cutoff": [300.0, 6000.0]},
        "cmr": {"op": "cmr"},
        "out": {"op": "write", "path": "filtered.zarr", "format": "zarr"},
    },
    "edges": [["raw", "notch"], ["notch", "band"], ["band", "cmr"], ["cmr", "out"]],
}
print(json.dumps(dspk.pipeline.canonical(spec)["nodes"]["band"]))  # every default written out
(plan,) = dspk.pipeline.plan(spec, work)  # what it would do, without running
print({k: plan[k] for k in ("mode", "halo", "windows", "stored_upload")})
(report,) = dspk.pipeline.run(spec, work)
print({k: report[k] for k in ("mode", "runtime", "windows", "seconds")})

# %% The same spec as TOML (comments allowed) gives the same file
(work / "spec.toml").write_text('''
dspk_pipeline = 1
name = "notch + band-pass + CMR"
edges = [["raw", "notch"], ["notch", "band"], ["band", "cmr"], ["cmr", "out"]]

[nodes.raw]
op = "read"
path = "recording.bin"

[nodes.notch]
op = "notch"   # mains hum
freq = 60.0
q = 30.0

[nodes.band]
op = "iir"
band = "bandpass"
cutoff = [300.0, 6000.0]

[nodes.cmr]
op = "cmr"

[nodes.out]
op = "write"
path = "from_toml.zarr"
format = "zarr"
''')
(report_toml,) = dspk.pipeline.run(work / "spec.toml")
out = dspk.io.open(report["output"]).read()
print("TOML equals JSON:", np.array_equal(out, dspk.io.open(report_toml["output"]).read()))
print("provenance:", report["provenance"])

# %% Figure: the recording and the pipeline's output file
fig, _ = figures.before_after(dspk.io.open(work / "recording.bin").read(), out, s.sample_rate, title="A spec's output, read back from its file",
                              labels=("recording.bin (int16)", "filtered.zarr"), duration=0.1)
plating.save(fig, FIG / "03_specs.png", note=f"dspk.pipeline.run: {plan['windows']} window(s), {report['mode']} mode, on {report['runtime']}; the spec saved beside the output.")
