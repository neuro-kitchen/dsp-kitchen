# %% [markdown]
# # dspk: pipelines as data (JSON or TOML)
# A pipeline spec names a recording, the stages and the output. `dspk.pipeline.run` streams the
# recording window by window through the device and writes the canonical spec next to the output
# (`<output>.pipeline.json`), so every result records how it was made.

# %% A synthetic recording on disk
import json
from pathlib import Path

import numpy as np

import dspk

OUT = Path(__file__).resolve().parents[1] / "output" / "pipeline_specs"
OUT.mkdir(parents=True, exist_ok=True)
FS = 30_000.0
rng = np.random.default_rng(2)
t = np.arange(150_000) / FS
x = np.stack([20 * np.sin(2 * np.pi * 60 * t) + 5 * np.sin(2 * np.pi * 900 * t) + rng.standard_normal(t.size) for _ in range(8)])
dspk.io.write_raw(x.astype(np.float32), OUT / "recording.bin", FS, format="int16", gain=0.01, unit="µV")
print(dspk.io.open(OUT / "recording.bin"))

# %% A spec as a dict (JSON)
spec = {
    "dspk_pipeline": 1,
    "name": "notch + highpass + CAR",
    "nodes": {
        "raw": {"op": "read", "path": "recording.bin"},
        "notch": {"op": "notch", "freq": 60.0, "q": 30.0, "mode": "forward"},
        "hp": {"op": "iir", "band": "highpass", "cutoff": [300.0], "mode": "forward"},
        "car": {"op": "car"},
        "out": {"op": "write", "path": "filtered.zarr", "format": "zarr"},
    },
    "edges": [["raw", "notch"], ["notch", "hp"], ["hp", "car"], ["car", "out"]],
}
print(json.dumps(dspk.pipeline.canonical(spec)["nodes"]["hp"], indent=1))  # defaults written out
report = dspk.pipeline.run(spec, base_dir=OUT)
print({k: report[k] for k in ("mode", "runtime", "windows", "seconds")})
print(dspk.io.open(report["output"]))

# %% The same pipeline as a TOML file (comments allowed)
(OUT / "filter.toml").write_text(
    """
dspk_pipeline = 1
name = "notch + highpass + CAR"
edges = [["raw", "notch"], ["notch", "hp"], ["hp", "car"], ["car", "out"]]

[nodes.raw]
op = "read"
path = "recording.bin"

[nodes.notch]
op = "notch"        # mains hum
freq = 60.0
q = 30.0
mode = "forward"

[nodes.hp]
op = "iir"
band = "highpass"
cutoff = [300.0]
mode = "forward"    # causal: the recording streams with filter state carried

[nodes.car]
op = "car"

[nodes.out]
op = "write"
path = "filtered_from_toml.zarr"
format = "zarr"
"""
)
report_toml = dspk.pipeline.run(OUT / "filter.toml")
same = np.array_equal(dspk.io.open(report["output"]).read(), dspk.io.open(report_toml["output"]).read())
print("TOML run equals the JSON run:", same)

# %% Provenance: the spec that made a result, next to it, with the runtime that actually ran
saved = dspk.pipeline.load(report["provenance"])
print("provenance:", report["provenance"])
print("runtime recorded:", saved["execution"]["runtime"])
saved["execution"].pop("runtime")
print("otherwise the spec as written:", saved == dspk.pipeline.canonical(spec))
