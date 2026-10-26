# `dsp_kitchen_py` — Python SDK and native bindings

- **`src/`**: the PyO3 native module `dsp_kitchen_bindings` (flat), one Rust module per SDK area:
  `runtime`, `buffer` (io), `math`, `filter/{iir, fir, non_linear, template}`, `spatial`,
  `linalg`, `pipeline`, `synapse/{probe, detection, extraction, spatial, sorting, metrics,
  comparison, storage, streaming, ml}`.
- **`dsp_kitchen/`**: the SDK (`import dsp_kitchen as dk`), which arranges the native names:
  `dk.runtime`, `dk.filter.{iir, fir, non_linear, template}`, `dk.spatial`, `dk.math`,
  `dk.linalg`, `dk.pipeline`, `dk.io`, `dk.synapse` (`dk.synapse.ml.{kilosort4, emusort}`,
  `ModelHub`).
- **`tests/`**: SDK tests against NumPy (and scipy when installed); binding soundness (zero-copy
  views, GIL release, errors).

```python
import dsp_kitchen as dk

dk.runtime.available()                     # e.g. ['wgpu', 'cpu']
rec = dk.io.Recording("session.ap.bin")    # any format dsp-io reads
x = rec.read(0, 30_000)                    # [channels, samples], each channel's unit
y = dk.filter.iir.bandpass_filter(x, 300, 6_000, fs=rec.sample_rate)

pipe = dk.pipeline.Pipeline([dk.filter.iir.HighpassFilter(300), dk.spatial.CommonAverageReference()])
probe = dk.synapse.ProbeLayout.from_recording("session.ap.bin")
result = dk.synapse.detect_recording(rec, pipe, probe)   # whole recording, in Rust, bounded memory

print(dk.synapse.ml.kilosort4.provenance().citation())
```

Conventions: `fs` (Hz) is required wherever time matters; defaults are the Rust crates' (scipy,
scikit-learn, SpikeInterface conventions), never invented; values are in each recording's unit;
device calls take `runtime=` (else `dk.runtime.set(...)`, `DSP_KITCHEN_RUNTIME`, or the first
compiled-in runtime).

Build: `uv pip install maturin && maturin develop` (features: `wgpu` + `hub` by default; `cpu`,
`cuda`, `hip`).
