# Open items (index, 2026-10-08)

Everything still open from the cleanup records, in one place. Each line points to where the details
are. **D** = a decision needed from the user. Checked against the code's current state (the book's
*Limitations* sections); items the records list but later work resolved are not repeated here.

Task lists with their own detail: `GPU_TASKS.md` (GPU review; task 8 partly open), `SORTER_TASKS.md`
(everything missing from Kilosort4 and EMUsort).

## Decisions

| Item | Crate | Source |
|---|---|---|
| **D** New CLI commands `detect`, `sort`, `convert` (CLI15) | dsp-cli | `dsp-cli/README.md`, `REVIEW.md` |
| **D** Client authentication (mutual TLS) for streaming (ST8 / SR6) | dsp-stream | `dsp-stream/REVIEW.md` |
| **D** Scope of remote execution (SR7): what a server may run for a client | dsp-stream | `dsp-stream/REVIEW.md` |
| **D** Is the HD-EMG test recording (`nwb/15-25-33_meps.nwb.zarr`) shareable? | playground | `playground/REVIEW.md` |
| **D** EMUsort linear channel map (S8) | dsp-synapse-ml | `SORTER_TASKS.md` |

## Work

| Item | Crate | Source |
|---|---|---|
| Un-park Curation (sorter views) on the new backend; uses `dsp-base/math/geometry.rs` (lasso) | dsp-app | `dsp-app/README.md` |
| Remote `SignalBackend` over dsp-stream's `Session` | dsp-view, dsp-app | `dsp-view/README.md` |
| Live acquisition producers (the parked `dsp-stream/buffer/ring.rs` may serve; delete it if not) | dsp-stream | book `crates/dsp-stream.md` |
| Subscribe to a subset of channels | dsp-stream | book `crates/dsp-stream.md` |
| Generate the Python stub with `pyo3-stub-gen` (PY14) | dsp_kitchen_py | `dsp_kitchen_py/README.md` |
| Bind `PipelineWorkspace` for chunk-by-chunk Python streaming (PY4) | dsp_kitchen_py | `dsp_kitchen_py/README.md` |
| ONNX model runtime (`burn-onnx`, device-resident weights) once a model artifact is validated; parked `catalog/`, `runtime/kernels/` | dsp-synapse-ml | `dsp-synapse-ml/README.md` (step 4) |
| Raw sidecar schema still names µV (`gain_uv`); storing a unit is a schema change | dsp-io | `dsp-io/README.md` |
| `decode_run` duplicates dsp-core `SampleFormat::decode` | dsp-io | `dsp-io/README.md` |
| NWB `ProbeSource` (electrode positions); `.npz` zip64; `.sorting.zarr` vs SpikeInterface's layout | dsp-io | book `crates/dsp-io.md` |
| `TimeRange` unused | dsp-core | book `crates/dsp-core.md` |
| Host-side: FastICA iterations, PPCA EM, ZCA assembly, trimmed / IQR noise | dsp-base | book `crates/dsp-base.md` |
| Host-side: localizers (one spike at a time), `cluster_kde_merge`; NEO / matched filter negative-only and greedy (SP7) | dsp-synapse | `dsp-synapse/README.md`, book |
| Drift registration with a single reference bin | dsp-synapse | book `crates/dsp-synapse.md` |
| Clip extraction for template learning on the host | dsp-synapse-ml | book `crates/dsp-synapse-ml.md` |
| `gh://` cannot fetch Git-LFS / release assets; blocking downloads | dsp-synapse-hub | book `crates/dsp-synapse-hub.md` |
| Verify `WHITENING_EPSILON = 1e-6` against Kilosort4 | dsp-synapse-ml | `playground/README.md` |
| Name the literals in dsp-io's `SyntheticRecording` | dsp-io | `playground/README.md` |
| Run the remaining end-of-cleanup tests (dsp-view suites, app UI test) | several | `dsp-view/README.md` |

## Watching

- One intermittent native crash ("corrupted double-linked list", 1 in 10 runs of an EMUsort script)
  and a one-off "NVVM compilation failed: 3" at exit of a wgpu run
  (`.knowledge/dsp-kitchen-gpu-lessons.md`, section 7).
