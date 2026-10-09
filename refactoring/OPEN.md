# Open items (index, 2026-10-08)

Everything still open from the cleanup records, in one place. Each line points to where the details
are. **D** = a decision needed from the user. Checked against the code's current state (the book's
*Limitations* sections); items the records list but later work resolved are not repeated here.

Task lists with their own detail: `GPU_TASKS.md` (GPU review; task 8 partly open), `SORTER_TASKS.md`
(everything missing from Kilosort4 and EMUsort), `NEW_SORTERS_PLAN.md` (MountainSort 5, SpyKING CIRCUS 2,
Tridesclous 2), `DOCS_TASKS.md` (done).

## Decisions

| Item | Crate | Source |
|---|---|---|
| **D** Client authentication (mutual TLS) for streaming (ST8 / SR6) | dsp-stream | `dsp-stream/REVIEW.md` |
| **D** Scope of remote execution (SR7): what a server may run for a client | dsp-stream | `dsp-stream/REVIEW.md` |
| **D** EMUsort linear channel map (S8) | dsp-synapse-ml | `SORTER_TASKS.md` |

Done 2026-10-08 (phase 1): raw sidecar and Zarr traces store their unit (`gain`, `offset`,
`unit`; old `gain_uv` files still read as µV; `write_zarr` stores were read back as dimensionless:
fixed); `decode_run` removed for `SampleFormat::decode`; `WHITENING_EPSILON` checked against
Kilosort4's saved whitening (eigenvalues 11–277: ε irrelevant below 10⁻³); `SyntheticRecording`
literals named; `TimeRange` kept (deliberate, see the book).

Decided 2026-10-08: CLI15 → `detect`, `sort kilosort4|emusort`, `convert` added (clap, every
setting a flag with its default; `sort … --show-config`). Nothing in `data/` is shareable (HD-EMG
included): tests and docs must not depend on it being published.

## Work

| Item | Crate | Source |
|---|---|---|
| Un-park Curation (sorter views) on the new backend; uses `dsp-base/math/geometry.rs` (lasso) | dsp-app | `dsp-app/README.md` |
| Remote `SignalBackend` over dsp-stream's `Session` | dsp-view, dsp-app | `dsp-view/README.md` |
| Live acquisition producers (the parked `dsp-stream/buffer/ring.rs` may serve; delete it if not) | dsp-stream | book `crates/dsp-stream.md` |
| Subscribe to a subset of channels | dsp-stream | book `crates/dsp-stream.md` |
| Bind `PipelineWorkspace` for chunk-by-chunk Python streaming (PY4) | dsp_kitchen_py | `dsp_kitchen_py/README.md` |
| ONNX model runtime (`burn-onnx`, device-resident weights) once a model artifact is validated; parked `catalog/`, `runtime/kernels/` | dsp-synapse-ml | `dsp-synapse-ml/README.md` (step 4) |
| NWB `ProbeSource` (electrode positions); `.npz` zip64; `.sorting.zarr` vs SpikeInterface's layout | dsp-io | book `crates/dsp-io.md` |
| Host-side: FastICA iterations, PPCA EM, ZCA assembly, trimmed / IQR noise | dsp-base | book `crates/dsp-base.md` |
| Host-side: localizers (one spike at a time), `cluster_kde_merge`; NEO / matched filter negative-only and greedy (SP7) | dsp-synapse | `dsp-synapse/README.md`, book |
| Drift registration with a single reference bin | dsp-synapse | book `crates/dsp-synapse.md` |
| Clip extraction for template learning on the host | dsp-synapse-ml | book `crates/dsp-synapse-ml.md` |
| `gh://` cannot fetch Git-LFS / release assets; blocking downloads | dsp-synapse-hub | book `crates/dsp-synapse-hub.md` |
| Run the remaining end-of-cleanup tests (dsp-view suites, app UI test) | several | `dsp-view/README.md` |

## Watching

- Fixed 2026-10-09: the exit crash/hang (devices are now shut down at exit; see
  `.knowledge/dsp-kitchen-gpu-lessons.md` section 7). Watch for recurrences on other drivers.
