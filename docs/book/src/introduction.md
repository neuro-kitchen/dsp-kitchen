# dsp-kitchen

> **Under development.** The workspace is being reorganized crate by crate. `dsp-core`,
> `dsp-base` and `dsp-synapse-hub` build; `dsp-io`, `dsp-synapse` and `dsp-synapse-ml` are being
> brought up to the new core API; fixes to `dsp-stream`, `dsp-cli`, `dsp-app` and the Python
> bindings are coming. Nothing here is released.

dsp-kitchen is a digital-signal-processing framework for multi-channel recordings, written in
Rust, with a focus on neural and muscle electrophysiology: filtering, resampling and spatial
operators, spike detection, waveform extraction, localization, drift correction, sorting and
quality metrics, and spike sorters reimplemented from their papers.

## Highlights

### One code base, any GPU or the CPU

Every device algorithm is a [CubeCL](https://github.com/tracel-ai/cubecl) kernel, written once in
Rust and compiled for whichever runtime is available: **WebGPU** (Vulkan, Metal, DirectX 12),
**CUDA**, **ROCm / HIP**, or the **CPU**. The runtime is chosen at run time, not at build time:

- `dsp_core::ComputeTarget::available()` lists the runtimes compiled in (GPUs first);
  `ComputeTarget::from_env()` honours `DSP_KITCHEN_RUNTIME`.
- Algorithms never choose a device: they take a `ComputeClient<R>` for any CubeCL runtime `R`.
  Only entry points (CLI, app, Python) pick one.
- Launch shapes come from the device (`dsp_core::compute::LaunchGeometry`), and where the
  fastest variant depends on the hardware (filter block counts and memory layouts, FIR direct vs
  tiled, peak-candidate block length) CubeCL's autotuner measures and keeps the winner per device.
- Kernels are generic over the float type (`f32` everywhere; `f64` / `f16` where the runtime
  supports them).

See [Architecture](architecture.md#compute).

### Data stays where it is processed

Moving samples between host RAM and device memory is usually the slowest part of a GPU
pipeline, so the framework is built to move as little as possible:

- **Device-resident pipelines.** `dsp_base::Pipeline` chains stages (scaling, filters, common
  reference, whitening, Laplacian, smoothing, median, …) on the device with ping-pong buffers;
  intermediate results never come back to the host.
- **Integer recordings upload as stored.** int16 data is uploaded as is (half the bytes of f32)
  and scaled on the device.
- **Compaction on the device.** Detection finds and compacts candidates on the device
  (`dsp_base::peaks::find_peak_candidates`); only the candidates are downloaded, not the trace.
  The same pattern is used for template matching, deduplication, noise estimation (one σ per
  channel comes back) and EM clustering (only model parameters cross the bus).
- **Out-of-core reading.** `dsp_io::PrefetchReader` reads the next halo window on a background
  thread while the device processes the current one; host memory stays at two windows whatever
  the recording length.
- **Exact streaming.** Detection and deduplication give the same result in chunks as on the
  whole recording, so long recordings never need to fit in memory.

See [Architecture](architecture.md#data-movement).

## Crates

| Crate | Role |
|---|---|
| [`dsp-core`](crates/dsp-core.md) | Exact time and sample rates, buffers, errors, the `RecordingSource` contract, runtime selection. |
| [`dsp-io`](crates/dsp-io.md) | Recording formats, containers (`.npy` / `.npz`, Zarr), neural formats, probe geometry, sorting files. |
| [`dsp-base`](crates/dsp-base.md) | DSP primitives on the device: filters, resampling, spatial operators, linear algebra, peaks, statistics, pipelines. |
| [`dsp-synapse`](crates/dsp-synapse.md) | Spike detection, extraction, features, localization, drift, clustering, matching, metrics, storage conversions, streaming detection. |
| [`dsp-synapse-ml`](crates/dsp-synapse-ml.md) | Sorters reimplemented from papers (Kilosort4, EMUsort) and pretrained models, each with its provenance. |
| [`dsp-synapse-hub`](crates/dsp-synapse-hub.md) | Verified download and cache of published artifacts. |
| [`dsp-stream`](crates/dsp-stream.md) | Network sessions for continuous signals (QUIC + TLS, protobuf): exact header, stored samples, views. |
| [`dsp-view`](crates/dsp-view.md) | Preparing signals for viewing, locally or remotely: min/max envelopes (host and device) and pyramids. |
| `dsp-cli`, `dsp-app`, `dsp_kitchen_py` | Command line, desktop app, Python bindings. *Documentation pending.* |

## Conventions

- **scipy / numpy semantics** for defaults wherever an equivalent exists (filter edges, `find_peaks`,
  `correlate`, `var(ddof)`, …); each crate page lists the correspondence.
- **Units are data.** Channels carry their `SignalUnit`; nothing assumes µV.
- **Named constants.** Defaults, floors and tolerances are named and documented in the module that
  owns them.
- **Provenance.** Every sorter and model names the paper (with DOI), code and license it comes from
  ([Citations](reference/citations.md)).
