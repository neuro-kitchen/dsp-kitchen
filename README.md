# dsp-kitchen

> [!WARNING]
> **Under development.** The workspace is being reorganized crate by crate, and downstream crates
> do not build while that happens. Today `dsp-core`, `dsp-base` and `dsp-synapse-hub` build;
> `dsp-io`, `dsp-synapse` and `dsp-synapse-ml` are being brought up to the new core API.
> **Fixes to `dsp-synapse-ml`, `dsp-stream`, `dsp-cli`, `dsp-app` and the Python bindings are
> coming soon.** Nothing here is released; APIs change.

GPU-accelerated digital signal processing for multi-channel recordings, in Rust, with a focus on
neural and muscle electrophysiology: filtering, resampling, spatial operators, spike detection,
waveform extraction, localization, drift correction, clustering, quality metrics, and spike
sorters reimplemented from their papers (Kilosort4, EMUsort).

**Documentation:** the mdBook in [`docs/book`](docs/book/src/introduction.md)
(`mdbook build docs/book` → `target/book`; published at
<https://neuro-kitchen.github.io/dsp-kitchen/>).

## Highlights

- **Any GPU or the CPU, chosen at run time.** Device code is written once as
  [CubeCL](https://github.com/tracel-ai/cubecl) kernels and runs on WebGPU (Vulkan, Metal,
  DirectX 12), CUDA, ROCm / HIP or the CPU. Libraries take a compute client and never pick a
  device; entry points choose with `ComputeTarget` (`DSP_KITCHEN_RUNTIME`). Device-dependent
  choices are autotuned per device.
- **Minimal data transport.** Processing pipelines keep data on the device between stages;
  integer recordings upload as stored and are scaled there; detection, matching and clustering
  compact their results on the device so only spikes, statistics or parameters come back;
  recordings are read out of core with background prefetching, so host memory stays bounded.
- **scipy / numpy semantics** for defaults where an equivalent exists, and exact chunked
  processing: any batch size gives the whole-recording result.
- **Provenance.** Every sorter and model names the paper (DOI), code, license and downloaded
  files it comes from.

## Crates

| Crate | Role |
|---|---|
| `dsp-core` | Exact time and sample rates, buffers, errors, the `RecordingSource` contract, compute-runtime selection. |
| `dsp-io` | Recording formats and containers (`.npy`, `.npz`, Zarr), NWB / SpikeGLX / mtscomp, probe geometry, sorting files (Phy, NWB units, `.sorting.zarr`), out-of-core prefetching. |
| `dsp-base` | DSP primitives on the device: IIR / FIR / non-linear filters, resampling, spatial operators, linear algebra, peak finding, statistics, device pipelines. |
| `dsp-synapse` | Spike detection, deduplication, extraction, features, localization, drift, clustering, template matching, metrics, streaming detection. |
| `dsp-synapse-ml` | Sorters from papers (Kilosort4, EMUsort) and pretrained models, with provenance. |
| `dsp-synapse-hub` | Verified download and cache of published artifacts. |
| `dsp-stream` | Network transport of continuous signals (QUIC + TLS). |
| `dsp-cli`, `dsp-app`, `dsp_kitchen_py` | Command line, desktop app, Python bindings. |

## Building

Requires a recent stable Rust toolchain (edition 2024).

```bash
cargo build -p dsp-core --features wgpu   # pick runtimes: wgpu, cuda, hip, cpu
cargo build -p dsp-base
cargo build -p dsp-synapse-hub
```

The rest of the workspace builds again once the reorganization reaches it (see the warning).

## Citing

If you use a sorter, cite its authors — see
[Citations](docs/book/src/reference/citations.md). In code:
`dsp_synapse_ml::Attributed::provenance()` and `Provenance::citation()`.

## License

MIT OR Apache-2.0 (see [LICENSE](LICENSE)). Sorters are written from their papers; upstream
implementations under other licenses (e.g. GPL-3.0) are not included.
