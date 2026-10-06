# dsp-kitchen

> [!WARNING]
> **Under development.** Every crate and the Python bindings build except `dsp-app`, which is
> being brought up to the reorganized crates. Kilosort4 and EMUsort run over whole recordings
> (preprocessing, universal templates, detection); their clustering and deconvolution stages are
> not implemented yet. Nothing here is released; APIs change.

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
  constant data is uploaded once per run; recordings are read out of core with background
  prefetching (`dsp_core::WindowLoader`), so host memory stays bounded.
- **Sorters from their papers.** Kilosort4 and EMUsort (one shared runner) stream a whole
  recording through preprocessing, universal-template learning and detection on the device, in
  Rust or from Python.
- **scipy / numpy semantics** for defaults where an equivalent exists, and exact chunked
  processing: any batch size gives the whole-recording result.
- **Provenance.** Every sorter and model names the paper (DOI), code, license and downloaded
  files it comes from.

## Crates

| Crate | Role |
|---|---|
| `dsp-core` | Exact time and sample rates, buffers, errors, the `RecordingSource` contract, out-of-core window streaming, compute-runtime selection. |
| `dsp-io` | Recording formats and containers (`.npy`, `.npz`, Zarr), NWB / SpikeGLX / mtscomp, probe geometry, sorting files (Phy, NWB units, `.sorting.zarr`). |
| `dsp-base` | DSP primitives on the device: IIR / FIR / non-linear filters, resampling, spatial operators, linear algebra, peak finding, statistics, device pipelines. |
| `dsp-synapse` | Spike detection, deduplication, extraction, features, localization, drift, clustering, template matching, metrics, streaming detection. |
| `dsp-synapse-ml` | Sorters from papers (Kilosort4, EMUsort) and pretrained models, with provenance. |
| `dsp-synapse-hub` | Verified download and cache of published artifacts. |
| `dsp-stream` | Network sessions for continuous signals (QUIC + TLS, protobuf): exact header, stored samples, views. |
| `dsp-view` | Preparing signals for viewing, locally or remotely: min/max envelopes and pyramids. |
| `dsp-cli`, `dsp-app`, `dsp_kitchen_py` | Command line, desktop app, Python bindings. |

## Building

Requires a recent stable Rust toolchain (edition 2024).

```bash
cargo build --workspace --exclude dsp-app   # dsp-app: see the warning
cargo build -p dsp-core --features cuda     # runtimes are features: wgpu (default), cuda, hip, cpu
```

Python (`dsp_kitchen`, default features `wgpu` and `hub`):

```bash
cd dsp_kitchen_py && maturin develop
python ../playground/sorters/kilosort4_universal_templates.py   # examples: see playground/README.md
```

## Citing

If you use a sorter, cite its authors — see
[Citations](docs/book/src/reference/citations.md). In code:
`dsp_synapse_ml::Attributed::provenance()` and `Provenance::citation()`.

## License

MIT OR Apache-2.0 (see [LICENSE](LICENSE)). Sorters are written from their papers; upstream
implementations under other licenses (e.g. GPL-3.0) are not included.
