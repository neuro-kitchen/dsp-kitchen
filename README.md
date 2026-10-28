# dspk

> [!WARNING]
> **Early development (0.0.x).** APIs change between releases. Documentation is being rewritten.

Multi-channel signal processing written once in [CubeCL](https://github.com/tracel-ai/cubecl) and
run on any GPU (wgpu: Vulkan, Metal, DirectX 12; CUDA; ROCm / HIP) or the CPU, with the device
chosen at run time. Processing stages keep data on the device between them, so a recording
crosses disk → RAM → device memory as few times as possible.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/neuro-kitchen/dsp-kitchen/main/assets/img/dsp-app_main-darkmode.png">
  <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/neuro-kitchen/dsp-kitchen/main/assets/img/dsp-app_main-lightmode.png">
  <img alt="dspk-app: 385 channels of a Neuropixels recording as traces and a heatmap, with the channel list, view settings and timeline" src="https://raw.githubusercontent.com/neuro-kitchen/dsp-kitchen/main/assets/img/dsp-app_main-darkmode.png">
</picture>

*`dspk-app` (desktop workbench, not on crates.io): a Neuropixels recording (385 channels, 30 kHz)
drawn from min/max pyramids built by `dspk-view`.*

## Crates

| Crate | Contents |
|---|---|
| `dspk` | Entry crate: re-exports the crates below; features choose what is included. |
| `dspk-core` | Recordings (`RecordingSource`), buffers, exact time and sample rates, windows, compute targets. |
| `dspk-io` | Raw binary and Zarr recordings, format detection, chunk cache, `.npy`/`.npz`/Zarr containers. |
| `dspk-ops` | IIR / FIR / non-linear filters, resampling, spatial operators, linear algebra, statistics, peak finding, device pipelines. |
| `dspk-view` | Min/max envelopes and pyramids for viewing, local or remote. |
| `dspk-stream` | Network sessions for continuous signals (QUIC + TLS, protobuf). |

Electrophysiology (spike detection, sorters, neural file formats) lives in the `synapse/` crates of
the repository and is not published yet.

## Runtimes

Runtimes are crate features; the device is picked at run time with `dspk::core::ComputeTarget`
(or the `DSPK_RUNTIME` environment variable) among those compiled in.

```toml
[dependencies]
dspk = "0.0.1"                                               # wgpu (default)
# dspk = { version = "0.0.1", features = ["cuda", "cpu"] }   # add CUDA and the CPU runtime
```

| Feature | Runtime | Tested |
|---|---|---|
| `wgpu` (default) | WebGPU: Vulkan, Metal, DirectX 12 | NVIDIA RTX 2070 Max-Q, Intel UHD 630 (Vulkan) |
| `cuda` | NVIDIA CUDA | NVIDIA RTX 2070 Max-Q |
| `cpu` | CubeCL CPU runtime (MLIR / LLVM) | x86-64 |
| `hip` | AMD ROCm / HIP | compiles; **untested** (no AMD GPU available) |

CUDA and HIP are loaded at run time, so building with them needs no toolkit installed.

**Device size limit.** Kernels index with 32-bit integers (WebGPU has no 64-bit integers, and one
kernel source serves every runtime): one window or batch on the device holds at most 2³² − 1
values (e.g. 384 channels × 11.1 million samples). Larger requests are refused with an error.

## License

MIT OR Apache-2.0.
