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

## Repository

```text
crates/       dspk, dspk-core, dspk-io, dspk-ops                    (published)
serve/        dspk-view, dspk-stream                                (published)
engine/       dspk-pipeline: pipelines as data (JSON / TOML specs), many inputs, merges
synapse/      dspk-synapse (detection, clustering, matching, sorter building blocks, provenance),
              dspk-synapse-io (ephys formats), dspk-synapse-hub (downloads),
              dspk-synapse-ml (pretrained models)
  sorters/    one crate per sorter: ks4 (Kilosort4 + EMUsort), ms5 (MountainSort 5),
              sc2 (SpyKING CIRCUS 2), tdc2 (Tridesclous 2)
app/          dspk-app (desktop workbench), dspk-cli (`dspk run`, `dspk sort`, …)
bindings/     python: the `dspk` package on PyPI
playground/   worked examples as `# %%` cells (see playground/README.md)
benchmarks/   timings and validations against scipy and other sorters (see benchmarks/README.md)
```

Every crate builds with wgpu, CUDA and the CPU runtime by default. Development and tests use one
profile:

```bash
cargo build --workspace --profile validate
cargo test  --workspace --exclude dspk-app --exclude dspk-py --profile validate
```

Each kernel is tested on every runtime compiled in (`DSPK_TEST_RUNTIMES=cuda` limits a run to one).
Timings against scipy and kernel benchmarks: [benchmarks](benchmarks/README.md).

## Runtimes

Runtimes are crate features; the device is picked at run time with `dspk::core::ComputeTarget`
(or the `DSPK_RUNTIME` environment variable) among those compiled in.

```toml
[dependencies]
dspk = "0.0.4"                                                                  # wgpu, cuda and cpu (default)
# dspk = { version = "0.0.4", default-features = false, features = ["wgpu", "zarr"] }  # wgpu only
```

| Feature | Runtime | Tested |
|---|---|---|
| `wgpu` (default) | WebGPU: Vulkan, Metal, DirectX 12 | NVIDIA RTX 2070 Max-Q, Intel UHD 630 (Vulkan) |
| `cuda` (default) | NVIDIA CUDA | NVIDIA RTX 2070 Max-Q |
| `cpu` (default) | CubeCL CPU runtime (MLIR / LLVM) | x86-64 |
| `hip` | AMD ROCm / HIP | compiles; **untested** (no AMD GPU available) |

CUDA and HIP are loaded at run time, so building with them needs no toolkit installed. The CPU
runtime downloads a prebuilt LLVM on its first build (cached in `~/.local/share/tracel`); leave it
out with `default-features = false` for a lighter build. All three are the default because the
library, its tests, the CLI and the Python package are built and tested together that way.

### Kernel and autotune cache

Autotune decisions and compiled kernels are cached across processes: the first run on a device
tunes and compiles, later runs start in about a second. The cache is one SQLite database,
`default.db` (with its `-wal` / `-tshm` journal files), in CubeCL's folder of the user cache
directory:

| OS | Location |
|---|---|
| Linux, macOS | `$XDG_CACHE_HOME/cubecl/`, by default `~/.cache/cubecl/` |
| Windows | `%LOCALAPPDATA%\cubecl\` |
| No user directory (no `HOME`, minimal containers) | the temporary directory, `cubecl/` |

- `DSPK_CACHE_DIR=/some/dir` puts it in that directory instead (shared machines, CI caches,
  read-only home directories).
- `DSPK_KERNEL_CACHE=0` turns the compiled-kernel cache off; autotune decisions still persist.
- Deleting the folder is safe: the next run tunes and compiles again. It grows with every runtime,
  device and kernel variant used (about 130 MB after the full test suite on wgpu, CUDA and CPU).
- A `cubecl.toml` in the working directory, or CubeCL's own environment variables, still apply.
- Code that opens a device through CubeCL directly instead of dspk gets CubeCL's default:
  `target/environment/` of the nearest Cargo project, or the user cache directory outside one.

## Python

```bash
pip install dspk
```

```python
import dspk
from dspk.ops import Pipeline, stage

dspk.runtime.available()                      # e.g. ['wgpu', 'cuda', 'cpu']
pipe = Pipeline([stage.car(), stage.iir("bandpass", (300, 6000), mode="forward")],
                sample_rate=30_000, stream=True)
out = pipe.process(chunk)                     # (channels, samples) in, same shape out, on the GPU
```

Wheels for Linux x86_64 and Windows x64 (wgpu, CUDA, CPU runtime) and macOS arm64 (wgpu, CPU);
CPython ≥ 3.10. See [bindings/python](bindings/python/README.md).

**Device size limit.** Kernels index with 32-bit integers (WebGPU has no 64-bit integers, and one
kernel source serves every runtime): one window or batch on the device holds at most 2³² − 1
values (e.g. 384 channels × 11.1 million samples). Larger requests are refused with an error.

## License

MIT OR Apache-2.0, at your option ([LICENSE-MIT](LICENSE-MIT), [LICENSE-APACHE](LICENSE-APACHE)),
except the sorter crates that follow their upstream's licence:

| Crate | Licence | Why |
|---|---|---|
| `dspk-ks4` (Kilosort4, EMUsort) | GPL-3.0-only | Kilosort4 and EMUsort are GPL-3.0 |
| `dspk-ms5` (MountainSort 5) | Apache-2.0 | MountainSort 5 and isosplit6 are Apache-2.0 |
| `dspk-sc2`, `dspk-tdc2` | MIT OR Apache-2.0 (+ SpikeInterface's MIT notice) | ported from SpikeInterface (MIT) |
| `dspk-synapse` | (MIT OR Apache-2.0) AND Apache-2.0 | isosplit6 port; SpikeInterface components |

Each sorter crate carries its upstream licence and a `NOTICE.md`. The published crates never depend
on `dspk-ks4`; the `dspk-cli` binary does, so a CLI built with Kilosort4 is distributed under
GPL-3.0 as a whole.
