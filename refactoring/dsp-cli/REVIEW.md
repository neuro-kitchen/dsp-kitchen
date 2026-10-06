# dsp-cli — review (2026-10-06)

Read-only review; no code changed. About 2,000 lines: `main.rs` (clap) and 10 commands.
**D** = a decision needed from you.

## Role

A headless command line over the library crates: inspect devices and recordings, generate test
data, benchmark, serve / receive streams, manage the hub. It should hold no algorithms; every
command is a thin call into a crate. It is also the only place, besides the app and Python,
allowed to choose a device: `--runtime` / `DSP_KITCHEN_RUNTIME` → `ComputeTarget`, which is
correct.

## Build state

It does not build. Upstream changes it uses (already listed in the other crates' READMEs):

| Command | Upstream change |
|---|---|
| `probe` | `dsp_synapse::probe` → `dsp_io::neuro::probe` |
| `open` (`info.rs`) | `ChannelInfo::gain_uv` → `gain` / `offset` / `unit` |
| `serve` | `dsp_stream::PrefetchReader` → `dsp_io::PrefetchReader`; synthetic data behind dsp-io `neuro` |
| `benchmark` | `execute_median_9p`, `match_spikes_omp_on`, `precompute_knn_table`, `StreamingSpikeRunner` / `StreamingSortConfig`, `dsp_core::layout` all renamed or moved |
| `hub` | `dsp_synapse_ml::hub::{ModelHub, ModelStatus}` → dsp-synapse-hub / the `hub` feature |

## Findings

### Commands that report things that are not true

- **CLI1 — `components` is hard-coded text.** It prints:
  - "INSTALLED" for every crate, including the Python bindings, which a CLI binary cannot know;
  - a ring buffer (parked), "Zarr v3, Decimate LOD" in dsp-stream (not there), and a
    "Burn-ONNX External Sorter Bridge" (none);
  - `memmap2 v0.9.11` as a literal;
  - "CPU JIT ACTIVE" whatever was compiled in, and "ALL CORE SUBSYSTEMS OPERATIONAL".

  It is also aliased `info`, while `open` lives in `info.rs`.
  *Proposal:* replace it with a `doctor` command that reports facts:
  - the version;
  - the runtimes compiled in, which ones are usable on this machine
    (`ComputeTarget::available()` / `is_available`), and each one's device and limits;
  - the cargo features enabled (via `cfg!`);
  - the hub cache path.
- **CLI2 — `stream` is a calculator for a mode that does not exist.** It prints the bandwidth for
  "processing", and a min/max decimation kernel for "vis" (the default); any other word also
  means "vis". *Proposal:* remove it (ST4); `serve` prints the bandwidth it actually uses. **D**
- **CLI3 — `receive` measures loss that cannot happen.** Frames travel on one reliable, ordered
  QUIC stream, so sequence gaps cannot occur: "Packet Drop Rate" is always 0. "Latency & jitter"
  is the interval between frame arrivals, not latency, because frames carry no send time.
  *Proposal:* report throughput, the real-time factor and inter-arrival times under those names.
  For latency, add a send timestamp to the frame or header (ST1).

### Bugs

- **CLI4 — `receive --save` can panic and writes a file dsp-io cannot open.**
  - It assumes every frame has the first frame's `samples`. The last frame of a non-looping
    file is shorter, so `copy_from_slice` indexes past the chunk and panics.
  - It keeps the whole stream in memory until the end.
  - It writes its own `.meta` JSON (a name SpikeGLX also uses), not dsp-io's raw sidecar, and
    casts the floats with `unsafe`.

  *Proposal:* write the chunks as they arrive through dsp-io's raw writer, so `dsp-cli open`
  reads the file back.
- **CLI5 — `benchmark --save` writes a bare `.bin` with no sidecar** (the same problem as CLI4),
  and allocates buffers with a byte literal (`* 4`) rather than the element type.
- **CLI6 — `probe` with an unknown model prints a message and exits 0.** It should return an
  error, so scripts can tell.

### Defaults and units

- **CLI7 — 30 kHz and 384 channels are built in.**
  - `benchmark` hard-codes `const FS = 30 kHz` (it has no option for it), and its kernels-only
    real-time factor assumes 30 kHz.
  - `inspect`, `generate`, `benchmark` and `stream` default to 384 channels and 30,000 samples;
    `serve` falls back to 384 channels at 30 kHz for its synthetic source.
  - These are fine as values for synthetic data, but the rate should be an option wherever time
    is reported.
- **CLI8 — µV is assumed.**
  - `generate --noise-uv`, `--line-noise-uv` and `--gain-uv`; `serve --int16-gain-uv`.
  - `open` prints "uV/bit" and the first channel's range in µV.
  - The benchmark chain scales by 0.195.

  *Proposal:* `open` prints each channel's own `unit`; the generator options drop the suffix
  and state the unit once.
- **CLI9 — Output defaults point into the repository**: `playground/data/…` and
  `playground/benchmarks`, relative to the working directory. *Proposal:* require `--output`,
  or write to the current directory.

### Structure

- **CLI10 — The pipeline benchmark is not the production path.** It chains `execute_scaling`, a
  notch and TKEO by hand. What users run is `PipelineWorkspace` (int16 upload, ping-pong buffers).
  *Proposal:* benchmark `PipelineWorkspace` with stages from the command line, and the streaming
  detector on a real file (`dsp-cli benchmark <recording>`).
- **CLI11 — Too many aliases and loose strings.**
  - Aliases: `components` = `info` = `detect` = `doctor`; `generate` = `mock-signal`;
    `serve` = `stream-server`; `receive` = `benchmark-net` = `test-stream`; `hub` = `models`;
    `--file` = `--input`; `--no-realtime` = `--stress`; `np1` / `utah-array`.
  - Free strings parsed by hand: `--format`, `--dtype`, `--order`, `--mode`, `--model`.

  *Proposal:* no aliases; `clap::ValueEnum` for every choice, so `--help` lists the valid values
  and a typo is rejected.
- **CLI12 — Features do not match the runtimes.** `cubecl/wgpu` is always on. The `cpu` / `cuda` /
  `hip` features enable dsp-base / dsp-synapse features, not dsp-core's runtimes, which are
  what `ComputeTarget::available()` reads. *Proposal:* features `wgpu` (default), `cpu`, `cuda`,
  `hip` forwarding to dsp-core and to every crate that compiles kernels.
- **CLI13 — `main.rs` re-declares every command's arguments** and passes them on one by one
  (`run_serve` has 9 positional parameters). *Proposal:* one `#[derive(Args)]` struct per command
  inside its module, as `generate` and `hub` already do; `main` only dispatches.
- **CLI14 — Dependencies.** `memmap2` is unused (it is only named in the `components` text).
  `quinn` is used only for the `quinn::Connection` type in `serve.rs`. That type would disappear
  if dsp-stream exposed its own session type (ST1).
- **CLI15 — No commands for the main library features.** There are none for detection, sorters,
  sorting files or Phy export. *Proposal (later):*
  - `dsp-cli detect <recording>` (streaming detection → `.sorting.zarr` / Phy);
  - `dsp-cli sort kilosort4|emusort <recording>` once those pipelines exist;
  - `dsp-cli convert <sorting> --to phy|nwb|zarr`.

  These are thin over dsp-synapse and dsp-synapse-ml. **D**

### Fine as is

- `--runtime` is global and checked against what is compiled in.
- `open` times both the scaled read and the stored read (the low-transport path).
- `generate` writes in bounded chunks and re-opens the result through format detection.
- `serve` streams through a bounded queue and stops cleanly on Ctrl+C.
- Benchmark regions are device-synchronized (`dsp_core::compute::bench`).

## Proposed order (after your decisions)

1. Fix the upstream breakage (table above) and CLI4–CLI6.
2. Restructure: CLI11–CLI14; `components` → `doctor` (CLI1); remove `stream` (CLI2).
3. Rework `receive` and `serve` after dsp-stream's header and stored payload (ST1, ST2), then fix
   CLI3.
4. Benchmarks on the production path (CLI10); new commands (CLI15).
