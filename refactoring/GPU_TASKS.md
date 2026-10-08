# GPU tasks (from the 2026-10-07 GPU review)

Status: **in progress** (user go-ahead 2026-10-07). Each step is written up in the mdBook part "GPU engineering". Numbers match the
review. Each task ends with `cargo check` only; tests and benchmarks run at the end, as for the
crate cleanup.

| # | Task | Decision | Crates |
|---|------|----------|--------|
| 0 | Upgrade to cubecl 0.11 / cubek 0.3 / burn 0.22 | **done 2026-10-07** (compiles, tests not run) | workspace |
| 1 | Covariance / second moment as a matmul | **done 2026-10-07** (+ dense spatial, projection; direct kernel as tuning candidate) | dsp-base |
| 2 | No CPU↔GPU stalls per window in the sorter pipeline | **done differently 2026-10-07**: measured, stalls = 7 of 316 ms; the real costs fixed instead (host upload copy, `centre_response`, correlation, `neighbour_max`): 477 → 123 ms/window. Capacity/double-buffer redesign not built (~2%) | dsp-base, dsp-synapse-ml |
| 3 | Upload stored samples (int16), unpack on the device | **done 2026-10-07** (`DeviceWindows`) | dsp-synapse-ml |
| 4 | Exact k-th smallest `|x|` by bit-pattern select | **done 2026-10-07** (radix select, 8-bit digits; 4–10× faster, exact) | dsp-base |
| 5 | Eigensolver tolerance aware of `F` and `n` | **done 2026-10-07** (Demmel–Veselić pair test; measured: no speed change, review hypothesis wrong) | dsp-base |
| 6 | 32-bit device index guard + README warning | **done 2026-10-07** (`device_elements`, buffer backstop, README) | dsp-core, dsp-base, all kernels, README |
| 7 | HDBSCAN scaling (exact pruning) | **done 2026-10-07** (+ hang fix: bounded insertion; 500k clips in 52 s) | dsp-synapse |
| 8 | Smaller kernel inefficiencies | **partly done** (correlation tiled; rest open: delay CC, plane reductions, zeros fill, transpose via cubecl-std) | dsp-base, dsp-synapse-ml, dsp-view |
| 9 | Reproducible sorts | **done 2026-10-08** (`pin_tuned_choices`, `Kilosort4Config::reproducible`; identical spikes and units run to run) | dsp-core, dsp-base, dsp-synapse(-ml) |

---

## 0. CubeK and the CubeCL 0.11 upgrade (done 2026-10-07: option B)

**CubeK** (`tracel-ai/cubek`, "CubeCL Kernels") is Tracel's kernel library on CubeCL: matmul,
reduce, FFT, random, convolution, attention, quantization, interpolate, pool. **cubek 0.2.0 is
already in our `Cargo.lock`** (pulled in by burn 0.21, built against cubecl 0.10); we just never
depend on it directly.

Versions (crates.io, 2026-10-06): cubecl **0.11.0**, cubek **0.3.0** (needs cubecl ^0.11),
burn **0.22.0**. We are on cubecl 0.10 / cubek 0.2 / burn 0.21. The three move together (burn
is used by dsp-synapse-ml).

What cubecl 0.11 adds that matters here: command-graph capture and replay on CUDA / HIP / WGPU
(lower launch overhead: Jacobi global path, per-window detection chain), device fencing for
stream sync (task 2), a Metal backend (`ComputeTarget` would gain a target), a lighter CPU
runtime and CPU GEMM. Breaking: a major frontend refactor (references, new IR), so our
`#[cube]` kernels will need edits.

What cubek offers per task (from the 0.2.0 sources):
- **Task 1**: `cubek_matmul::launch::launch_ref(strategy, client, lhs, rhs, out, dtypes)` on
  `TensorBinding`s with shapes and strides, so `Xᵀ` is a stride swap and the interior columns
  `cols` are an offset + stride (no copy). `Strategy::Auto` (default) autotunes among
  unit / plane / MMA / CMMA / TMA / double-buffered routines; `Naive` exists. **Caveat:** the
  docs say "the matmul elements may get changed during selection", i.e. stage / register types
  may drop below f32 on tensor-core paths: for the whitening covariance we must pin f32
  accumulation and check precision against the current split sums.
- **Task 4**: `cubek-reduce` has `topk` / `argtopk` instructions (made for small `k`); the
  median-like `k` of `row_abs_kth` is not that case, so the radix-select research still applies.
- **Task 8**: `cubek-reduce` (`reduce(..)` with `ReduceStrategy` = Unit / Plane / Cube routine +
  vectorization; instructions `sum`, `mean`, `min`, `max`, `maxabs`, `argmin/argmax`, `prod`,
  and a public `ReduceInstruction` trait) can replace our hand tree reductions; there is no
  Welford variance (keep ours or write a `ReduceInstruction`), and the envelope's joint
  `[min, max]` would be a custom instruction. `cubek-fft` could compute the EMUsort delay
  cross-correlation by FFT (compare with the direct kernel: only `2·max_lag + 1` lags).
- **Task 9**: every cubek `Auto` strategy is autotuned, so reproducible sorts must pin the
  strategy (and dtypes) like our own tuned kernels. `cubek-random` gives seeded device RNG
  (k-means++ seeding on the device, if wanted).

**Options.** A: use cubek 0.2 now. **B (chosen):** upgrade first.

### What the upgrade changed (2026-10-07)

Versions: cubecl 0.10 → **0.11.0**, burn-tensor / onnx-ir 0.21 → **0.22.0**, cubek 0.2 → 0.3
(through burn; not a direct dependency yet). `burn-wgpu` and `burn-flex` removed: burn 0.22 picks
its backend through `burn-tensor` features (`burn-tensor/wgpu`, `burn-tensor/flex`). Whole
workspace (dsp-app and `dsp_kitchen_py` included) passes `cargo check --workspace --all-targets`
with no warnings; also checked: dsp-core `cpu`, `cuda`, `hip`; dsp-synapse-ml all features and no
default features. **Tests not run.**

cubecl 0.11 removed the runtime generic: a `Client` is one concrete type and the runtime is
chosen when it is made (`cubecl::Device`). Consequences, all **breaking** for Rust callers:
- Every `<R: Runtime>` parameter is gone: `ComputeClient<R>` → `cubecl::prelude::Client`;
  dispatchers `foo::<R, F>` → `foo::<F>`, `foo::<R>` → `foo`; structs lose `R`
  (`PipelineWorkspace<F = f32>`, `UniversalDetector`, `SecondMomentAccumulator<F>`, …).
- dsp-core: `ComputeTarget::device()` (`cubecl::Device`) and `ComputeTarget::client()`;
  `ComputeTask::run(self, client: Client)` (no generic); `run` kept for existing callers.
- Tests / entry points: `XRuntime::client(&device)` → `cubecl::Device::Wgpu(WgpuDevice::default()).client()`
  (device types from `cubecl::device`).
- Workspace `cubecl` now enables its `std` feature explicitly (0.11 with `default-features = false`
  otherwise builds `pliron` without std, which fails).

Kernel language (0.11 frontend):
- Kernel buffer arguments are slices: `&Array<T>` → `&[T]`, `&mut Array<T>` → `&mut [T]`;
  `ArrayArg::from_raw_parts` → `BufferArg::from_raw_parts`. `Array::<T>::new(n)` is now only a
  local (register) array; `#[cube]` helpers that take one keep `&Array<T>` (median rank window,
  HDBSCAN `own`).
- Shared memory: `SharedMemory::<T>::new(n)` → `Shared::<[T]>::new_slice(n)`.
- Float scalar arguments need `LaunchArg`: kernels with an `F` scalar take
  `F: Float + CubeElement + LaunchArg`; `DspFloat` = `Float + CubeElement + LaunchArg<RuntimeArg = Self>`.
- `let mut x: u32 = 0u32;` (annotated mutable locals) no longer expands: annotations dropped.
- Autotune: `LocalTuner::init(&id, …)` takes the device id (it was implicit); the
  `semicolon_in_expressions_from_non_local_macros` allow for `local_tuner!` is gone.

burn 0.22 (`dsp-synapse-ml/src/runtime/burn_engine.rs`): tensors have no backend generic
(`Tensor<D>`), one `burn_tensor::Device` (`Device::wgpu(DeviceKind::DefaultDevice)` for the wgpu
target, `Device::flex()` otherwise); `TensorData::shape()` is a method; `to_vec` → `try_to_vec`.

Not taken up yet (later tasks): cubek itself (tasks 1, 4, 8, 9), graph capture and device fences
(task 2), the Metal runtime as a `ComputeTarget`.

---

## 1. Covariance / second moment as a matmul

**Problem.** `covariance_partial_kernel` (`dsp-base/src/linalg/covariance.rs`) runs one unit per
channel pair over 4096 samples: ~channels²·samples global loads, uncoalesced, no reuse. It is
`X Xᵀ` (SYRK). `pca_project_kernel` (`linalg/kernels/matmul.rs`) is the same pattern.

**Do.**
- Use `cubek-matmul` (task 0): strided / offset `TensorBinding`s for `Xᵀ` and the interior
  columns `cols`; pinned f32 element types; confirm it runs on wgpu, cpu, cuda and hip.
- `covariance`, `SecondMomentAccumulator::add`: `C (+)= X Xᵀ / n` through the matmul; centring
  for `covariance` by subtracting `n · μ μᵀ` (or a centred copy, if precision needs it: compare
  against the current two-pass result on the test data).
- `pca_project_kernel` → matmul `Wᵀ (X − μ)`.
- Keep accumulation precision at least as good as today's split sums (the current tests'
  tolerances must still hold).

**Done when.** Old kernels removed (one GPU path), tests unchanged, mdBook page of dsp-base
updated.

## 2. No CPU↔GPU stalls per window

**Problem.** Per detection window: read-back of candidate counts
(`dsp-base/src/peaks/device.rs`, `find_peak_candidates_on_device`) → launch → six separate
downloads (`dsp-synapse-ml/src/sorters/kilosort4/detect.rs`) → only then the next window's
upload. The GPU idles at every step.

**Do.**
- Peak compaction without the mid-pipeline read: candidates written to a persistent buffer
  with a capacity (grown and re-run when the device total exceeds it, which is rare); the total
  is read together with the results.
- One read per window: pack picked / features / amplitudes / rows / times / values into one
  device buffer (or one multi-handle read).
- Software pipelining in `run_plan`: queue window `i + 1` (upload + preprocessing + detection
  kernels) before reading window `i`'s results. Needs two sets of per-window device buffers
  (workspace ping/pong + detector scratch) so `i + 1` does not overwrite `i`.
- Keep the rule "a window crosses CPU↔GPU at most once each way".

**Done when.** One upload and one read per window; no sync between them.

## 3. Upload stored samples

**Problem.** `run_plan` / `fit_with_stages` (`kilosort4/runner.rs`) stream scaled `f32`
(`WindowLoader::stream` → `process_chunk_in_vram`): 4 bytes per sample over PCIe and a CPU
scaling pass on the main thread.

**Do.** Use `WindowLoader::stream_stored` + `PipelineWorkspace::set_stored_scaling` +
`process_stored_chunk_in_vram` (already exist). The host clip path for template learning reads
the processed device buffer, unchanged.

**Done when.** Kilosort4 / EMUsort runs move `format.bytes()` per sample; results identical up
to rounding of the scaling (gain · stored + offset, same formula as the host).

## 4. Exact k-th smallest `|x|` — research first

**Problem.** `row_abs_kth_kernel` (`dsp-base/src/core/reduce.rs`) bisects the value range
`ROW_SELECT_ITERATIONS = 64` times: 66 passes over the row, and inexact near zero.

**Research (online, before code).**
- Radix select / bit-pattern bisection on GPUs (e.g. "radix select" in CUB/RAFT `select_k`,
  "Efficient top-k on GPUs", PyTorch `kthvalue` CUDA implementation).
- Confirm: for non-negative IEEE floats the bit pattern as `u32` is monotone in the value
  (also for subnormals, ±0 → `|x|` gives +0; NaN handling: decide and match the host).
- f64 (`DspFloat` allows it): u64 bit patterns are unavailable on WebGPU; decide whether the f64
  path keeps the value bisection or splits the 64 bits into two u32 words.
- Options: 31-step bit bisection (1 pass per step) vs 4-pass 8-bit radix with a 256-bin
  histogram in shared memory (fewer passes, atomics in shared memory: check CubeCL support on
  every runtime).

**Then implement** the chosen one; result must equal `select_nth_unstable` exactly (existing
test), and remove the "~1e-9" caveat from the docs.

## 5. Eigensolver tolerance — validate online, then implement

**Problem.** `EIGEN_RELATIVE_TOLERANCE = 1e-6` (`dsp-base/src/linalg/eigen.rs`) does not depend
on `F` or `n`. In f32 the converged off-diagonal noise is ~√n·ε_F·‖A‖ (~1.2e-6 at n = 384), so
large matrices may run all `EIGEN_MAX_SWEEPS` sweeps (correct result, wasted time). The global
path also reads the norms back every sweep.

**Validate (online).** Jacobi stopping criteria in the literature / LAPACK (`dsyevj`,
Demmel–Veselić relative criterion `|a_pq| ≤ tol·√(a_pp a_qq)`, cuSOLVER `syevj` tolerance
default = machine ε), and the expected off-diagonal floor in finite precision.

**Then implement.** Tolerance floor depending on `F` and `n` (or the Demmel–Veselić per-pair
criterion, which also keeps small eigenvalues accurate for whitening); named constants, no
literals. Check sweep counts on n = 384 before/after.

## 6. 32-bit device index guard + README warning

**Problem.** Kernels index with `u32` (WebGPU has no 64-bit integers). Only dsp-view checks.
Everywhere else `ch * num_samples` etc. wrap silently past 2³² elements (e.g. 384 channels ×
~11 M samples, reachable with a large `batch_size`). 59 unsafe launches, 29 `SAFETY` comments;
on CUDA an out-of-bounds write corrupts memory silently.

**Do.**
- One guard in dsp-core (`compute`): a named constant for the largest element count a device
  buffer may index (`u32::MAX`), checked where buffers are sized (`dsp_base::core::buffer`
  `empty` / `upload`, `PipelineWorkspace::reserve`, sorter/detector constructors), returning a
  clear error naming the shape, not a panic deep in a launch.
- Also guard products computed inside kernels that exceed the buffer length (e.g.
  `channels · n_templates · samples` in the Kilosort4 detector, `(row · samples + t)`).
- `SAFETY` comment on every unsafe launch.
- README top warning (and mdBook introduction): device buffers are indexed with 32-bit
  integers, so one window / batch can hold at most 2³² − 1 values (≈ 4.29 · 10⁹, e.g. 384
  channels × 11.1 M samples); larger requests are rejected. Explain why (WebGPU portability).

## 7. HDBSCAN scaling — recommendation (awaiting decision)

Keep it **exact** (matches upstream; no clip cap, as decided 2026-10-06). Mutual reachability
`MRD(i, j) = max(core_i, core_j, d_ij)` is always `≥ core_i`; that lower bound gives exact
pruning:

- **a. Seed Borůvka from the k-NN pass.** Keep the neighbour indices of the core-distance pass
  (`[n, k]`, not only distances). For a neighbour `j`, `d_ij ≤ core_i`, so `MRD = max(core_i,
  core_j)`. If any neighbour in another component has `core_j ≤ core_i`, the edge weight equals
  the lower bound `core_i`: provably the cheapest, no full scan for that point (ties: all points
  with `d_ij ≤ core_i` are the neighbour list, so the tie order (weight, lo, hi) can be decided
  there; handle distance ties at the k-th neighbour by including them). Typically most points of
  dense clusters resolve this way in round 1, the most expensive round.
- **b. Per-component bound in later rounds.** A point can only improve its component's cheapest
  edge if `core_i < best_w(component)`. After the cheap points answer, points with `core_i ≥`
  their component's current best skip the full scan (compact the active points on the device;
  scan only those). Big late-round components are made mostly of such points.
- **c. Launch sizing by time, not terms.** Replace `PAIR_TERMS_PER_LAUNCH = 2³²` by a size
  measured on the device (first launch timed, then scaled to a target below the 2 s Windows
  TDR), so slow integrated GPUs are safe.
- **d.** `kbest` in `core_distance_tile_kernel` (runtime-indexed, local memory): for large
  `min_cluster_size` keep the k-best list in shared memory per unit, or use a fixed-size
  register insertion network for small `k` (comptime).

Do a+b+c+d first and measure on the 500 000-clip HD-EMG run. A spatial index on the device
(grid / k-d tree in the low-dimensional PCA space) is the larger step that turns `O(n²)` into
`~O(n log n)`; only if a+b are not enough.

## 8. Smaller kernel inefficiencies

- `correlate_templates_kernel` (`kilosort4/kernels.rs`): all `n_templates` per unit from one
  shared-memory tile of `x`, branch-free interior path.
- `delay_cc_kernel` (`emusort/kernels.rs`): compute `a ≤ b` only (`CC[b, a, lag] =
  CC[a, b, −lag]`), split the sample sum as covariance does (precision).
- Tree reductions (`core/reduce.rs`, `linalg/eigen.rs` norms and shared path,
  `dsp-view/src/envelope/device.rs`): plane (warp/subgroup) reductions for the last levels
  where the runtime supports them, shared memory only across planes. Check CubeCL plane-op
  support per runtime (CPU plane size 1).
- `buffer::zeros`: a fill kernel instead of uploading a host vector of zeros (also HDBSCAN's
  `f32::MAX` initial `best`).

## 9. Reproducible sorts

**Requirement.** The same recording, config and device give the same spikes, run after run.
Today the IIR autotuner (`filter/iir/sos.rs`, block count and layout) and the peak-candidate
autotuner can choose differently between runs, which changes rounding and can flip spikes at
`Th_universal`.

**Do.**
- A run setting (e.g. `Determinism::Reproducible` in dsp-core `compute`) that pins every
  autotuned choice that changes results: IIR `with_block_len` / `with_layout` (both exist),
  and any future tuned kernel that changes rounding. Tuned choices that do not change results
  (peak `block`, which only changes work split) may stay tuned; document which is which.
- Sorters (Kilosort4, EMUsort) use the reproducible setting by default; viewing/streaming keep
  autotuning.
- No atomics with float accumulation in sorter paths (order-dependent sums); check after tasks
  1, 4 and 8.
- k-means / any random choice: seeds come from the config (no silent defaults).
- Record in `SortingOutput` / provenance: runtime, device id (`tune_id`), the pinned settings.
- Scope: reproducible on the same device and runtime; bit-identical across devices is not
  promised (different FMA/rounding), and the docs say so.

**Done when.** Two runs of the playground scripts on the same machine give identical spike
trains (check at test time).

**Done (2026-10-08).** `dsp_core::compute::pin_tuned_choices()` (thread-scoped guard) pins the IIR
split (`PINNED_BLOCK_COUNT` = 16 blocks, channel-major) and the matmul routine (`SimpleUnit`, max
tiles; `direct_matmul_kernel` where unavailable); `Kilosort4Config::reproducible` (default `true`)
holds it for the whole run; `Kilosort4Result::{reproducible, device}` record it (also in Python).
The clustering added since uses only integer atomics (counts), seeded k-means++ (seed from the
learn options) and ordered reductions. Checked: test `pinned_choices_match_the_fixed_split`
(bit-identical); two runs give identical spikes **and unit labels** on the Neuropixels and HD-EMG
test recordings.

## Reported, not acted on: upstream EMUsort differences (user, 2026-10-07)

Documented in the book (`sorters/emusort/pipeline.md`, "Known differences from upstream"): upstream
drops the batch that would overflow the 500k clip buffer (we fill to `MAX_CLIPS`), and de-duplicates
cross-threshold peaks on the time index only (we use (channel, time)). To check later.

## Pending (2026-10-08): EMUsort pipeline vs the paper — decided, not coded yet

Tracked as tasks S6–S8, S11 in `SORTER_TASKS.md` (with everything else missing from both sorters).

From the EMUsort paper (O'Connell et al., eLife 2026, RP110417; Table 5, Methods). Notes in the book:
`sorters/emusort/tuning.md` ("Notes on our implementation"). **Do not change code until told.**

- **Filtering (decided by user):** one band-pass **300–5000 Hz** instead of upstream's cascade
  (SpikeInterface 250–5000 Hz, then Kilosort4's 300 Hz high-pass: the two high-passes are redundant,
  only the 5000 Hz low-pass adds anything). Keep the low-pass edge below Nyquist (drop it at low rates).
- **Notch (decided: add):** 60 Hz notch as an option (50 Hz outside the Americas). Note: after the
  300 Hz high-pass, 60 Hz is already ~80 dB down; in-band harmonics (300, 360, 420 Hz…) would need a
  comb — only if data shows it.
- **`nt` (decided: keep 121 for the HD-EMG recording):** `nt` counts samples, so it depends on fs
  (5 ms = 121 at 24.4 kHz, 151 at 30 kHz). Proposed: optional window in ms converted per recording
  (`round(ms·fs)`, odd, `nt0min ≈ nt/3`), alongside `nt`. Library default stays 61.
- **Linear channel map (to decide):** EMUsort places channels on a dense line 2 µm apart so spatial
  templates span ~5–25 channels; we use the physical grid (100 µm), where templates collapse onto one
  contact (root cause of the tied detections). Plan: detection and clustering on the linear map,
  positions mapped back onto the real grid for export / inspection.
- **Composite score** (`cluster_score_threshold`): not implemented.
- Also still open from the Kilosort4 work: stage 5 (refractory CCG split criterion, global merges,
  duplicate-spike removal); provenance DOI to update to the eLife reviewed preprint
  (10.7554/eLife.110417.1).

## IIR "from-rest bug" (resolved 2026-10-07, not a kernel bug)

Absolute vs relative error (0.0702 of a ~650 peak = 1.08e-4) from f32 coefficient rounding of
0.5 Hz poles, excited by the from-rest DC step; tests now use relative errors and a named from-rest
low-cutoff limit. Stale scipy fixtures regenerated. `DeviceFilter` checks state/scratch sizes.
Written up in the book (GPU engineering → Pitfalls).
