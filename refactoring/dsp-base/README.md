# refactoring/dsp-base

Items removed from `crates/dsp-base`.

- Branch / commit at removal: `versions/v0.14` @ `0f7e600`
- Build state: `dsp-base` compiles (`cargo check -p dsp-base --tests`, 2026-10-05); downstream
  users of the removed items are broken until the final fix-up pass.

## Parked: visualization code (destination: dsp-app, with a GPU min/max reduction later)

| Parked path | Original path | Reason |
|---|---|---|
| `resampler/` (`mod.rs`, `minmax.rs`, `decimate.rs`, `cache.rs`, `summary.rs`, `summarize.rs`) | `crates/dsp-base/src/resampler/` | Display envelopes: min/max folds, an mmap'd min/max pyramid file written next to the recording, an in-memory summary, a background summarizer thread. Not DSP; the name will be reused for real sample-rate conversion kernels. **Moved on 2026-10-06 to the new crate `crates/dsp-view/`** (see `refactoring/dsp-view/README.md`). |
| `math/ticks.rs` (`nice_step`, `ticks`) | `crates/dsp-base/src/math/ticks.rs` | Axis tick labels (UI). |
| `math/geometry.rs` (`point_in_polygon`) | `crates/dsp-base/src/math/geometry.rs` | Lasso selection (UI). |

## Parked: replaced implementations

| Parked path | Original path | Reason | Replacement |
|---|---|---|---|
| `linalg/svd.rs` (`SymmetricEig`) | `crates/dsp-base/src/linalg/svd.rs` | Host classical Jacobi: counts single rotations as "iterations" (100–120) so it does not converge past ~15 channels, absolute `1e-7` tolerance, `O(n⁴)`. | `linalg::eigen` (device parallel cyclic Jacobi, relative tolerance). |

## Lines removed from files that stayed

| File | Removed |
|---|---|
| `src/lib.rs` | `pub mod resampler;`, `pub use resampler::{min_max_decimate, min_max_decimate_into};` |
| `src/math/mod.rs` | `pub mod geometry;`, `pub mod ticks;`, `pub use geometry::point_in_polygon;`, `pub use ticks::{nice_step, ticks};` |
| `Cargo.toml` | `rayon`, `memmap2` (only the parked resampler used them) |

## Downstream breakage (to fix at the end)

- `dsp-app`: `engine/time/renderer.rs`, `engine/time/view.rs`, `viewmodels/trace.rs`, `store.rs`,
  `engine/data/dataset.rs`, `engine/curation/derived.rs` (resampler, `MinMaxCache`,
  `MinMaxSummary`, `Summarizer`, `Progress`, `min_max_decimate_into`, `BASE`, `DEFAULT_BASE`,
  `peak_to_peak`); `ticks` / `nice_step` (3 files), `point_in_polygon` (1 file).
- `dsp-synapse`: `core/template.rs` uses `resampler::minmax::peak_to_peak` (a one-line min/max fold).
- Note: the parked code has two pyramids with different bases (`summary::BASE = 256`,
  `cache::DEFAULT_BASE = 64`); unify when it lands in dsp-app.

## API / behaviour changes (step 2: `core/` + named constants)

| Change | Downstream action |
|---|---|
| New `dsp_base::core`: `DspFloat`, `buffer::{bytes, empty, upload, zeros, download, truncate}`, `Scratch`, `EdgeMode` + `read_extended` | — |
| `PipelineStage::SubtractBaseline { baseline_uv }` → `{ baseline }` (unit-neutral) | `dsp_kitchen_py/src/pipeline/engine.rs`, `dsp_kitchen_py/src/math/arithmetic.rs` (Python arg `baseline_uv`) |
| Gaussian truncation 3σ → **4σ** (`GAUSSIAN_TRUNCATE`, scipy default) and radius rounding `ceil(t·σ)` → scipy's `int(t·σ + 0.5)` (`gaussian_radius`) | Pipeline settling for `GaussianSmooth` grows accordingly; `gaussian_smooth_1d` results change slightly. |
| `estimate_noise_std` divisor 0.6745 → `MAD_TO_SIGMA = Φ⁻¹(0.75) = 0.67448975` | Results change in the 5th digit. |
| `interquartile_range` uses selection instead of a full sort (same ranks) | — |
| Named constants: `DEFAULT_MAX_LAG`, `MIN_TEMPLATE_ENERGY`, `MAD_TO_SIGMA`, `IQR_TO_SIGMA`, `TRIMMED_SIGMA_REL_TOL`, `MIN_SIGMA`, `MIN_NEIGHBOR_DISTANCE`, `MEDIAN9_RADIUS`, `GAUSSIAN_TRUNCATE`, `MIN_KERNEL_WIDTH` (public); design / window / settling constants (private) | — |

## API / behaviour changes (step 3: generic kernels, edge modes, scipy start)

| Change | Downstream action |
|---|---|
| Every kernel is generic over `F: Float`; every dispatcher over `<R: Runtime, F: DspFloat>` (`execute_scaling::<R, F>`, `execute_fir::<R, F>`, …). Kernels with `F` scalars need `F: Float + CubeElement`. | Callers add `, f32` to turbofish. |
| `DeviceFilter<F = f32>`, `PipelineWorkspace<R, F = f32>`; `Pipeline::execute::<R, F>`. Default type params do not drive inference: `DeviceFilter::<f32>::new`, `PipelineWorkspace::<R, f32>::new`. | `dsp-cli` (`execute_filter`/`DeviceFilter`), `dsp-synapse` (`PipelineWorkspace`, 2 files). |
| Host APIs take `&[F]` (`process_chunk`, `process_chunk_in_vram`); `apply_gpu` / `project_gpu` take `F`. | — |
| `EdgeMode` on stencils: `execute_fir(.., edge)` (default `FIR_DEFAULT_EDGE = Zeros`, lfilter), `execute_fir_centered(.., edge)`, `execute_gaussian_smooth(.., edge)` (default `Reflect`, ndimage), `execute_median(.., width, edge)` (default `Zeros`, medfilt), `execute_teager_kaiser(.., edge)` (default `Reflect`). | Callers pass an edge (or the `*_DEFAULT_EDGE` constant). |
| Gaussian border: clamp → **reflect**; `gaussian_smooth_1d` (host) reflects too. | Output near the edges changes. |
| Median: any odd width ≤ `MAX_MEDIAN_WIDTH` (31); width 9 keeps `med9`. Borders: pass-through → **zero padding** (medfilt). | Edge samples change. |
| Teager-Kaiser: negative energies are **kept** (no `max(0)` floor); end samples use reflected neighbours instead of `x²`. | Detectors relying on non-negative output must rectify. |
| `PipelineStage::Median9p` → `Median { width, edge }` (`PipelineStage::median9()`, `median(w)`); `TeagerKaiser` → `TeagerKaiser { edge }` (`teager_kaiser()`); `GaussianSmooth { sigma_samples }` → `{ sigma_samples, edge }`. | `dsp-cli` (1 file), `dsp_kitchen_py` (4 files). |
| **Forward IIR start**: new `FilterSpec.start: FilterStart { Rest (default), SteadyState }`. Default follows scipy `sosfilt` (at rest); the previous behaviour is `.with_start(FilterStart::SteadyState)`. Stateful streams' first chunk follows it too. Forward-backward unchanged (steady state, as `sosfiltfilt`). | Code building `FilterSpec { design, mode }` literally adds `start`. DC-heavy chunked forward filtering should opt into `SteadyState`. |
| IIR coefficient buffer adds `2n` at-rest states (`coeffs_len`); host computes them from the coefficients rounded to `F`. | — |
| `Sos::svf_coeffs_f32` removed (unused; replaced by the generic coefficient builder). | — |
| Unit tests run on every compiled-in runtime (`runtime_test!` macro over `ComputeTask`); `tests/filter_scipy.rs` and `tests/filter_chunks.rs` are gated on `feature = "wgpu"`. | — |
| **Fixture to regenerate**: `tests/scipy_reference.py` now also writes `forward_rest` (`sosfilt` without `zi`), which `filter_scipy.rs` requires. Needs `scipy` in the venv (not installed): `uv pip install scipy numpy && python tests/scipy_reference.py`. | End-pass task. |

## Additions (step 4: missing kernels)

| Added | Notes |
|---|---|
| `dsp_core::compute::LaunchGeometry::per_row` + `row_position` (dsp-core) | One cube per row, power-of-two cube for tree reductions. |
| `core::reduce::{row_mean_std, row_min_max}` | Parallel per-row reductions (Welford per unit + Chan merge in shared memory). `math::execute_channel_mean_std` now uses it; the one-thread-per-channel `channel_mean_variance_kernel` is **deleted**. |
| `linalg::covariance` (`COVARIANCE_SPLIT_SAMPLES = 4096`) | Device covariance (`/ samples`) + means; replaces the four host copies once linalg moves to the device (step 5). |
| `spatial::{SparseRows, DeviceSpatialMatrix, execute_sparse_rows_multiply, SPARSE_MAX_ROW_FILL}` | ELLPACK sparse rows; `DeviceSpatialMatrix` picks sparse when the widest row fills ≤ 50 % of the columns. Pipeline whitening / Laplacian stages and `apply_gpu` use it. |
| `filter::design::chebyshev1_sos`, `FilterDesign::Chebyshev1`, `FilterSpec::chebyshev1`, `FilterError::InvalidRipple` | `scipy.signal.cheby1`; Butterworth and Chebyshev share one design path. |
| `math::{bessel_i0, kaiser_window}` | f64, for FIR design. |
| `resampler::{firwin, FirWindow, resample_poly, resample_poly_len, ResampleFilter, decimate, decimate_len, DecimateFilter, DECIMATE_DEFAULT}` | Real sample-rate conversion (`resample_poly`, `decimate` IIR default / FIR) with `upfirdn_kernel` and `downsample_kernel`. Not a pipeline stage (it changes the sample count). |
| `filter::template::device` — `TemplateFilter::apply_device` | Device template subtraction; overlapping events run in dependency layers so results equal the host's sequential order. |

Still host-only by design: filter / window design, `estimate_noise_*`, histogram / percentile. Host-only until step 5: PCA / PPCA / ICA / whitening fits (they will use `covariance` + the device eigensolver).

## API / behaviour changes (step 5: device eigensolver)

| Change | Downstream action |
|---|---|
| New `linalg::eigen`: `symmetric_eigen`, `symmetric_eigen_batched`, `symmetric_eigen_host`, `EigenOptions` (`EIGEN_RELATIVE_TOLERANCE = 1e-6`, `EIGEN_MAX_SWEEPS = 30`), `SymmetricEigen { n, values (desc, f64), vectors (row-major, column k = vector k, f64) }`. Parallel cyclic Jacobi (round-robin rounds); shared-memory single-launch path when `3·n²` values fit the runtime's `max_shared_memory_size`, else three launches per round with a per-sweep convergence check. | — |
| `SymmetricEig` removed (parked). | `dsp-synapse/src/sorting/gmm.rs:540` (`SymmetricEig::decompose(cov, d, 80)`) → `symmetric_eigen_host::<R, f32>(client, cov_f64, d, EigenOptions::default())` (needs a client) or a host solver in synapse. |
| `PcaModel::fit`, `PpcaModel::fit`, `PpcaModel::fit_em_masked`, `FastIcaModel::fit`, `SpatialWhitening::fit_zca`, `SpatialWhitening::fit_local_knn` now take `client: &ComputeClient<R>` first and `<R, F>`; data stays `&[f32]` on the host (uploaded). Covariance and eigendecomposition run on the device; local k-NN whitening solves all neighbourhoods in one batch. | `dsp-synapse/src/features/mod.rs` (3), `dsp-synapse/src/sorting/cbss.rs`, `dsp-synapse-ml/examples/emusort_nwb_zarr.rs`, `dsp_kitchen_py/src/linalg/{pca,ppca,ica}.rs`, `dsp_kitchen_py/src/spatial/whitening.rs`. |
| Precision: covariance and eigendecomposition now in `F` (f32 by default) instead of f64 host accumulation; PCA previously accumulated in f32 anyway. | Results differ in low digits. |
| Named: `ppca::{MIN_NOISE_VARIANCE, FULL_RANK_NOISE_VARIANCE}`, `ica::MIN_WHITENING_EIGENVALUE`, `whitening::MIN_WHITENING_EPSILON`. | — |
| Still on the host: FastICA fixed-point iterations, PPCA EM projection / reconstruction, ZCA matrix assembly from the eigenpairs (`O(n³)`). | Candidates for later kernels. |

## Optimizations (step 6)

| Change | Notes |
|---|---|
| IIR: block start states and staging buffers reused per filter and per thread (`PassWorkspace` in `Scratch`) instead of allocated every pass. | Keyed by thread: CubeCL streams are per thread, so clones on several threads never share a buffer. |
| IIR: time-major pass (`PassLayout::TimeMajor`: transpose → strided kernel → transpose) as autotune candidates next to channel-major, at every block count. `DeviceFilter::with_layout` pins one. | Kernel takes channel / time strides; `core::read_extended_strided`, `core::layout::transpose` added. The autotuner measures which layout wins per device and size. |
| FIR: shared-memory tiled kernels (causal and centered) as autotune candidates next to the direct ones; `execute_fir_with` / `execute_fir_centered_with` + `FirKernel` pin one. | Tiled only where `(tile + halo) · rows · size_of::<F>()` fits `max_shared_memory_size`, else direct. Tap count / radius are comptime in the tiled kernels (one compilation per filter length). |
| Stored samples: `math::upload_stored` uploads whole-word byte buffers directly (no host repack); odd tails still padded. `PipelineWorkspace::process_stored_chunk_in_vram` uses it. | Assumes little-endian devices (all CubeCL targets). |
| Weights uploaded once: `PcaModel::to_device` / `PpcaModel::to_device` → `linalg::DeviceProjection`; `SpatialWhitening::to_device` / `SurfaceLaplacian::to_device` → `DeviceSpatialMatrix`. `project_gpu` / `apply_gpu` stay as one-off conveniences. | — |
| `Pipeline::execute` **unchanged** (still one-off; documented). A hidden cache in `Pipeline` would need to tell devices apart (the workspace holds its client's buffers) and CubeCL clients expose no reliable identity; `PipelineWorkspace` is the reuse path. | Callers running many chunks should hold a `PipelineWorkspace`. |
| Tests: `filter_blocks` covers both IIR layouts; FIR unit test compares tiled vs direct for every edge mode; `kernel_runtimes` covers both stored-upload paths. | — |

## Parked: unused items (2026-10-05, after step 7)

No users anywhere in the workspace (only `tests/kernel_runtimes.rs` exercised the baseline kernel; that
check was removed).

| Parked path | Original location | What |
|---|---|---|
| `math/baseline.rs`, `math/kernels/baseline.rs` | `crates/dsp-base/src/math/` | `execute_baseline_subtract` + `baseline_subtract_kernel` (per-channel DC offsets). The pipeline's `SubtractBaseline` uses the scaling kernel with a constant. |
| `spatial/reference.rs` | `crates/dsp-base/src/spatial/reference.rs` | `SpatialReferenceConfig` (channel mask for referencing) — never read; CAR ignores masks. |
| `spatial/car_precomputed.rs` | extracted from `spatial/car.rs` + `spatial/kernels/car.rs` | `execute_car` + `subtract_common_average_kernel` (CAR with a precomputed average). `execute_direct_car` stays. |
| `filter/fir/causal_kernels.rs` | extracted from `filter/fir/gaussian.rs` | `causal_exponential_kernel_1d`, `causal_alpha_kernel_1d`. |

Exports removed: `math::execute_baseline_subtract`, `math::kernels::baseline_subtract_kernel`,
`spatial::{execute_car, subtract_common_average_kernel, SpatialReferenceConfig}`,
`filter::{causal_alpha_kernel_1d, causal_exponential_kernel_1d}`.

## Addition — host Cholesky (2026-10-05, dsp-synapse step 2)

`src/linalg/cholesky.rs`: `cholesky`, `spd_inverse_logdet`, `cholesky_solve` (host, f64, exact).
Used by dsp-synapse GMM (covariance inverse / log-det) and kriging (kernel solve), replacing the
parked 80-rotation `SymmetricEig` and a hand-written Gauss–Jordan solver. Exported from `linalg`.

## Addition — `peaks/` (2026-10-05, dsp-synapse step 3a)

`src/peaks/{mod, host, device, kernels}.rs`: scipy `find_peaks` (host), `local_extrema`,
`select_by_distance`, `DistanceRule { Scipy, LocallyExclusive }`, device `find_peak_candidates`
(from dsp-synapse `detection/kernels/threshold.rs`, made generic over `F`, both polarities,
per-channel heights). Exported as `dsp_base::peaks`.

## Additions — synapse step 3b (2026-10-05)

`math/xcorr.rs` (lagged cross-correlation, `peak_lag`, `parabolic_vertex_offset`),
`math/moments.rs` (`RunningMoments`), `math/stats.rs::peak_to_peak`, `resampler/fractional.rs`
(fractional-delay windowed sinc, host + device taps kernel); `math/windows.rs` Blackman-Harris
coefficients and `SINC_ZERO` made public named constants.

## Additions — synapse step 4 (2026-10-05)

`core/reduce.rs`: `row_abs_kth` / `row_abs_kth_kernel` (per-row k-th smallest `|x|` by value
bisection, exact sample value; `ROW_SELECT_ITERATIONS = 64`). `math/stats.rs`:
`execute_channel_noise_std` (device MAD noise per channel). Closes the "median-based noise
estimators on the host" open item for the MAD estimator.

## 2026-10-06 — `row_min_max` moved to dsp-view

`core::reduce::{row_min_max_kernel, row_min_max}` had no users. They are removed here and
replaced by dsp-view's segmented `envelope_kernel` / `envelope_on_device` (min/max per column
of each row; a whole row is one column). dsp-view does not depend on dsp-base (user). The
min / max part of `test_row_reductions_match_host` went with it. `cargo check -p dsp-base
--all-targets` passes (three pre-existing warnings, in `sos.rs`, `conv.rs` and
`peaks/device.rs`).


## 2026-10-06 — device whitening statistics, review fixes

- `core/reduce.rs` `row_abs_kth_kernel`: race fixed (a `sync_cube()` between every unit reading
  `lo = vals[0]` and unit 0 overwriting `vals[0]`). Affects `execute_channel_noise_std`.
- `linalg/covariance.rs`: the pair-sum kernels take a column range of `[channels, row_len]` rows
  and can skip centring / accumulate; new `SecondMomentAccumulator` (mean over batches of
  `X Xᵀ / n`, on the device, one download). `covariance` / `covariance_of_host` unchanged.
- Removed `PipelineWorkspace::download_interior` (download → upload round trip; use device column
  ranges), `PipelineStage::car()` alias and the unused `spatial::CommonAverageReference` struct
  (`PipelineStage::CommonAverageReference` / `common_average_reference()` remain).
- The crate-wide `#![allow(semicolon_in_expressions_from_non_local_macros)]` is now on the three
  `local_tuner!` statics (cubecl-runtime 0.10 macro, rust-lang/rust#79813).

## 2026-10-06 — device-resident candidates, partial reads

- `peaks`: `find_peak_candidates_on_device` → `DevicePeakCandidates` (sample, value and channel
  buffers left on the device; only per-channel counts read back). `find_peak_candidates` is it
  plus the download and per-channel ordering. `write_peak_candidates_kernel` also writes each
  candidate's channel.
- `core::buffer::download_prefix(client, handle, len)`: reads only the first `len` elements;
  `PipelineWorkspace::process_chunk` uses it.
- `SecondMomentAccumulator` keeps its partial-sum buffer between batches.
- `buffer::download_range(client, handle, start, len)`; `SecondMomentAccumulator::into_sum` (the
  device sum, nothing read back).
