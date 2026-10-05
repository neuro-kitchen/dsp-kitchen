# dsp-base

> **Status:** cleaned on 2026-10-05 (`versions/v0.14`) in six steps: visualization code parked,
> `core/` and named constants, generic kernels with scipy edge modes, missing kernels, device
> eigensolver, autotuned optimizations. Extended the same day with the generic primitives the
> dsp-synapse review found re-implemented there: peak finding, lag cross-correlation, fractional
> delay, running moments, Cholesky. `cargo check -p dsp-base --tests` passes; **only the Cholesky
> tests have been run** (tests and benchmarks wait for the final pass). Changes are tracked in
> `refactoring/dsp-base/README.md`.

## Intent

Deterministic signal-processing primitives for multi-channel signals, running on any CubeCL runtime:
filters, resampling, spatial operators, linear algebra and statistics, and a pipeline that chains
them on the device without host round-trips between steps.

### Owns
- Filter design (host, f64) and filtering kernels (IIR, FIR, non-linear, template subtraction).
- Sample-rate conversion (`resample_poly`, `decimate`) and fractional-delay interpolation.
- Peak finding (`scipy.signal.find_peaks` semantics) on the host and candidate compaction on the
  device.
- Spatial operators (CAR, whitening, Laplacian; dense or sparse rows).
- Linear algebra on the device (covariance, symmetric eigendecomposition, PCA / PPCA / ICA fits and
  projections).
- Reductions and statistics (noise estimators, running moments, lag cross-correlation); small
  dense SPD solves on the host (Cholesky).
- The device pipeline (`Pipeline`, `PipelineWorkspace`).

### Must not contain
- Visualization (min/max display envelopes, display caches, axis ticks, lasso geometry): parked for
  `dsp-app`.
- File formats or recordings I/O (`dsp-io`).
- Domain models (spikes, units, probes): neuro layers above.
- Runtime selection or launch geometry (`dsp-core::compute`); algorithms take a `ComputeClient<R>`
  and never inspect the device.

## Features

| Feature | Default | Enables |
|---|---|---|
| `wgpu` | yes | `dsp-core/wgpu` |
| `cpu` / `cuda` / `hip` | no | the matching `dsp-core` runtime |

## Kernel conventions

Every device algorithm in this crate follows the same rules:

1. **Generic float.** Kernels take `F: Float` (`F: Float + CubeElement` when a scalar argument is of
   type `F`); dispatchers take `<R: Runtime, F: DspFloat>`. Coefficients are designed in f64 on the
   host and rounded to `F` once (`core::cast`). Types that hold device data default to `f32`
   (`DeviceFilter<F = f32>`, `PipelineWorkspace<R, F = f32>`), but defaults do not drive inference:
   callers write `::<f32>`.
2. **Typed buffers.** Sizes come from the element type (`core::buffer`), never a literal byte count.
   Reused device buffers go through `core::Scratch`.
3. **Geometry from the runtime.** Launches use `dsp_core::compute::LaunchGeometry`
   (`elementwise`, `channels_samples`, `per_sample`, `per_channel`, `per_row`). Paths that depend on
   resources (shared-memory tiles, the shared eigensolver) are chosen from the runtime's
   `max_shared_memory_size`, never from the device type.
4. **Edges are explicit.** Every stencil takes an `EdgeMode` (`Zeros`, `Odd`, `Reflect`, `Nearest`),
   read through `core::read_extended` / `read_extended_strided` (bounds-safe; one reflection deep, then
   the far end is held). Defaults follow the scipy function each kernel mirrors.
5. **Measured choices are autotuned.** Where the fastest variant depends on the device (IIR block
   count and memory layout, FIR direct vs tiled), the variants are CubeCL autotune candidates keyed by
   device, element type, shape and size class. `with_block_len` / `with_layout` / `*_with(..., kernel)`
   pin one for tests.
6. **Named constants.** Tolerances, defaults and floors are named constants in the module they
   belong to; no unexplained literals in library code.
7. **Tests on every runtime.** Unit tests use `runtime_test!` (a `ComputeTask` over
   `ComputeTarget::available()`); integration tests that still construct WGPU directly are gated on
   `feature = "wgpu"`.

## scipy correspondence

| dsp-base | scipy | Notes |
|---|---|---|
| `butterworth_sos`, `chebyshev1_sos`, `notch_sos` | `butter`, `cheby1`, `iirnotch` (`output="sos"`) | Same prototype → band transform → bilinear → `zpk2sos(pairing="nearest")` path. |
| `FilterMode::Forward` + `FilterStart::Rest` (default) | `sosfilt(sos, x)` | `FilterStart::SteadyState` = `sosfilt(sos, x, zi=sosfilt_zi·x[0])`. |
| `FilterMode::ForwardBackward` (default mode) | `sosfiltfilt(padtype="odd")` | Pad length = settling length (longer than scipy's default `padlen`). |
| `execute_fir` (`Zeros`) | `lfilter(taps, 1, x)` | |
| `execute_gaussian_smooth` (`Reflect`, `GAUSSIAN_TRUNCATE = 4`) | `ndimage.gaussian_filter1d` | Radius `int(4σ + 0.5)`. |
| `execute_median` (`Zeros`) | `signal.medfilt` | Odd widths ≤ 31; width 9 uses the exact `med9` network. |
| `firwin` | `firwin(..., window=…)` | Hamming, Kaiser. |
| `resample_poly` (`Zeros`, Kaiser β = 5) | `resample_poly` | Same filter length, scaling and output alignment. |
| `decimate` (`DECIMATE_DEFAULT`: Chebyshev I, order 8, 0.05 dB) | `decimate(ftype="iir", zero_phase=True)` | FIR option = `decimate(ftype="fir")`. |
| `estimate_noise_std` | `median(|x|) / Φ⁻¹(0.75)` | |
| `peaks::find_peaks` (`DistanceRule::Scipy`) | `signal.find_peaks` | `height`, `threshold`, `distance`, `prominence` (`wlen`), `width` (`rel_height`); plateaus → middle sample; extra: `Polarity::{Negative, Both}`, `DistanceRule::LocallyExclusive`. |
| `math::cross_correlation(x, y, L)` | `signal.correlate(y, x, "full")` at lags `−L..=L` | |
| `RunningMoments::variance(ddof)` | `numpy.var(..., ddof)` | |
| `math::peak_to_peak` | `numpy.ptp` | 0 for an empty slice. |

## Module map

```text
dsp-base/src/
├── core/                 shared device building blocks
│   ├── float.rs          DspFloat, cast / cast_all / cast_f32 / to_f64
│   ├── buffer.rs         bytes, empty, upload, zeros, download, truncate
│   ├── scratch.rs        Scratch (grow-only reusable buffer)
│   ├── edge.rs           EdgeMode, read_extended, read_extended_strided
│   ├── layout.rs         transpose
│   └── reduce.rs         row_mean_std, row_min_max (one cube per row)
├── filter/
│   ├── design/           FilterSpec / FilterDesign / FilterBand / FilterMode / FilterStart,
│   │                     Butterworth + Chebyshev I (butterworth.rs), notch, Sos (settling, host reference)
│   ├── iir/              DeviceFilter, PassLayout, execute_filter; SVF block-parallel kernel + scan
│   ├── fir/              execute_fir(_centered)(_with), FirKernel, Gaussian kernels; direct + tiled kernels
│   ├── non_linear/       execute_median, execute_median_9p, execute_teager_kaiser
│   └── template/         TemplateFilter (host) + apply_device (layered device kernels)
├── resampler/            firwin, FirWindow, resample_poly, decimate; upfirdn + downsample kernels;
│                         fractional.rs (fractional delay, host + device taps)
├── peaks/                find_peaks, local_extrema, select_by_distance, DistanceRule (host);
│                         find_peak_candidates + count / scan / write kernels (device)
├── spatial/              CAR, SpatialWhitening, SurfaceLaplacian, SparseRows, DeviceSpatialMatrix
├── linalg/               covariance, eigen (parallel Jacobi), cholesky (host), PcaModel, PpcaModel,
│                         FastIcaModel, DeviceProjection
├── math/                 scaling, clamp, unpack (stored → scaled), stats, histogram,
│                         windows (incl. Kaiser, bessel_i0), xcorr, moments
└── pipeline/             PipelineStage, Pipeline, PipelineWorkspace, ChunkMode
```

## Reference

### `core`
| Item | Purpose |
|---|---|
| `DspFloat` | `Float + CubeElement`: the element type of dispatchers. |
| `buffer::*` | Typed allocate / upload / download / view. |
| `Scratch` | Device buffer reallocated only when a call needs more. |
| `EdgeMode`, `read_extended(_strided)` | One edge policy for every stencil. |
| `layout::transpose` | `[rows, cols]` → `[cols, rows]`. |
| `reduce::row_mean_std`, `row_min_max` | Parallel per-row reductions (Welford + Chan merge in shared memory). |
| `reduce::row_abs_kth` | Per-row k-th smallest `|x|` over a column range (value bisection, exact sample value). |

### `filter`
| Item | Purpose |
|---|---|
| `FilterSpec` | Design + mode + start; `bandpass`, `highpass`, `lowpass`, `bandstop`, `notch`, `chebyshev1`, `sos`, `with_order`, `with_mode`, `with_start`, `design`, `settling`. |
| `Sos` | Sections, `validate`, `zi`, `settling_samples`, host `filter` / `filtfilt` (f64 references). |
| `DeviceFilter<F>` | Uploaded design; `apply` (independent chunk), `apply_stateful` (stream), `state_len`, `scratch_len`. Cascades of trapezoidal SVF sections (accurate in f32 next to `z = 1`), split into time blocks joined by a state scan. |
| `execute_fir`, `execute_fir_centered` | Causal / centered FIR; autotuned direct vs tiled. |
| `gaussian_kernel_1d`, `gaussian_radius`, `execute_gaussian_smooth`, `gaussian_smooth_1d` (host reference) | Gaussian smoothing. |
| `execute_median`, `execute_median_9p` | Running median (registers; `med9` for width 9). |
| `execute_teager_kaiser` | `x² − x₋x₊` (negative values kept). |
| `TemplateFilter` | Aligned, scaled template subtraction at events; `apply_1d` / `apply_multichannel` (host), `apply_device`. |

### `resampler`
| Item | Purpose |
|---|---|
| `firwin`, `FirWindow` | Windowed-sinc low-pass design. |
| `resample_poly`, `resample_poly_len`, `ResampleFilter` | Rational resampling. |
| `decimate`, `decimate_len`, `DecimateFilter`, `DECIMATE_DEFAULT` | Integer down-sampling. Not a pipeline stage (changes the sample count). |
| `fractional_delay`, `fractional_delay_taps` | `x(t + shift)` for `|shift| ≤ ½` with a Blackman-Harris windowed sinc normalized to sum 1 (taps at shift 0 are the identity). |
| `fractional_delay_taps_device`, `windowed_sinc_weight` | The same taps computed on the device, once per shift (e.g. once per spike). |

### `peaks`
| Item | Purpose |
|---|---|
| `find_peaks(x, polarity, &PeakOptions)` → `Peaks` | scipy `find_peaks`, conditions applied in scipy's order (height, threshold, distance, prominence, width); properties returned for the conditions used. |
| `PeakOptions`, `Interval`, `Polarity` | Conditions as `min` / `max` intervals; maxima, minima (maxima of `−x`) or both (conditions on magnitude). |
| `DistanceRule` | `Scipy` (default): largest first, removed peaks remove nothing — a chain can reach beyond `distance`. `LocallyExclusive`: kept unless a larger peak is nearer than `distance` — depends only on neighbours, so chunked processing with `distance` of context equals the whole-signal result. |
| `local_extrema`, `select_by_distance` | The building blocks, for detectors that score candidates differently. |
| `find_peak_candidates::<R, F>` → `PeakCandidates` | Device: local extrema above a per-channel height, counted, scanned and compacted on the device (autotuned block length) so only candidates are downloaded; plateaus → first sample. |

### `spatial`
| Item | Purpose |
|---|---|
| `execute_direct_car` | Common average reference in one pass (mean over channels per sample). |
| `SpatialWhitening` | `fit_zca`, `fit_local_knn` (batched device eigendecompositions), `to_device`, `apply_gpu`, `apply_cpu`. |
| `SurfaceLaplacian` | `from_grid_2d`, `from_coordinates_knn`, `to_device`, `apply_gpu`, `apply_cpu`. |
| `DeviceSpatialMatrix`, `SparseRows` | Uploaded `[C, C]` operator, sparse rows when the widest row fills ≤ `SPARSE_MAX_ROW_FILL`. |

### `linalg`
| Item | Purpose |
|---|---|
| `covariance`, `covariance_of_host` | Device covariance (`/ samples`) and means, sample-split for parallelism. |
| `symmetric_eigen(_batched / _host)`, `EigenOptions`, `SymmetricEigen` | Parallel cyclic Jacobi; relative tolerance; shared-memory or global path by size. |
| `PcaModel`, `PpcaModel`, `FastIcaModel` | Fits take a client (device covariance + eigensolver); `to_device` → `DeviceProjection`. |
| `cholesky`, `spd_inverse_logdet`, `cholesky_solve` | Host, f64, exact: SPD factor, inverse with `ln det`, multi-RHS solve (`None` when not positive definite). |

### `math`
`execute_scaling`, `execute_clamp`, `execute_unpack_stored` +
`upload_stored`, `execute_channel_mean_std`, `execute_channel_noise_std` (MAD σ per channel on the
device, only the σ downloaded); host: `estimate_noise_std` / `_rms` / `_trimmed`,
`interquartile_range`, `peak_to_peak`, `standard_error`, `percentile`, `histogram`, `bin_centers`,
windows (`hann`, `hamming`, `blackman`, `gaussian`, `kaiser`, `sinc`, `bessel_i0`; Blackman-Harris
coefficients are public constants shared with device code), `cross_correlation` / `lagged_dot` /
`peak_lag` / `parabolic_vertex_offset`, `RunningMoments` (f64 Welford push, Chan merge,
`variance(ddof)` / `std(ddof)`).

### `pipeline`
| Item | Purpose |
|---|---|
| `PipelineStage` | `Scale`, `SubtractBaseline`, `Clamp`, `Filter`, `CommonAverageReference`, `SpatialWhitening`, `SurfaceLaplacian`, `GaussianSmooth`, `Median`, `TeagerKaiser`; `settling(fs)`. |
| `Pipeline` | Stage list; `settling`, `validate`, one-off `execute`. |
| `PipelineWorkspace<R, F>` | Designs, weights and ping-pong buffers kept on the device; `process_handle`, `process_chunk(_in_vram)`, `process_stored_chunk_in_vram`; `ChunkMode::{Independent, Stateful}`. |

## Open items

- **Run everything**: unit and integration tests on every available runtime; first autotune runs
  decide the IIR layout / block count and FIR kernel per device.
- **Regenerate the scipy fixture** (`tests/scipy_reference.py`, now also writes `forward_rest`);
  needs scipy in the venv.
- **Downstream fixes** (`dsp-synapse`, `dsp-synapse-ml`, `dsp-cli`, `dsp_kitchen_py`, `dsp-app`): see
  `refactoring/dsp-base/README.md`.
- **Still on the host**: FastICA iterations, PPCA EM projection / reconstruction, ZCA assembly from
  eigenpairs, the trimmed / IQR noise estimators (the MAD estimator has a device form).
- Parked (no users): per-channel baseline kernel, precomputed-average CAR, causal exponential /
  alpha kernels, `SpatialReferenceConfig` — see `refactoring/dsp-base/README.md`.
- `filter/template/subtraction.rs` keeps its own full-overlap lag search (a different operation
  from `math::cross_correlation`).
- `butterworth.rs` now holds the shared IIR design path (Butterworth and Chebyshev I); a rename to
  `iir_design.rs` would read better.
