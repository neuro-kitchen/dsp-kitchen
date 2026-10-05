# dsp-synapse review notes (working file)

Started 2026-10-05 on `versions/v0.14`. Review only — no code changes yet. Each finding records the
file and line so it can be verified before acting.

## Inventory

- ~12.5k lines, 11 modules: core, detection, extraction, features, metrics, probe, sorting, spatial,
  storage, streaming (+ lib.rs), 3 integration tests + a SpikeInterface reference script.
- Dependencies: dsp-core (compute), dsp-base, **dsp-stream (not reviewed yet)**, cubecl, serde,
  serde_json, thiserror, tracing, **zarrs (direct)**.

## Decisions (2026-10-05)

- Keep both sorting models: `SortingOutput` (modern, SpikeInterface-style, stored as Zarr) and
  `PhySorting` (legacy Phy2 folders).
- Rename misnamed algorithms to what they implement (no re-implementation of OMP / isocut /
  Gauss–Newton dipole for now).
- Cleanup order agreed (structure → bugs → primitives to dsp-base → device → names/docs →
  constants).
- dsp-stream reviewed: synapse only used `PrefetchReader` (local I/O) → moved to dsp-io; synapse now
  depends on dsp-io instead of dsp-stream (`Cargo.toml` + one import in `streaming/runner.rs`).

## Summary (read first)

### Bugs (verified by reading; none run yet) — all 8 fixed 2026-10-05, see README step 2
1. **GMM** inverts covariances with the old non-convergent eigensolver (80 rotations) → wrong
   likelihood / BIC for `Full` / `Masked` beyond ~12 dims (G1).
2. **Dipole localizer** reads the snippet centre (`num_samples / 2`) instead of the trough at `pre`
   → 0.5 ms off with default config (SP1).
3. **Dedup "deeper"** assumes negative troughs → wrong survivor for positive / both polarity (DD2).
4. **NEO & matched filter** drop early spikes on every channel after the first (shared `events`
   vs per-channel `last_spike`) (D2).
5. **Drift** reference = time bin 0 only → zero drift when bin 0 is empty (DR1).
6. **`compute_morphology`** runs across channel boundaries on multi-channel snippets (F4).
7. **`PhySorting::to_sorting_output`** uses the last spike as recording length (SG2).
8. **IsoSplit** dead code and stale-centroid logic (I2, I3); **kriging** solver skips singular pivots
   silently (K1).

### Not using CubeCL / dsp-base properly
- Host-only where device work is natural: all detectors except threshold, all localizers (per-spike
  LM / grid — embarrassingly parallel), GMM EM, density peaks, silhouette, kriging application,
  noise calibration (thread per channel), PCA embedders' projection.
- Device code that exists is f32-only, uses raw `client.empty(n * 4)` instead of dsp-base
  `core::buffer`, hard-codes WGPU in tests, and two APIs choose the device themselves (OMP,
  streaming `run`).
- Re-implemented dsp-base primitives: TKEO (`compute_neo_energy_1d`), FIR correlation (matched
  filter), peak-to-peak (×3), Gaussian smoothing (×2), histogram (×2), dense solvers (×3), Chan /
  Welford merge (×4 mean/SD implementations with inconsistent SD conventions), windowed sinc (×3,
  device copy with inline Blackman-Harris literals), lag cross-correlation (×4), KNN (×5).

### Belongs elsewhere
- **dsp-io (phase 3)**: Phy / NWB-units / `.sorting.zarr` schema code in `storage/`, `DenseTemplates`
  axis packing, NWB ragged pack / unpack in `SortingOutput`, probe presets (`probe/`), `RecordingMeta`.
- **dsp-base** (generic DSP that synapse needs and others could use): scipy-style `find_peaks` with
  `distance` (would replace the 5 greedy peak pickers), lag cross-correlation, epoch / window
  averaging (STA, templates), fractional-delay sinc interpolation, small dense SPD solves (Cholesky).
- **dsp-stream**: to review (synapse depends on its `PrefetchReader`).

### Naming / documentation that misstates the algorithm
"OMP" (is matching pursuit), "IsoSplit" (is a KDE valley heuristic), "Gauss-Newton" dipole (is a
compass search), "GridConvolution matches SpikeInterface" (different method), "Fisher LDA" `d'` /
"Mahalanobis" isolation distance (diagonal approximations), "Hungarian" comparison (greedy only),
"k-means++" GMM init (farthest-first from an outlier), wavelet "Daubechies-4 / Lilliefors" (Haar +
IQR), "Lanczos" margin (Blackman-Harris), `StreamingSpikeRunner` "spike sorting" (no clustering).

### Hidden constants / fabricated values
Dozens across every module (listed per section): unit-dependent thresholds (`min_explained_energy
= 500` µV²), physical µm assumptions in localizers, fallback noise `10.0` µV, and fabricated outputs
(`0.0`, `1.0`, `[0,0,0]`, `[0,0,15]`) where NaN / `None` is correct.

### Solid parts to keep
Spatial dedup (exact, order-independent, streaming), device threshold detection (count → scan →
compact, autotuned), streaming runner architecture, `metrics::firing` and `correlogram` (faithful
SpikeInterface ports), `QualityCriteria`, density peaks, monopolar LM with variable projection.

## Findings

### Manifest
- M1 `Cargo.toml`: `cubecl = { features = ["wgpu"] }` forces WGPU on; there is no `wgpu` feature, so a
  CPU-only / CUDA-only build still compiles WGPU. Inconsistent with dsp-core / dsp-base feature model.
- M2 `zarrs` used directly (storage); dsp-io's `container::zarr` now owns this (phase 2 moved
  `zarr_store.rs` there already).
- M3 No dependency on `dsp-io` — needed for probe geometry (`SensorLayout` now in `dsp_io::neuro::probe`)
  and containers.

### lib.rs
- L1 Compatibility aliases: `pub use core::{bands, traits}`, `metrics::correlogram`, `sorting as
  clustering`, `sorting as matching`, `spatial as localization`, `spatial as motion`, plus a merged
  `pub mod kernels` re-exporting four kernel modules. Same pattern removed from dsp-core. Check who
  uses the alias paths before removing.
- L2 ~70 flat root re-exports; makes the module structure irrelevant to callers.

### core/ (domain types)
- C1 `traits.rs`: stage traits (`SpikeDetector`, `SpikeMatcher`, …) take host `&[f32]`; no device
  buffer path. `PeakLocalizer` uses `dsp_core::SensorLayout` (moved → `dsp_io::neuro::probe`).
- C2 `events.rs`: `peak_amplitude_uv` naming — neuro layer may keep µV, but it should come from the
  recording's `SignalUnit`, not be assumed.
- C3 `snippets.rs`: two representations (`WaveformSnippet` per spike with own `Vec`s, contiguous
  `SnippetBatch`) with copying conversions. `from_snippets` silently drops mismatched snippets.
- C4 `template.rs`:
  - `compute_mean_template` host-only; `streaming/kernels/template_reduce.rs` reduces templates on
    the device → probable duplicate (verify).
  - `DenseTemplates` pack/unpack of Phy `[n, samples, channels]` vs Zarr/NWB `[n, channels, samples]`
    is file-schema work (belongs with dsp-io phase 3).
  - `unpack_unit` invents `std = 1.0` when the file has none (hidden default).
  - `UnitQualityLabel::{Good, Mua}` consts alias the variants (alias cruft).
  - Imports `dsp_base::resampler::minmax::peak_to_peak` (parked) → broken.
- C5 `sorting_output.rs`:
  - `SortedUnit` couples data with metrics: constructors compute ISI / presence / amplitude-cutoff /
    SNR / label with hidden constants: presence bin `(duration/10).clamp(0.5, 60)` s, noise floor
    `sd.max(1e-3)`, fallback noise `10.0` µV when a channel sigma is missing
    (`from_clustered_spikes`).
  - `from_motor_units`: overwrites `snr` with `pnr_db` (dB in a ratio field), labels with hard-coded
    `20.0` dB / CoV `0.35`, `primary_channel = 0`, amplitude fallback `1.0`, noise `1.0`.
  - `flattened_spikes`: amplitude fallback `1.0`, location fallback `[0,0,0]` (fabricated values).
  - Time as `u64` samples + `f64` rate; core now has `SampleRate` / `RationalTime`.
  - NWB ragged pack/unpack (`to_ragged_spikes`, `unpack_ragged_spikes`) and Phy grouping
    (`flattened_spikes`, `group_spikes_by_cluster`) are file-schema helpers → dsp-io phase 3.
  - `probe: Option<SensorLayout>` from dsp_core (moved).

### detection/
- D1 **Duplicated peak picking**: the "local extremum past threshold, then skip refractory" loop is
  written 5 times (`threshold.rs`, `adaptive.rs`, `neo.rs`, `matched_filter.rs`, and the host half of
  `kernels/threshold.rs`). All greedy (first extremum wins, not the largest within the refractory
  window as SpikeInterface `locally_exclusive`).
- D2 **BUG** `neo.rs:57`, `matched_filter.rs:90`: refractory test `events.is_empty() || t > last_spike +
  refractory` — `last_spike` resets per channel but `events` is shared across channels, so from the
  second channel on, a spike within the first `refractory_samples` samples is dropped.
- D3 Host vs device: `threshold.rs` (host, 3 polarities) and `kernels/threshold.rs` (device count →
  scan → compact, autotuned; **negative polarity only**, **f32 only**, raw `client.empty(n *
  size_of)` instead of dsp-base `core::buffer`). Adaptive / NEO / matched filter are host-only.
- D4 **Reimplemented elsewhere**:
  - `compute_neo_energy_1d` = dsp-base `execute_teager_kaiser` (host copy, with a `max(0)` floor
    and copied end samples).
  - matched filter correlation = FIR with reversed taps (dsp-base `execute_fir`); host `O(S·K)` loop;
    last `k_len` samples of `corr` never written (stay 0).
  - `noise.rs` is only a re-export of dsp-base stats (and `mod.rs` re-exports it again).
- D5 Threshold statistics: NEO / matched filter threshold on `estimate_noise_std(psi)` /
  `(corr)` — the MAD-σ rule assumes zero-mean Gaussian samples; ψ ≥ 0, so this is a heuristic scale,
  not a noise σ (document or replace).
- D6 Hidden constants: defaults 4.5 σ / 1 ms / 8.0 (NEO) as literals in `Default` impls; adaptive
  `block_samples.clamp(32, …)`, `max(1e-6)`, `alpha.clamp(0.01, 1.0)`; matched filter prototype
  `0.38`, `len/10`, `len/6`, `max(9)`, `1e-8`.
- D7 `SpikeEvent.peak_amplitude_uv` filled with whatever unit the trace is in.

### detection/dedup
- DD1 Design is sound: exact "locally exclusive" rule, order-independent, streaming form
  (`StreamingDedup`) equal to the whole-recording result; well tested.
- DD2 **BUG (polarity)** `dedup.rs` `deeper()` and `kernels/dedup.rs`: "deeper" = smaller amplitude,
  i.e. assumes negative troughs. For `SpikePolarity::Positive` / `Both` crossings the *smaller*
  positive peak survives. Needs a magnitude / polarity-aware comparison.
- DD3 Device version (`deduplicate_spikes_spatial_gpu`): builds a dense `C×C` distance matrix on the
  host on every call, re-derives `participating_channels` on the host anyway (same `O(n·window)` work
  as the CPU path), and is not chained to device detection (detection downloads events, dedup
  uploads them again). f32 only; raw `client.empty(n * 4)`.
- DD4 Uses `dsp_core::layout::{SensorLayout, Position3D}` (moved to dsp-io).
- DD5 Tests hard-code `WgpuRuntime` without a feature gate.

### extraction/
- E1 **Three windowed-sinc interpolators**: `alignment::interpolate_window` (normalized fixed
  weights, taps must stay in the row), `alignment::resample_sinc_1d` (renormalizes truncated windows
  at edges → different edge behaviour), and the device kernel `kernels/sinc.rs` (same formula as
  `interpolate_window`). The device kernel re-implements Blackman-Harris and sinc inline with literal
  coefficients (`0.35875…`, `1e-7`), duplicating `dsp_base::math::windows` (host only today).
- E2 Device kernel inefficiency: one unit per output value recomputes the parabolic offset and all
  `2r+1` sinc/window weights (3 `cos` + 1 `sin` per tap) — weights depend only on the spike's shift,
  so they could be computed once per spike.
- E3 Host extraction (`extract_snippet_batch_multichannel`) duplicates the device extractor; KNN
  table recomputed per call; `extract_snippets_multichannel` = batch + `to_snippets()` copy;
  `_channels` parameters unused.
- E4 `read_snippets` reads through `RecordingSource` (good), clones the buffer per snippet.
- E5 Hidden constants: `1e-6` (parabolic denominator), `1e-4` / `1e-5` (shift skip thresholds — two
  different values for the same decision), `1e-7`.
- E6 f32 only; `spike_center_samples` as `u32` (window-local, OK while chunks < 4 G samples).

### features/
- F1 Same block three times (`extract_waveform_pca`, `PcaFeatureEmbedder`, `PpcaFeatureEmbedder`):
  transpose snippets to `[features, spikes]`, fit, host `project_cpu`, transpose back. Uses the old
  host `fit` signatures (now take a client; broken until the fix pass).
- F2 Design: PCA over the whole flattened snippet (`channels · samples` features, e.g. 32 × 61 =
  1952-dim covariance + eigendecomposition per embedding). Kilosort-style per-channel temporal PCA
  (or a fixed temporal basis) is far smaller; decide.
- F3 `wavelet.rs` docs promise Haar / Daubechies-4 and a Lilliefors-style score; the code is Haar
  only + IQR ranking. Defaults `8` components / `4` levels as literals.
- F4 **BUG (shape)** `morphology.rs::compute_morphology`: documented "single-channel snippet" but not
  checked; on a multi-channel `WaveformSnippet` it searches the flattened `[channels, samples]` array,
  so trough / following peak / half-width can cross channel boundaries. Also assumes a negative
  trough; half-width at integer samples; field `peak_amplitude_uv` holds the trough.
- F5 `conduction.rs`: host cross-correlation `O(S·L)`; on failure returns `velocity = 0` (fabricated)
  instead of `None` / NaN; hidden thresholds `r > 0.2`, `max_lag = samples/3`, `1e-3`, `1e-8`, `1e-9`.

### metrics/
- MT1 Strong parts: `firing.rs` (ISI violations, presence ratio, amplitude cutoff with scipy-exact
  integer smoothing, Llobet contamination) and `correlogram.rs` (ordered-pair ACG = self-CCG minus
  `i = j`, tested) are careful SpikeInterface ports. `QualityCriteria` names its thresholds (good).
- MT2 Duplicates: `firing.rs` has a private numpy-style `histogram` **and** uses
  `dsp_base::math::histogram` for `isi_histogram` (two edge rules in one file);
  `gaussian_filter1d_nearest_int` is a third Gaussian smoother (host, `nearest` edges).
- MT3 **Window averaging written 3×** with **inconsistent SD**: `core::compute_mean_template`
  (population `/n`), `evoked::compute_stimulus_triggered_average` (Welford f64, sample `/(n−1)`),
  and the device `streaming/kernels/template_reduce.rs` (to check). PSTH SE uses `/(n−1)`.
- MT4 `isolation.rs`: `d'` documented as Fisher LDA and isolation distance as Mahalanobis, both use a
  **diagonal** covariance (approximation of the published metrics). Host `O(N²·D)` silhouette.
  Invalid inputs return fabricated `0.0` / `1.0` instead of NaN (`compute_snr` → 0 when noise ≤ 1e-6).
- MT5 `comparison.rs`: doc says "greedy/Hungarian" assignment — only greedy is implemented. Spike
  matching is a greedy two-pointer (SpikeInterface builds a match-count matrix); can under-count when
  one spike could match two. Empty-vs-empty returns agreement `1.0`.
- MT6 `rate.rs`: firing rate uses dsp-base `gaussian_smooth_1d` (edges now `reflect`, conserves
  count better than the old clamp); hidden floors `1e-4` s, `1e-3` ms, σ `> 0.05` bins.
- MT7 `evoked.rs::quantify_mep`: baseline floor `1e-4`, threshold `threshold_sigma.max(1.0)`, empty
  baseline → `(0, 1)`, no response → `ptp = rms = auc = 0` (fabricated, vs NaN onset).
- MT8 Host-only throughout (fine for spike-train metrics; template / STA averaging could share the
  device reducer).

### probe/
- P1 Probe presets are split across crates: NP 1.0 in `dsp_io::neuro::probe` (moved from core),
  NP 2.0 / HD-EMG grids / tetrode / Utah here. All presets and geometry helpers belong next to
  `SensorLayout` in `dsp-io/neuro/probe` (decision taken for dsp-io: 1a).
- P2 Neighbour search re-implemented: `probe::neighbors` (KNN, skips disabled sites),
  `detection::dedup::SitePositions` (radius), dsp-base `SpatialWhitening::fit_local_knn` and
  `SurfaceLaplacian::from_coordinates_knn` (their own KNN), extraction KNN tables per call. One
  geometry-query helper (KNN + radius, cached table) would serve all.
- P3 Presets are simplifications (NP 2.0 channel map ignores the IMRO table; Utah numbering is
  row-major, real maps are vendor-specific) — document as nominal geometry.
- P4 All files import `dsp_core::layout` (moved).

### sorting/omp (template matching)
- O1 **Name**: called Orthogonal Matching Pursuit but there is no orthogonal re-fit of earlier atoms;
  it is greedy matching pursuit with bounded amplitudes (`a ∈ [min, max]`) — rename or document.
- O2 **Rule break**: `match_spikes_omp` (and the `SpikeMatcher` impl) picks the device itself via
  `ComputeTarget::from_env()`; algorithms must take a `ComputeClient<R>` (only entry points choose).
- O3 Cost: score kernel = one unit per start, looping all units × rows × `t_len`
  (`O(S · U · R · T)`) every pass, re-scoring every start even where nothing was subtracted.
  Kilosort-style low-rank (SVD temporal × spatial) templates + per-pass local re-scoring would cut it
  by orders of magnitude.
- O4 Transfers: each pass downloads three full `valid_starts` arrays (12 B per sample) to pick local
  maxima on the host; picking / compaction could run on the device (as detection does).
- O5 Hidden constants: defaults `0.65` / `1.45` / **`min_explained_energy = 500.0` (µV², depends on
  the signal's scale)** / `4` passes; energy floor `1e-8`. f32 only; raw `client.empty(n * 4)`.
- O6 Good: picks in a pass are ≥ `t_len` apart so the subtract kernel never races; matches reported
  at the template trough; tested on collisions and on channel subsets.

### sorting/gmm
- G1 **BUG (precision)** `invert_spd_and_logdet` inverts each covariance with the old host
  `SymmetricEig::decompose(cov, d, 80)` — 80 single rotations do not converge past ~12 dimensions
  (same defect as the parked dsp-base solver), so precisions, log-determinants, likelihoods and
  **BIC model selection are wrong for `Full` / `Masked` covariance on typical feature sizes**. A
  host Cholesky (`O(d³)`, exact inverse and `log det = 2 Σ log Lᵢᵢ`) is the right tool for `d ≲ 64`.
- G2 Doc says "k-means++"; code is deterministic farthest-first, and the comment "first centroid
  closest to the overall mean" contradicts the code (it takes the **farthest** point → an outlier).
  Then 8 Lloyd iterations (literal).
- G3 Masked EM: the mean update divides the mask-weighted sum by `nk` (not by the mask-weighted
  count `Σ r·m`), and the covariance ignores the mask except for a final shrink toward 1 — an
  approximation of KlustaKwik masked EM; verify or document.
- G4 Hidden constants: weight floors `1e-4` / `1e-5`, responsibility skip `1e-9`, `1e-12`, `1e-30`,
  `reg.max(1e-6)`, init diagonal regularized twice (`max(reg) + reg`), `80` rotations, `8` Lloyd steps.
- G5 Host `O(n · k · d²)` per EM iteration; BIC sweep refits every `k` from scratch. Device candidate
  (E-step and M-step are reductions).

### sorting/isosplit
- I1 **Not IsoSplit**: MountainSort's IsoSplit tests unimodality with isotonic (up-down) regression
  (Magland & Barnett). This is a heuristic: a Gaussian KDE along the centroid axis, valley/peak
  ratio, `dip_score = 3 · (1 − ratio)`. Rename (e.g. "KDE valley merge") or implement isocut.
- I2 **Dead code** `isosplit.rs` ~L85–95: a first projection using `ca[0.min(0)]` (nonsense
  indexing) is computed and then immediately overwritten by "Simpler exact projection".
- I3 Logic: in the bimodal branch labels are re-cut but the loop goes on over the remaining pairs with
  centroids from the start of the pass (stale); passes stop when no merge happened even if cuts
  changed labels (no convergence check on splits).
- I4 Order-dependent seeding (centroids at evenly spaced input indices). Hidden constants: 30
  passes, 10 k-means iterations, KDE bandwidth floor `dist · 0.25`, valley search at `0.2…0.8 · dist`,
  `n < 6`, `1e-10`, `1e-6`, `1e-8`. Host only.

### sorting/density_peaks, similarity, cbss
- DP1 `density_peaks.rs`: correct Rodriguez–Laio (Gaussian ρ, δ, γ = ρ·δ, label propagation);
  exact `O(N²·D)` host; capped variant subsamples by stride. Device candidate (pairwise work).
  Constants `1e-4`, `1.1`.
- S1 `similarity.rs`: another host lag cross-correlation (see also matched filter, conduction
  velocity, dsp-base template alignment) — one shared "max normalized cross-correlation over lags"
  primitive would serve all. Norms use the whole template while the dot product uses only shared
  channels (documented, intentional).
- B1 `cbss.rs` is a simplification of Negro et al. 2016: no iterative source refinement (CoV-ISI
  minimization), threshold-on-RMS peak picking instead of k-means on `s²`; docs say "squared IPTs",
  code thresholds the sign-oriented IPT. Builds the full dense `(M·L) × S` extended matrix (memory
  grows with `L` and recording length). Uses host FastICA iterations.
- B2 `spike_train_coincidence` is another spike-train matcher (vs `metrics::compare_spike_trains`)
  with a hidden `±2` sample tolerance on top of the shift search.
- B3 Hidden constants: ICA tol `1e-4`, dedup `> 0.5` coincidence, `tol = L + 2`, PNR window `±2`,
  `1e-12`, `1e-6`, `1e-8`; `compute_pnr_db` returns `0.0` (fabricated) for empty input;
  `refractory_ms.max(2.0)`, `samples < 16`, `samples / 4`.

### spatial/ (localization)
- SP1 **BUG (dipole sample)** `dipole.rs` `DipoleLocalizer::localize` reads the signed potential at
  `batch.num_samples / 2`, assuming the trough is centred. With the default `StreamingSortConfig`
  (`pre_ms = 1.0`, `post_ms = 2.0`) the trough is at sample `pre` (30 at 30 kHz) but the code reads
  sample 45 — 0.5 ms after the trough. Root cause: `SnippetBatch` does not record its trough /
  `pre_samples`; it should.
- SP2 Doc mismatches: dipole says "damped coordinate-wise Gauss-Newton" — code is a compass /
  pattern search (± step per axis, halving). Grid convolution says it matches SpikeInterface
  `GridConvolution` — SI convolves prototype waveforms; this scores PTP against monopole footprints.
- SP3 Monopolar: LM with variable projection (good design) but builds a 4-parameter step and then
  discards the α component in favour of the closed-form α (two methods mixed).
- SP4 Physical assumptions as literals (µm): softening `+1` (monopolar, grid) vs `+4` (dipole), start
  depth `15` / `20`, step clamps `25` / `20`, depth bounds `[1, 250]` / `[2, 250]`, grid default z
  values; plus `1e-3`, `1e-8`, `1e-12`, fallback `α = 100`.
- SP5 Fabricated fallbacks: CoM → `[0, 0, 0]`, grid → `[0, 0, 15]`, dipole → residual `0`.
- SP6 `waveform_peak_to_peak` = 3rd peak-to-peak copy (dsp-base parked `peak_to_peak`,
  `core::template`).
- SP7 All localizers are host loops, one spike at a time; per-spike LM / grid search is ideal batched
  device work (independent per spike).

### streaming/config
- SC1 Two constants for the same realignment margin: `SINC_RESAMPLE_MARGIN = 8` (halos; doc says
  "Lanczos", the kernel is Blackman-Harris windowed sinc) vs `extraction::SINC_KERNEL_RADIUS = 5`
  (`detection_range`, extraction). One source of truth.
- SC2 Defaults documented in `Default` (good); floors `max(0.1)` ms / `max(0.5)` s as literals.

### spatial/drift
- DR1 **Reference = time bin 0 only** (`ref_profile = smoothed[0..num_depth_bins]`); the comment
  promises "first active time bin or mean profile". If bin 0 has no spikes every correlation is 0 and
  the drift is 0 everywhere. Single-reference registration also accumulates error over long
  recordings (Kilosort / DREDge use iterative-template or decentralized pairwise registration).
- DR2 Correlation is an unnormalized dot over the truncated overlap → biased toward lag 0.
- DR3 Literals: 3-tap smoothing `[0.25, 0.5, 0.25]` (edge bins not smoothed), `dt.max(0.1)`,
  `dz.max(1.0)`, `≥ 4` depth bins; non-rigid overlap `0.65` while the comment says "25 % overlap each
  side" (0.65 = 30 %); each block registers against its own bin 0.
- DR4 Kriging only consumes the rigid `DriftEstimate`; the non-rigid estimate has no correction path.

### spatial/kriging
- K1 Host Gauss–Jordan solver `continue`s past a near-zero pivot (silently wrong weights on singular
  systems); also a 3rd / 4th hand-written dense solver (`solve_3x3`, `solve_4x4`,
  `solve_linear_system`, dsp-base's new `solve` in the IIR module).
- K2 Applying weights is a host sparse matrix–vector product per drift step — dsp-base now has
  `DeviceSpatialMatrix` / sparse rows on the device; `correct_snippet_batch_drift_kriging` is a host
  `[K,K]×[K,T]` per spike.
- K3 `correct_traces_drift_kriging` evaluates `interpolate_drift_at` (a binary search) for every
  sample to find runs of equal quantized drift.
- K4 Literals: regularization `1e-2` passed in both call sites, floor `1e-5`, `two_sigma_sq.max(1.0)`,
  pivot `1e-12`. `DRIFT_QUANTUM_UM = 0.1` is named (good).

### streaming/ (runner, accumulator, kernels)
- ST1 Architecture is sound: prefetching reader (dsp-stream) → persistent `PipelineWorkspace`
  (halo windows) → device detection → exact streaming dedup (host) → device sinc extraction →
  device per-channel Welford → host Chan merge. Integer recordings upload stored values.
- ST2 **Name overstates it**: no clustering — `to_sorting_output` turns each primary channel into a
  "unit" (quality metrics then describe channels, not neurons). It is threshold detection +
  per-channel templates.
- ST3 `run` / `run_with` pick the device (`ComputeTarget::from_env`) inside the library (same rule
  break as OMP); `run_on(client)` is the right API.
- ST4 `calibrate_noise`: downloads each filtered calibration chunk and spawns **one OS thread per
  channel per chunk** (`std::thread::scope`, 384 at a time for Neuropixels) for host MAD; median-based
  σ on the device (selection / histogram) or a bounded pool.
- ST5 Device ↔ host ping-pong per window: detection downloads candidates, dedup on host, extraction
  re-uploads spikes; template reduce downloads `channels·K·L` mean and M2 every window even for
  channels with no spikes (could accumulate on device).
- ST6 Uses old core API (`gain_uv`, `offset_uv`), `ProbeLayout` alias, fallback noise `10.0` µV;
  negative polarity only (device detection); forward filter stages now start at rest by default
  (dsp-base `FilterStart`) — halos absorb it, but DC-heavy data would prefer `SteadyState`.
- ST7 Mean/SD implementations count: `compute_mean_template`, STA Welford, `TemplateAccumulator`
  (host Welford + Chan), device `reduce_channel_templates_kernel` — four.
- ST8 Tests hard-code `WgpuRuntime`.

### storage/
- SG1 Imports the moved `storage::npy` / `zarr_store` (now `dsp_io::container::{npy, zarr}`) → broken
  until phase 3 / fix pass.
- SG2 **Two parallel sorting models**: `SortingOutput` (unit-level, with metrics) and `PhySorting`
  (spike-level, for curation), with conversions both ways. `PhySorting::to_sorting_output` sets
  `total_samples = max spike time` (fabricated duration → firing rate / presence ratio biased high).
- SG3 Layering: `storage::phy` (a file writer) calls `sorting::similarity` to compute template
  similarity at export time; a schema writer should not depend on sorting algorithms.
- SG4 Hidden defaults: Phy `dtype` `"int16"`, `n_channels_dat` fallback; `params.py` parse failures
  become `sample_rate = 0.0` / `0` silently; NWB missing std → `1.0`, primary channel → `0`.
- SG5 Schema vs domain mixed in every file (dsp-io phase 3: schema readers / writers →
  `dsp_io::neuro::{phy, nwb::units, sorting_zarr}`; conversion to domain types stays here).

### Cross-crate
- X1 dsp-app re-detects sorting formats with `storage::zarr_store::has_array` (3rd format-detection
  site after `dsp_io::open` and `storage::load_sorting`). Otherwise the app reuses synapse metrics.
- X2 Alias paths (`clustering`, `localization`, …) are used by one synapse test and one Python
  binding file only.
