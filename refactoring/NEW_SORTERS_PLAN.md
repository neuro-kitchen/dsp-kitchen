# New sorters: MountainSort 5, SpyKING CIRCUS 2, Tridesclous 2 (plan, 2026-10-08)

Status: **plan only**, no code. Wait for the user's go-ahead per task. Decided with the user: add these
three; Herding Spikes 2 and Kilosort 2.5 / 3 are not planned.

Same rules as `SORTER_TASKS.md` and `GPU_TASKS.md`:
- GPU first, with every host↔device crossing counted.
- Reproducible by default (named seeds).
- Tests compare results against a reference.
- Timeouts ≤ 5 min; each step written up in the mdBook.
- Every tunable value is a setting whose default is visible: a Rust `Default` and a Python keyword
  argument with its default in the signature.

## Why these three, and what licence allows

| Sorter | Code | Licence | What we may use |
|---|---|---|---|
| MountainSort 5 | `mountainsort5` (Flatiron), with `isosplit6` | Apache-2.0 (confirm) | Paper **and code**: validate stage by stage against it |
| SpyKING CIRCUS 2 | in `spikeinterface.sortingcomponents` | MIT | Code; components shared with Tridesclous 2 |
| Tridesclous 2 | in `spikeinterface.sortingcomponents` | MIT | Code |

Unlike Kilosort4 (GPL, paper only), these permit reading the source. We reimplement in Rust on the
device; we do not copy Python. Each sorter gets a `Provenance` (paper, code, licence) like
Kilosort4 / EMUsort.

**First step for each sorter:** read the current source and write its stage list with defaults into
its book page *before* coding. The algorithm notes below are from memory and must be confirmed there.

## Tasks

| # | Task | Depends on |
|---|---|---|
| E1 | **First, before N0:** EMUsort speed after phase 2. **Found (2026-10-09):** the 60 Hz notch (Q 30) settles over 24 600 samples (1 s) per side at 24.4 kHz, so every 60 000-sample batch reads 1.83× its length in each of the four passes: 23.1 s warm with the notch, 15.9 s without (benchmark 15.3 s); band-pass and merges cost ~nothing. Quality is not a regression. **Done:** filter settings are three switches (`do_car`, `do_bandpass` + `bandpass_low_hz` / `bandpass_high_hz`, `do_notch` + `notch_hz` / `notch_q`), defaults from `NeuralBand::Ap` (Kilosort4, 300–6000 Hz; upstream has no upper edge) and the new `NeuralBand::Emg` (EMUsort, 300–5000 Hz); built by `runner::filter_stages`. **Decided (user, 2026-10-09):** notch off by default (the 300 Hz edge attenuates 60 Hz by ~84 dB; checked); EMUsort 16.4 s warm without it (benchmark 15.3 s), same quality; documented in the EMUsort tuning page. Upper band edge optional (`None`: a high-pass). Kilosort4's default band 300–6000 Hz with thresholds raised to 10 / 9 (same agreement as the high-pass at 9 / 8; 11 / 10 misses spikes); EMUsort keeps 9 / 8. Fixed 2026-10-09: the exit hang/crash (devices shut down at exit, dsp-core `compute::target`), and EMUsort's nested Kilosort4 config (option A: `EmusortConfig` is flat, every setting with EMUsort's default; Python `kilosort4.Config` and `emusort.Config` are generated from one settings list) | — |
| N0 | Test data and harness (below): ground-truth sets, reference outputs, one benchmark script | — |
| N1 | Shared components: peak selection (subsampling), local SVD features, template estimation, a common `SorterResult` → `SortingOutput` | N0 |
| N2 | **MountainSort 5** | N1 |
| N3 | **SpyKING CIRCUS 2** | N1 (+ S1 correlograms for its merges) |
| N4 | **Tridesclous 2** | N1, N3 (shares its components) |
| N5 | Book: sorter pages (intro, pipeline, parameters, tuning) per sorter; Benchmarks page with all five sorters | each |
| N6 | Python: `dsp_kitchen.synapse.ml.<sorter>.run` with typed configs, numpydoc, Phy / Zarr export | each |

Order: E1 → N0 → N1 → N2 (simplest, fewest settings) → N3 → N4.

### N2. MountainSort 5 (to confirm from source)

- Input is filtered and whitened data; in SpikeInterface the user does this beforehand. We provide
  it as a fitted `Pipeline`: band-pass + whitening, with the settings visible.
- **Scheme 1:**
  1. detection (threshold in σ, sign, time and channel radii, locally exclusive);
  2. snippets;
  3. PCA per channel, then per subdivision;
  4. **isosplit6** clustering: unimodality tests on 1-D projections between cluster pairs, with no
     cluster-count or tuning parameter;
  5. templates.
- **Scheme 2** (long recordings): scheme 1 on a training subset, then every spike classified against
  the learned clusters.
- **Scheme 3:** chunked scheme 2 for drift.
- **GPU:** detection, snippets, PCA and classification on the device. isosplit's pair tests are many
  small 1-D problems: batch the projections on the device, and keep the dip tests on the host unless
  measurement says otherwise.

### N3. SpyKING CIRCUS 2 (to confirm from source)

Stages:
1. preprocessing: filter, common median reference, whitening;
2. locally exclusive detection;
3. smart peak subsampling;
4. SVD features;
5. "circus" clustering: local clusterings, iteratively split, then merged;
6. template estimation;
7. template merges (correlogram-based, which reuses S1);
8. template matching ("circus-omp" family: orthogonal matching pursuit).

Reuse: detection, whitening, our matching pursuit kernels (`kilosort4/matching.rs`) as a base for
OMP, HDBSCAN / k-means.

### N4. Tridesclous 2 (to confirm from source)

Shares SpyKING CIRCUS 2's preprocessing, detection and peak selection. Its own pieces:
- clustering per channel neighbourhood (local SVD + HDBSCAN, iterative splits);
- merges;
- its "peeler" template matching.

Mostly a new composition of N1 + N3 pieces.

## Test data (N0)

**Can we use the Kilosort4 data?** Yes, for real-data runs, speed, and agreement with Kilosort4 /
between sorters. But it has **no ground truth**: Kilosort4's sorting is a reference, not the truth.
Validation needs two more things: ground truth, and the original sorters' own outputs.

| # | Data | Have it? | Used for |
|---|---|---|---|
| D1 | **Synthetic** (`SyntheticRecording`: 32 ch, known spike times per unit, drift option) | yes | Unit tests and correctness: accuracy / precision / recall per unit against the truth. Probe layouts include `tetrode` and linear |
| D2 | **Kilosort4 Neuropixels file** (`data/kilosort4/ZFM-02370_mini…bin`) + Kilosort4's saved results | yes | Real noise and density; speed; agreement with Kilosort4 (same `ks4_reference.py` / `compare` machinery) |
| D3 | **Hybrid ground truth on D2**: Kilosort4's *good* units' templates injected at known times into the same file (the SpikeInterface hybrid method; known units, real background) | to build (generator script) | The main correctness benchmark on realistic data, for all five sorters including our Kilosort4 |
| D4 | **Original sorters' outputs** on D1, D2, D3: `mountainsort5`, SpyKING CIRCUS 2, Tridesclous 2 run once through SpikeInterface (CPU is enough), saved to `data/sorters/<name>_reference/` | to produce (**needs your OK**) | Stage-by-stage validation, the way `saved_results` was used for Kilosort4 |
| D5 | **IBL Neuropixels** (`data/ibl/imec_385_100s`, 100 s, 385 ch) | yes | A second real recording: robustness and speed. No ground truth |
| D6 | **Paired ground truth** (juxtacellular / patch + extracellular, e.g. SpikeForest's paired sets) | no (download) | Optional: one real neuron with known spikes. Most useful for MountainSort 5 on tetrode-like data |

The HD-EMG data is not for these sorters: they are built for extracellular neural recordings.

**Harness:** `playground/benchmarks/sorter_benchmark.py`. It runs any of the five sorters on D1–D5
and reports:
- per-unit accuracy against the truth (D1, D3);
- agreement with the reference outputs (D4) and with Kilosort4 (D2);
- units, spikes, *good* units, and run time.

It reuses `dsp_kitchen.synapse` comparison and the `SortingOutput` exports.

## Decisions (user, 2026-10-09)

- **D4: no** reference runs of the original sorters: the new sorters are validated against
  Kilosort4's output (`data/kilosort4/saved_results`).
- **D6: no** paired download: the Kilosort4 dataset is the real-data set.
- **D3: no** hybrid ground truth (2026-10-09): validation is against Kilosort4's output only.

## Decisions needed before starting (answered above)

1. **D4:** may we install and run the original sorters once (Python, through SpikeInterface) to get
   reference outputs? Without them we validate against ground truth only, not stage by stage.
2. **D6:** download a paired ground-truth set, or skip it?
3. **D3 hybrid size:** how many injected units and at which amplitudes. Proposed: Kilosort4's good
   units at their own amplitudes, plus a set at reduced amplitude for detection limits.

## N1 progress (log, updated after each step)

Layout decided 2026-10-09: subsampling `dsp-synapse/src/sorting/subsample.rs`, local features
`dsp-synapse/src/features/local_svd.rs`, templates from labels `dsp-synapse/src/core/template.rs`,
common result `dsp-synapse-ml/src/sorters/result.rs` (Kilosort4Result built on it). New sorters in
`dsp-synapse-ml/src/sorters/{mountainsort5,spykingcircus2,tridesclous2}/`; isosplit in
`dsp-synapse/src/sorting/isosplit.rs` (N2). Builds: no CLI; Python via `--profile validate`;
each run ≤ 5 min.

- [x] Step 1: peak subsampling (2026-10-09). `dsp_synapse::sorting::subsample_peaks(samples,
  channels, SubsampleOptions { n_peaks, per_channel, seed })`: uniform without replacement (partial
  Fisher–Yates), result sorted by sample; host (metadata only). 4 tests (size, no repeats, sorted,
  reproducible, per-channel cap, equal probability over 4000 seeds). Only upstream's `"uniform"`:
  SpyKING CIRCUS 2 (`n_peaks = max(100 000, 5000 · channels)`, seed 42) and Tridesclous 2
  (`max(20 000, 5000 · channels)`) use nothing else; the `smart_sampling_*` methods were tried and
  dropped (after their quantile transform, acceptance `1 − s` favours low values, it does not
  flatten). Found on the way (upstream source, for N3/N4): SpyKING CIRCUS 2 = Bessel 150–7000 Hz +
  CMR (≥ 32 ch) + whitening, matched-filter detection (−, 5σ), uniform selection, **iterative
  HDBSCAN**, templates (SVD/median), `circus-omp` matching, auto-merge; Tridesclous 2 = Bessel
  150–6000 Hz + CMR + whitening, locally exclusive detection (−, 5σ), uniform selection, SVD
  features (120 µm radius, 5 per channel), **iterative isosplit** (shared with MountainSort 5),
  accumulated templates, `tdc-peeler` matching, auto-merge.
- [x] Step 2: local SVD features (2026-10-09). `dsp_synapse::features::local_svd`: `LocalSvd::fit`
  / `transform`, `LocalSvdOptions` (upstream `extract_peaks_svd` defaults: 5 components, 0.5 / 1.5 ms,
  120 µm, ≤ 5000 fit peaks), `ChannelNeighbourhoods::within_radius`. Algorithm from upstream source:
  single-channel fit rows on each peak's own channel, kept when argmax |x| is at the peak, signed
  positive there; TruncatedSVD (no centring) = top eigenvectors of XᵀX, sklearn's svd_flip sign
  (largest entry positive). All on the device (gather kernel → `matmul` XᵀX → Jacobi `symmetric_eigen`;
  only the `[width, width]` result comes back); transform = one kernel, features stay on the device,
  layout `[peaks, components, max_neighbours]` (channels ascending, 0-padded). 3 tests: radius
  neighbourhoods, rank-2 data spanned (residual < 1e-4) and device features = host dot products,
  misaligned waveforms excluded and sign convention.
- [x] Step 3: templates from cluster labels (2026-10-09). `dsp_synapse::core::UnitTemplateAccumulator`
  (`new(client, n_units, channels, n_before, width)`, `add(window, samples, peak_samples, labels)`,
  `templates() -> Vec<Option<WaveformTemplate>>`, `device_mean()`, `counts()`). Kernel
  `core::kernels::accumulate_unit_templates_kernel`: one unit per (unit, channel, sample) continues
  the Welford mean / M₂ over the unit's spikes, reading the window directly (no snippets: a full
  Neuropixels snippet set would be billions of values); state on the device between windows, only
  per-unit counts on the host, one download at the end; negative labels skipped. Same Welford as the
  existing streaming `reduce_channel_templates_kernel` (which groups pre-cut snippets by channel).
  2 tests: shapes recovered (max error < 0.05, std = noise), unassigned spikes ignored; two windows =
  one window.
- [x] Step 4: common sorter result (2026-10-09). `dsp_synapse_ml::sorters::{SorterResult, AmplitudeScale}`
  (`sorters/result.rs`): per-spike samples / units / amplitudes / locations, a template per unit, the
  amplitude scale (`Whitened`: SNR = median |amplitude|; `Recording { noise_levels }`: template peak
  on the primary channel / its noise), label and composite-score settings; `to_sorting_output` holds
  the logic moved from `Kilosort4Result` (primary channel = most template energy, good / mua from the
  ACG, composite score). `Kilosort4Result::sorter_result()` builds it; `to_sorting_output` delegates.
  2 tests (channel / label / score per unit, empty units left out; the scale sets the SNR). Note from
  the label test: the paper's statistic takes the minimum over k, so a train with an empty bin 1 but
  a full bin 2 reads as refractory (test data fixed; real trains do not do that). Kilosort4 output
  unchanged (validation: 130 225 spikes, 291 units, 154 good, same agreement); 34 sorter tests and
  the Python tests pass.

**N1 complete (2026-10-09).** Next: N2, MountainSort 5 (read its current source first and write its
stage list with defaults into its book page before coding; isosplit6 in
`dsp-synapse/src/sorting/isosplit.rs`, shared with Tridesclous 2).


## N2 progress (MountainSort 5; log, updated after each step)

Source read 2026-10-09 (`flatironinstitute/mountainsort5` v0.5.9 `3008b7a`, `magland/isosplit6`,
both Apache-2.0; SpikeInterface wrapper defaults for filtering / whitening / scheme 2 radii). 8 GB
VRAM: phase-1 snippets stored on their masked channels only (same numbers), PCA randomized above
8000 features as upstream (`pca_solver.py`). isosplit6 is deterministic (parcel seeds = first 3
points).

- [x] N2.1 book pages (2026-10-09): `docs/book/src/sorters/mountainsort5/{intro,pipeline,parameters,tuning}.md`
  (stages, isosplit6 / isocut6 algorithm, scheme 2, defaults with their source, device plan; all
  stages *planned*), SUMMARY and sorter index updated; book builds. Citations checked online
  (Chung et al. 2017 Neuron doi:10.1016/j.neuron.2017.08.030; Magland & Barnett arXiv:1508.04841).
- [x] N2.2 isosplit6 (2026-10-09): `dsp-synapse/src/sorting/isosplit.rs` (host, f64): `isosplit6`,
  `isocut6`, `parcelate`, `IsosplitOptions` (2.0 / 10 / 200 / 500); jisotonic5 up-down / down-up,
  ks4 / ks5, mutual-nearest pairs, inverse-average-covariance merge test. Ours: singular covariance →
  ridge retry (upstream aborts). 4 tests pass (isotonic, isocut uni/bimodal, 3 blobs + 1 blob, parcels).
- [x] N2.3 detection (2026-10-09): `dsp-synapse-ml/src/sorters/mountainsort5/{detect.rs,kernels/detect.rs}`:
  `Detector` (neighbourhood table once per probe), `DetectOptions` (5.5 / −1 / 0.5 ms / None);
  device candidates (`find_peak_candidates_on_device`) → `locally_exclusive_kernel` (window scan in
  the trace, margins honoured) → duplicate times removed. Test: equals an upstream port for signs
  −1/+1/0, radius None/30 µm, and split into two windows. Difference: a flat run reports its first sample.
- [x] N2.4 (2026-10-09): `dsp-base/src/linalg/subspace.rs` `TopComponents` (batched device PCA over a
  `RowSource`: covariance formed ≤ 8000 features → Jacobi or subspace iteration to 1e-6; above: implicit
  `Σ XᵀXQ`, scikit-learn's 7 / 4 iterations; Gram-eigen orthonormalisation; sklearn signs) + kernels
  `centre_columns`, `add_assign`; `mountainsort5/snippets.rs` `MaskedSnippets` (masked storage, device
  gather, dense rows rebuilt per batch, `roll`); `mountainsort5/subdivision.rs`
  `isosplit6_subdivision` (PCA 10 → isosplit6 → medians → MST longest-edge cut → recurse). Tests: PCA
  3 paths vs host covariance; snippets = upstream dense; 5 blobs in 30-D recovered.
- [x] N2.5 (2026-10-09): `mountainsort5/templates.rs` (medians with masked zeros, `align_templates` with the
  per-shift products on the device, `offsets_to_peak`, `peak_channels`), `scheme1.rs` `cluster_events`
  (PCA → subdivision → align/roll → recluster → to-peak → margins → units by peak channel; segments).
- [x] N2.6 (2026-10-09): `scheme2.rs` `Classifiers` (noise + per-unit training on each channel's mask,
  per-channel `TopComponents`, device projection + second-nearest kernels), `remove_duplicate_events`;
  `runner.rs` `run` (filters + SI-style global ZCA whitening on 20 evenly spaced 10 000-sample chunks,
  training segments as upstream, phase 1, training pass, chunked phase 2); `mod.rs`
  `Mountainsort5Config` (flat, visible defaults), provenance `PortedFromCode`. Synthetic 32 ch / 30 s / 6
  units (validate profile): scheme 2 all 6 units ≥ 0.99 accuracy (47 s total incl. scheme 1); scheme 1
  10 units, 4 at ≥ 0.97: it splits two broad units by detection jitter (whitened troughs flat over ±1
  sample; checked: alignment and times consistent). Ours: scheme-2 unit templates only within the mask
  radius of the peak channel (memory); segments not concatenated. Debug builds: the end-to-end test is
  ignored (minutes); run with `--profile validate`.
- [x] N2.7 (2026-10-09): config + `SorterResult`; Python `synapse/mountainsort5.rs` + `ml/mountainsort5.py`
  (defaults in the stub); book pages (status, preprocessing, device, choices table, parameters with our
  names) and Benchmarks. Speed work: snippets / features / classifier rows resident on the device; host
  `f64` eigensolver (`dsp-base/src/linalg/tridiagonal.rs`, tred2/tql2) for matrices ≤ 512 in
  `TopComponents`; isosplit pair tests + covariance sums in parallel (fixed order); medians by
  selection; classifier PCA randomized (choice recorded). KS4 recording: 404 s; 35 121 spikes / 165 units
  (KS4: 138 666 / 267); KS4 good units found 0.83, not split 0.95, precision 0.91, accuracy 0.59. Fewer
  spikes = 5.5 σ single-channel detection after global whitening (whitened σ checked = 1.00).
  **N2 complete.** Open speed-ups: phase-1 subdivision (297 s: lopsided tree), snippet PCA per channel group.

## N3 progress (SpyKING CIRCUS 2; log, updated after each step)

Source read 2026-10-09: SpikeInterface main `f08c987` (MIT), `sorters/internal/spyking_circus2.py`
(sorter version "2025.12") and `sortingcomponents/*`. Much larger than the 2026-10-08 plan assumed:
matched-filtering detection, motion correction on by default, iterative HDBSCAN, circus-omp,
`auto_merge_units` at the end. Motion correction (DREDge) is S10 (not done): SC2 runs without it
until S10 lands (recorded as a difference). Stages:

- [ ] N3.1 preprocessing: Bessel band-pass 150–7000 Hz order 2 (forward-backward), CMR (median)
  when ≥ 32 channels, **local** whitening (radius), noise levels (MAD of random chunks)
- [ ] N3.2 prototype: locally exclusive detection (threshold 5, radius 50 µm, sweep max(ms_before,
  ms_after)), 10 000 peaks, max-channel waveforms, `nanmedian(w / |w[nbefore]|)`
- [ ] N3.3 matched filtering: prototype convolution, spatial weights `exponential_3d` (5 depths 0–120
  µm, σ 2.5, sparsified at 0.5/√n, renormalised), thresholds 5 × MAD of the convolved random data,
  local maxima, locally exclusive over templates within 50 µm (normalised by threshold)
- [ ] N3.4 selection: uniform, `max(100 000, 5000 · channels)` peaks, seed 42 (N1 `subsample_peaks`);
  upstream stops detection after that many peaks over shuffled chunks (non-deterministic mode): here all
  peaks are detected then subsampled
- [ ] N3.5 features: `LocalSvd` (N1; 5 components, 0.5 / 1.5 ms, radius 100 µm)
- [ ] N3.6 iterative split (`split_clusters` + `LocalFeatureClustering`; shared with TDC2): per
  label, channels of the intersection of its peaks' neighbourhoods (75 µm; skip when intersection /
  union < 0.25), aggregated sparse features, TruncatedSVD to 3, HDBSCAN (min 20, single cluster
  allowed), recursive depth 3
- [ ] N3.7 templates from the SVD features (median), `clean_templates` (sparsify SNR 1, min SNR 5,
  max jitter 0.2 ms, mean SD ratio 3), merge by template similarity (l1 > 0.8, 3 shifts, lags), small
  clusters (0.1 Hz with the subsampling factor)
- [ ] N3.8 circus-omp matching
- [ ] N3.9 final merges (`auto_merge_units`, presets `x_contaminations` over template_diff_thresh
  0.05…0.45, 50 µm, censor 3 ms, sparsity overlap 0.5, recursive)
- [ ] N3.10 config (visible defaults), `SorterResult`, Python, book pages, validation vs Kilosort4
