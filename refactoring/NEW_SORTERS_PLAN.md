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

- [ ] Step 1: peak subsampling
- [ ] Step 2: local SVD features
- [ ] Step 3: templates from cluster labels
- [ ] Step 4: common sorter result

