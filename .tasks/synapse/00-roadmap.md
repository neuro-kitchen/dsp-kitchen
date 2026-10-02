# `dsp-base` & `dsp-synapse` Architecture Refactor & Feature Roadmap

This roadmap tracks the structural reorganization of `crates/dsp-synapse` (aligning its module and `kernels/` layout with `crates/dsp-base`) and the implementation of the missing mathematical, spatial, Gaussian/clustering, HD-EMG, and evoked-potential primitives across `dsp-base`, `dsp-synapse`, and `dsp_kitchen_py`.

---

## Core Architectural Rules

1. **Co-located CubeCL `kernels/` Subfolders**:
   - Every domain module that launches GPU work (`detection/`, `extraction/`, `sorting/`, `streaming/` in `dsp-synapse`, and `filter/`, `math/`, `spatial/`, `linalg/` in `dsp-base`) owns its own local `kernels/` subdirectory.
   - Remove the monolithic top-level `crates/dsp-synapse/src/kernels/` directory.
2. **Strict Separation of Domain-Agnostic Math (`dsp-base`) vs. Neuroscience/HD-EMG (`dsp-synapse`)**:
   - `dsp-base` owns general signal statistics (`MAD`, `RMS`, trimmed noise, `SE`), window functions (`Sinc`, `Blackman-Harris`, `Gaussian`, `Hann`), 1D Gaussian smoothing filters, Spatial Reference (`CAR`, `Surface Laplacian`, `Global ZCA & Local K-NN Spatial Whitening`), and Linear Algebra (`PCA`, `PPCA`, `FastICA`).
   - `dsp-synapse` delegates general math/PCA/windows/noise to `dsp-base` and focuses strictly on electrophysiology & HD-EMG domain algorithms.
3. **Central `core/` Module for Domain Types & Traits**:
   - All data containers (`SpikeEvent`, `DeduplicatedSpike`, `MatchedSpike`, `WaveformSnippet`, `SnippetBatch`, `WaveformTemplate`, `UnitQualityLabel`, `NeuralBand`) and polymorphic traits (`SpikeDetector`, `WaveformDenoiser`, `FeatureEmbedder`, `PeakLocalizer`, `SpikeMatcher`) live in `crates/dsp-synapse/src/core/`.
4. **Zero Hardcoded `const` Geometry**:
   - All CubeCL kernels must dispatch through `dsp_core::compute::LaunchGeometry` (`channels_samples`, `per_channel`, `per_sample`, `elementwise`) without branching on `is_cpu` or hardcoding workgroup sizes.

---

## Target Tree Structure

### 1. `crates/dsp-base/src/`
```diff
crates/dsp-base/src/
├── lib.rs
├── filter/
│   ├── design/                      # Butterworth, Notch, SOS coefficient design
│   ├── fir/
│   │   ├── mod.rs
│   │   ├── conv.rs                  # 1D FIR convolution
+   │   ├── gaussian.rs              # [Task 01] 1D Gaussian smoothing filter & causal exponential/alpha kernel
│   │   └── kernels/
│   ├── iir/                         # Forward & zero-phase forward-backward (filtfilt) SOS + kernels/
│   ├── non_linear/                  # Median & Teager-Kaiser Energy Operator (TKEO) + kernels/
│   └── template/                    # Template subtraction
├── math/
│   ├── mod.rs
│   ├── baseline.rs
│   ├── clamp.rs
│   ├── scaling.rs
│   ├── unpack.rs
+   ├── stats.rs                     # [Task 01] MAD (from dsp-synapse), RMS, Trimmed/Spike-Masked σ, IQR, SE
+   ├── windows.rs                   # [Task 01] Sinc, Blackman-Harris (from dsp-synapse), Gaussian, Hann, Hamming
│   └── kernels/
│       ├── mod.rs
│       ├── baseline.rs
│       ├── clamp.rs
│       ├── scale.rs
+       └── stats.rs                 # [Task 01] CubeCL parallel reduction for channel RMS / mean / variance
├── spatial/
│   ├── mod.rs
│   ├── car.rs                       # Common Average Reference (CAR)
│   ├── reference.rs                 # SpatialReferenceConfig
+   ├── whitening.rs                 # [Task 01] Global ZCA & Local K-NN Spatial Whitening (W = Σ^-1/2)
+   ├── laplacian.rs                 # [Task 01] 2D Surface Laplacian / Double-Differential filter for HD-EMG grids
+   └── kernels/
+       ├── mod.rs
+       ├── car.rs
+       └── whitening.rs             # [Task 01] CubeCL [C, C] spatial whitening matrix projection kernel
├── linalg/
│   ├── mod.rs
│   ├── pca.rs                       # Classical SVD / Covariance PCA
│   ├── svd.rs                       # Covariance + Jacobi SVD
+   ├── ppca.rs                      # [Task 01] Probabilistic PCA (Gaussian latent subspace with isotropic noise σ²)
+   ├── ica.rs                       # [Task 01] FastICA / Convolutive Whitening solver
│   └── kernels/
├── pipeline/                        # Composable in-VRAM Pipeline & PipelineStage (+ SpatialWhitening & Gaussian)
└── resampler/                       # Min-max decimation & multi-resolution pyramid cache
```

### 2. `crates/dsp-synapse/src/`
```diff
crates/dsp-synapse/src/
├── lib.rs
+├── core/                           # [Task 02] Central domain data types & polymorphic traits
+│   ├── mod.rs
+│   ├── events.rs                   # SpikeEvent, DeduplicatedSpike, MatchedSpike
+│   ├── snippets.rs                 # WaveformSnippet [K, T] & SnippetBatch [N, K, T]
+│   ├── template.rs                 # WaveformTemplate (mean, std, se, channel_ids) & UnitQualityLabel
+│   ├── bands.rs                    # NeuralBand definitions
+│   └── traits.rs                   # SpikeDetector, WaveformDenoiser, FeatureEmbedder, PeakLocalizer, SpikeMatcher
├── probe/
│   ├── mod.rs
│   ├── neighbors.rs
│   ├── neuropixels.rs
│   ├── tetrode.rs
│   ├── utah.rs
+   └── hdemg_grid.rs                # [Task 02] Canonical 2D planar HD-EMG grids (4x8, 8x8, 13x5)
├── detection/
│   ├── mod.rs
│   ├── threshold.rs                 # Negative / Positive / Both-polarity threshold detector
│   ├── neo.rs
│   ├── matched_filter.rs
│   ├── dedup.rs
+   ├── adaptive.rs                  # [Task 03] Sliding-window adaptive noise & threshold tracker
+   └── kernels/                     # [Task 02/03] Co-located CubeCL detection & deduplication kernels
+       ├── mod.rs
+       ├── threshold.rs
+       └── dedup.rs
├── extraction/
│   ├── mod.rs
+   ├── alignment.rs                 # [Task 02] Parabolic trough fit + windowed-sinc fractional shift
+   ├── extractor.rs                 # [Task 02] Unified multichannel snippet & SnippetBatch extractor
+   └── kernels/                     # [Task 02] Co-located CubeCL sinc extraction kernels
+       ├── mod.rs
+       └── sinc.rs
+├── spatial/                        # [Task 02/03] Merged localization/ + motion/
+│   ├── mod.rs
+│   ├── center_of_mass.rs
+│   ├── monopolar.rs
+│   ├── grid_convolution.rs
+│   ├── dipole.rs                   # [Task 03] Dipole source localization
+│   ├── drift.rs                    # [Task 03] Rigid + Non-Rigid multi-depth block drift registration
+│   └── kriging.rs
├── features/
│   ├── mod.rs                       # [Task 02/04] PCA & PPCA FeatureEmbedder adapters delegating to dsp-base
│   ├── morphology.rs
+   ├── wavelet.rs                   # [Task 04] Multi-scale Haar / Daubechies wavelet features
+   └── conduction.rs                # [Task 04] 2D HD-EMG propagation delay & muscle fiber conduction velocity
+├── sorting/                        # [Task 02/04] Merged clustering/ + matching/ + decomposition
+│   ├── mod.rs
+│   ├── density_peaks.rs
+│   ├── gmm.rs                      # [Task 04] Gaussian Mixture Model (EM + Masked EM + BIC auto-K selection)
+│   ├── isosplit.rs                 # [Task 04] MountainSort IsoSplit 1D dip-test clustering
+│   ├── omp.rs
+│   ├── similarity.rs
+│   ├── cbss.rs                     # [Task 04] Convolutive Blind Source Separation (CKC / FastICA) for HD-EMG
+│   └── kernels/                    # [Task 02] Co-located CubeCL OMP deconvolution kernels
+       ├── mod.rs
+       └── omp.rs
├── metrics/
│   ├── mod.rs
+   ├── firing.rs                    # [Task 02] ISI violations, presence ratio, amplitude cutoff, contamination
│   ├── isolation.rs                 # SNR, SE, d-prime, Mahalanobis isolation distance, silhouette
+   ├── correlogram.rs               # [Task 02] Auto- & cross-correlograms (ACG / CCG)
+   ├── rate.rs                      # [Task 05] Gaussian-smoothed firing rate r(t) & respiratory burst envelope
+   └── evoked.rs                    # [Task 05] Stimulus-Triggered Averaging (STA), PSTH, and MEP metrics
└── streaming/
    ├── mod.rs
    ├── config.rs
    ├── accumulator.rs
    ├── runner.rs
+   └── kernels/                     # [Task 02] Co-located CubeCL template moment reduction kernels
+       ├── mod.rs
+       └── template_reduce.rs
```

---

## Execution Tasks

- [x] **[Task 01: `dsp-base` Foundations — Robust Stats, Windows, Gaussian Filter, Spatial Whitening & PPCA/ICA](01-dsp-base-stats-windows-whitening-and-ppca.md)**
- [x] **[Task 02: `dsp-synapse` Structural Refactor — `core/`, Co-located `kernels/`, and Module Consolidation](02-dsp-synapse-structural-refactor.md)**
- [x] **[Task 03: Detection & Spatial Enhancements — Polarity/Adaptive Detection, Dipole Localization & Non-Rigid Drift](03-detection-and-spatial-drift.md)**
- [x] **[Task 04: Features & Sorting — Wavelets, Conduction Velocity, GMM (EM/BIC), IsoSplit & HD-EMG cBSS](04-features-gmm-isosplit-and-cbss.md)**
- [x] **[Task 05: Continuous Firing Rates, Correlograms & Stimulus-Aligned Evoked Potentials (PSTH / STA / MEP)](05-metrics-firing-rates-and-evoked-potentials.md)**
- [x] **[Task 06: `dsp_kitchen_py` Bindings Expansion & End-to-End Benchmarks](06-python-bindings-and-validation.md)**
- [x] **[Task 07: `SortingOutput` Container & Sorter-to-Sorter Comparison Metrics](07-sorting-output-and-comparison.md)**
- [x] **[Task 08: Sorter Output Persistence — Phy/Kilosort (`.npy`/`.tsv`), Zarr Analyzer (`.sorting.zarr`), and NWB `/units`](08-sorting-storage-phy-zarr-nwb.md)**
- [ ] **[Task 09: `dsp_kitchen_py` Storage & Sorter Comparison Bindings](09-python-storage-and-comparison-bindings.md)**

Order: 02 → 01 → 03 → 04 → 05 → 06 → 07 → 08 → 09.

---

## Git Commit Messages (Executed in Order 02 → 01 → 03 → 04 → 05 → 06 → 07 → 08 → 09)

```bash
# Task 02 (a875235)
git add crates/dsp-synapse/
git commit -m "refactor(dsp-synapse): reorganize into core, spatial, sorting, and co-located kernels"

# Task 01 (dd8e49d)
git add crates/dsp-base/
git commit -m "feat(dsp-base): add robust stats, windows, Gaussian FIR, spatial whitening, Laplacian, PPCA, and FastICA"

# Task 03 (43865f3)
git add crates/dsp-synapse/src/detection/ crates/dsp-synapse/src/spatial/ crates/dsp-synapse/src/lib.rs
git commit -m "feat(dsp-synapse): add polarity/adaptive detection, GPU spatial dedup, dipole localizer, and non-rigid drift"

# Task 04 (2bd7e03)
git add crates/dsp-synapse/src/features/ crates/dsp-synapse/src/sorting/ crates/dsp-synapse/src/lib.rs
git commit -m "feat(dsp-synapse): add PPCA/wavelet features, MFCV, GMM with BIC/Masked EM, IsoSplit, and HD-EMG cBSS"

# Task 05 (2d54c4f)
git add crates/dsp-synapse/src/metrics/ crates/dsp-synapse/src/lib.rs
git commit -m "feat(dsp-synapse): add continuous Gaussian firing rates, PSTH, STA, and MEP quantification"

# Task 06 (1e72bbb)
git add dsp_kitchen_py/
git commit -m "feat(dsp_kitchen_py): expose whitening, PPCA, FastICA, GMM, IsoSplit, cBSS, and evoked potential bindings"

# Task 07 (b4b9c1e)
git add crates/dsp-synapse/src/core/ crates/dsp-synapse/src/metrics/ crates/dsp-synapse/src/streaming/ crates/dsp-synapse/src/lib.rs
git commit -m "feat(dsp-synapse): add SortingOutput container and sorter-to-sorter comparison metrics"

# Task 08
git add crates/dsp-synapse/ .tasks/synapse/00-roadmap.md
git commit -m "feat(dsp-synapse): add Phy/Kilosort, Zarr SortingAnalyzer, and NWB /units storage"

# Task 09
git add dsp_kitchen_py/ .tasks/synapse/00-roadmap.md
git commit -m "feat(dsp_kitchen_py): expose SortingOutput storage (Phy, Zarr, NWB /units) and compare_sortings"
```


