# MountainSort 5: pipeline

Read from the source (`mountainsort5/schemes/sorting_scheme1.py`, `sorting_scheme2.py`,
`core/*.py`; `isosplit6/src/*.cpp`). Every stage is implemented (`dsp_synapse_ml::sorters::mountainsort5`).

## Scheme 1 (phase 1 of scheme 2)

1. **Detection** (`detect_spikes`). On the (filtered, whitened) traces, every sample at or below
   `−detect_threshold` (5.5) is a candidate (`detect_sign = −1`; +1 flips the traces, 0 uses
   `−|x|`). A candidate is kept when no candidate on a channel within `detect_channel_radius` of
   its own (all channels when unset) is **strictly** lower within `±detect_time_radius`
   (`ceil(0.5 ms · fs)` samples). Candidates within `snippet_T1` / `snippet_T2` of the ends are
   dropped. Then events at the same sample are reduced to one (`remove_duplicate_times`).
2. **Snippets** (`extract_snippets`): `snippet_T1 = 20` samples before to `snippet_T2 = 20` after,
   on every channel, zero outside `snippet_mask_radius` of the event's channel.
3. **PCA** (`compute_pca_features`): scikit-learn `PCA` (centred) to `npca_per_channel · channels`
   components (3 per channel) of the flattened snippets. Exact (`covariance_eigh`) up to 8000
   features, **randomized** above (a whole Neuropixels probe), seeded.
4. **isosplit6 subdivision** (`isosplit6_subdivision_method`), recursive: PCA of the subset to
   `npca_per_subdivision = 10` components, **isosplit6**; when it finds more than one cluster, the
   clusters' medians (in the step-3 space) are split in two groups by single-linkage, and each
   group is clustered again the same way.
5. **Templates** (`compute_templates`): the **median** snippet of each cluster.
6. **Alignment** (unless `skip_alignment`): each template pair's best circular shift (largest inner
   product); per-template offsets as the weighted average of the pairwise ones, iterated up to 20
   times; snippets shifted, times shifted, then steps 3–5 again on the aligned snippets; finally
   each spike moved to its template's peak (`detect_sign` peak on the peak channel).
7. **Units** sorted by peak channel; spikes within `snippet_T1` / `snippet_T2` of the ends dropped.

## isosplit6

Start from **parcels**: the data is split recursively, a parcel larger than `min_cluster_size`
(10) and wider than 95% of the widest such parcel is divided among its **first 3 points**
(nearest one wins), until `K_init` (200) parcels or nothing changes. Deterministic.

Then passes of iterations: pair each cluster with its **mutual nearest** centroid among the pairs
not compared yet; for each pair, if either has fewer than `min_cluster_size` points it merges;
otherwise project both on the direction `(Σ₁ + Σ₂)⁻¹ / 2 · (μ₂ − μ₁)` (normalised) and run
**isocut6**: dip score below `isocut_threshold` (2.0) merges, otherwise the points are
redistributed at the cut point. Centroids and covariances of changed clusters are recomputed. A
pass ends when no pair is left (at most 500 iterations); passes repeat until one merges nothing,
plus a final pass. Labels are renumbered `1..K`.

**isocut6** (1-D): sort; log-densities `log(1 / spacing)` between neighbours; the best **unimodal**
fit by up-down isotonic regression (`jisotonic5_updown`); the dip score is the largest
Kolmogorov–Smirnov distance (`√(n/2)`-scaled) between the data and the fit over ranges halving
from the peak on either side; the cut point is the minimum of the residual's down-up isotonic fit
on that critical range.

## Scheme 2 (the default)

1. **Training stretch**: `training_duration_sec` (300 s) of the recording, in 10 s chunks spread
   uniformly (`uniform`) or from the start (`initial`).
2. **Phase 1**: scheme 1 on it, with `phase1_detect_channel_radius` (200 µm),
   `phase1_detect_time_radius_msec` (1.5 ms), `phase1_detect_threshold` (5.5).
3. **Classifiers**, one per channel, on that channel's `snippet_mask_radius` neighbourhood: 200
   noise snippets at uniformly spread times (label 0), and for each unit whose median template
   reaches `0.4 · detect_threshold` on the channel, up to 200 of its snippets shifted to its peak
   there (label = unit, offset = shift). PCA of all of them (`max(12, 3 · channels in the
   neighbourhood)` components), and their nearest-neighbour index.
4. **Phase 2**, in chunks (100 M values per chunk, 1000-sample padding): detection with
   `detect_channel_radius` (50 µm) and `detect_time_radius_msec` (0.5 ms); every spike gets the
   label and offset of its **second** nearest training snippet (the first may be itself); label 0
   (noise) is dropped; spikes of one unit within the time radius of each other are reduced to the
   first (`remove_duplicate_events`).

## Preprocessing

As SpikeInterface's wrapper: band-pass 300–6000 Hz (Butterworth order 5, forward-backward), then
global ZCA whitening `W = U (Λ + ε)^(-1/2) Uᵀ` of the filtered data's second moment (uncentred, as
SpikeInterface's `whiten`), `ε = 10⁻¹⁶` as SpikeInterface for µV data (floored at 10⁻¹²). Optional switches as for the other sorters: `do_car`,
`do_bandpass` (no upper edge: a high-pass), `do_notch`, `do_whiten`.

## On the device

- Detection: threshold-reaching local extrema are found and compacted on the device; one thread per
  candidate scans its neighbourhood window in the trace. Only the kept detections are downloaded.
- Snippets are gathered on the device and kept on their masked channels only: outside the mask
  upstream stores zeros, so the numbers are the same in a fraction of the memory. For PCA they stay
  on the device and each batch of dense rows is rebuilt there.
- PCA (`dsp_base::linalg::TopComponents`): up to 8000 features the covariance is formed on the
  device and its leading eigenvectors computed exactly (subspace iteration until the leading
  values change by less than 10⁻⁶, or Jacobi when most components are wanted); above 8000 (a whole
  Neuropixels probe) the covariance is applied without forming it, for scikit-learn's randomized
  count of iterations.
- Template alignment: every pair's products at every shift as `T` matrix products.
- Classifiers: projections and the second-nearest training point of every spike, one thread per
  spike, after the spikes of a window are detected.
- On the host: isosplit6 (≤ 10-dimensional features of one subset at a time; pair tests and
  covariance sums in parallel, summed in a fixed order so runs repeat), medians, the subdivision
  tree.

## Choices of ours

| Where | Upstream | Here | Why |
|---|---|---|---|
| Whitening chunks | 20 random chunks of 10 000 samples | 20 evenly spaced chunks | reproducible without a seed |
| Equal samples in a detection window | every sample of a flat run is a detection | the first | device peak finder; exact ties of filtered floats |
| Two events at one sample | NumPy's unstable sort decides | the lowest channel | reproducible |
| Scheme 2 training stretch | chunks concatenated into one recording | each chunk its own segment (margins at its ends, no event across a junction) | no artificial waveform at the junctions |
| Scheme 2 unit templates | median over every channel | over the channels within `snippet_mask_radius_um` of the unit's peak channel | memory: a unit trains the classifiers of those channels only |
| Singular covariance in isosplit6 | the process aborts | a small ridge is added | a sorter should not exit |
| Randomized PCA (> 8000 features) | NumPy's random stream | our seeded stream | components equal up to the solver's tolerance, not bitwise |
| Classifier PCA | exact below 8000 features | randomized (scikit-learn's iteration count) | the nearest-neighbour distances depend only on the subspace; an exact solve per channel costs a full eigendecomposition (hundreds of components on a dense probe) |
| Spike amplitude and location | not reported | the whitened value at the detection, the detection channel's position | needed by the labels, metrics and exports |
