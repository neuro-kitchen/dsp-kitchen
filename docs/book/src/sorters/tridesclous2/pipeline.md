# Tridesclous 2: pipeline

Read from the source (`sorters/internal/tridesclous2.py`, `sortingcomponents/*`).

1. **Preprocessing.** Bessel band-pass 150–6000 Hz (order 2, forward-backward), common median
   reference on ≥ 32 channels, local whitening (100 µm); noise levels (MAD on 20 chunks of 500 ms).
2. **Detection.** `locally_exclusive` (5 × noise, negative, 150 µm, ±1.5 ms) over the whole
   recording; `max(5000 · channels, 20 000)` peaks drawn uniformly.
3. **Clustering** (`iterative-isosplit`). Local SVD (5 components, 0.5 / 1.5 ms, 120 µm); starting
   from each peak's channel, every cluster is split: its peaks' features on the channels common to
   their 60 µm neighbourhoods, reduced to 6 dimensions, clustered by SpikeInterface's isosplit
   (k-means start with 15 clusters, fewer for small sets; clusters under 10 points are noise), three
   levels deep. Templates: the **mean** of the SVD features mapped back; cleaning (channels with
   peak-to-peak / noise ≥ 1.5, SNR ≥ 3.5, trough within 0.2 ms); merging (l1 similarity > 0.8 over
   ±0.5 ms); clusters under 0.1 Hz dropped.
4. **Templates for the peeler.** Per unit, the channels within 100 µm of its peaks' channel
   barycentre; the mean waveform (1.0 ms before, 2.5 ms after) of its clustered peaks; cleaning again.
5. **Peeler.** Per chunk, levels: peaks of the residual (two levels with the fast detector, 80 µm,
   ±0.8 ms; then one with a matched filter), largest first; each assigned to the closest short
   template (0.5 / 0.8 ms) among the units within 150 µm, shifted by up to ±2 samples, its amplitude
   fitted with its neighbours by least squares; amplitudes in [0.7, 1.4] are subtracted.
6. **Final cleaning** (`auto_merge_units`, template differences 0.05 … 0.35, lag 0.5 ms).

## On the device

Preprocessing, detection candidates, SVD features and the mean templates on the device; the splits
(isosplit), templates' cleaning and merging, and the peeler on the host (the peeler subtracts
spike after spike: sequential within a window; windows in parallel).

## Choices of ours

| Where | Upstream | Here | Why |
|---|---|---|---|
| Seed | none (`seed=None`: every run differs) | `seed = 0` | reproducible |
| k-means start of isosplit | SciPy `kmeans2(minit="points")`, NumPy's stream | the same algorithm, our stream | same role |
| Whitening and noise chunks | random | evenly spaced | reproducible |
| Truncated SVDs | scikit-learn randomized | exact | equal up to the solver's tolerance |

## Upstream quirks kept

- SpikeInterface's isosplit adds each covariance diagonal term twice, labels the side below the cut
  point 2 (inverted), and skips a redistribution that would move a whole cluster's worth of points:
  its clusters are rarely redistributed, only merged.
- The peeler's matched-filter prototype is built in a loop that reads the last unit's template only
  (a variable left from the previous loop).
- A spike whose fitted amplitude is above the upper limit keeps amplitude 1 and is not subtracted.
- The template similarity's symmetric fill and lag sign (as in SpyKING CIRCUS 2).
