# SpyKING CIRCUS 2: pipeline

Read from the source (`sorters/internal/spyking_circus2.py`, `sortingcomponents/*`).

1. **Preprocessing.** Bessel band-pass (order 2 per edge, 150–7000 Hz, forward-backward), common
   **median** reference when the recording has at least 32 channels, then **local** whitening:
   each channel whitened over the channels within 100 µm (the ZCA row of that neighbourhood's
   covariance, uncentred, ε = 10⁻¹⁶). Noise levels: on 20 chunks of 500 ms, each channel's median
   absolute deviation (around its median, / 0.6745), averaged over the chunks.
2. **Prototype.** `locally_exclusive` peaks (5 × noise, negative; a channel's peak survives when no
   channel within 50 µm has a larger value / threshold within ±1.5 ms), 10 000 of them; their
   waveforms on their own channel (0.5 ms before, 1.5 ms after); the per-sample median of the
   waveforms divided by their peak's magnitude.
3. **Matched filtering.** Every channel correlated with the normalised prototype; for each channel
   and each of five source depths (0–120 µm), a weighted sum over the nearby channels (weights
   `exp(−√(d² + z²) / 2.5)`, normalised, small ones dropped); thresholds 5 × each row's MAD on 5
   random chunks; local maxima above them, locally exclusive over channels within 50 µm.
   Detection stops after `max(100 000, 5000 · channels)` peaks over shuffled chunks.
4. **Selection and features.** That many peaks drawn uniformly; a truncated SVD (5 components, no
   centring) fitted on 5000 of their own-channel waveforms; every peak's components on each channel
   within 100 µm.
5. **Iterative HDBSCAN.** Starting from each peak's channel, every cluster is split: the channels
   common to all its peaks' 75 µm neighbourhoods, the peaks' features there, reduced to 3 dimensions,
   HDBSCAN (min cluster size 20, a single cluster allowed); clusters that split are split again, three
   levels deep.
6. **Templates.** Per cluster, on its most frequent channel's neighbourhood, the median of each SVD
   component mapped back to a waveform. **Cleaning**: channels kept where peak-to-peak / noise ≥ 1;
   templates dropped if empty, if their trough is more than 0.2 ms off, below SNR 5, or noisier than
   3 × the noise on average. **Merging**: templates with l1 similarity > 0.8 (over ±3 samples)
   joined. Then templates and cleaning again, and clusters firing below 0.1 Hz dropped.
7. **circus-omp.** Templates compressed to rank 5 and normalised; their products with the data at
   every sample; repeatedly the best template at local maxima of the products (within two template
   widths, product / norm > 0.25), its amplitude and those of nearby selected spikes re-solved by a
   growing Cholesky factor, the change subtracted from the products through the templates' overlaps;
   stops after 5 rounds with no new spike of amplitude > 0.6. In 1 s chunks.
8. **Final cleaning** (`auto_merge_units`, preset `x_contaminations` at template differences 0.05,
   0.10, … 0.45, each repeated while it merges). Pairs of units, both with at least 100 spikes and a
   refractory contamination (1 ms, 0.3 ms censored) of at most 0.2, within 50 µm, whose template
   difference is under the threshold, whose cross-contamination test passes (p > 0.2 against 10%)
   and whose merge does not lower `firing rate · (1 − 2.5 · contamination)`, are joined into connected
   components; a group whose channels overlap by at least half merges: spikes closer than 3 ms
   dropped, the template the count-weighted average on the shared channels.

## On the device

- Filters, reference and whitening (the shared pipeline); detection candidates, the matched filter's
  correlation and spatial sums, the SVD transform, the matching's scalar products: on the device.
- On the host: the locally exclusive selection among candidates, HDBSCAN's hierarchy (its distances
  on the device), the splits' small SVDs, templates and their cleaning, and the pursuit itself.

## Choices of ours

| Where | Upstream | Here | Why |
|---|---|---|---|
| Whitening and noise chunks | random | evenly spaced | reproducible without a seed |
| Shuffled chunk order (prototype, detection) | NumPy's random stream | our seeded shuffle | same role, not the same chunks |
| Matched filter thresholds | random chunks concatenated, then filtered | each chunk filtered on its own | no artefact at the joins |
| Truncated SVDs (features, splits) | scikit-learn randomized | exact (host eigendecomposition) | equal up to the solver's tolerance |
| Motion correction | on for dense probes | not yet | drift correction is S10 |

The cross-contamination p-value is SciPy's binomial survival function interpolated by a quadratic
spline over the integers around `n` (upstream's `binom_sf`); here the incomplete beta function and
the same spline (equal to SciPy to 10⁻⁹ on the tests' values).

## Upstream quirks kept

The template similarity used for merging compares the template array with itself: upstream fills
both orientations of a pair, and both signs of a lag, with the value at the negative lag, so only a
later template shifted earlier is aligned and every lag is ≤ 0; the merge then shifts members by
`lags[member, first]` as if lags were antisymmetric. Both are kept, so the merges match upstream's.
