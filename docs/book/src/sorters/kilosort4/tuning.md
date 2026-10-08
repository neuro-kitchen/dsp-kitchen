# Kilosort4: tuning

Which parameters to change, in which direction, and how to tell whether a change helped. The
defaults (see [Parameters](parameters.md)) are Kilosort4's and suit Neuropixels-like probes at
30 kHz; most recordings sort well without changes. Change one group at a time and compare with the
checks at the end of this page.

## Detection sensitivity

| Parameter | Lower | Higher |
|---|---|---|
| `th_universal` (`Th_universal`, 9) | more candidates reach clustering: smaller units found, more noise clusters | fewer, larger units; small units missed |
| `th_learned` (`Th_learned`, 8) | template matching keeps weaker matches: more spikes per unit, more contamination; if units show gaps over time, lower it | cleaner, sparser units |
| `th_single_ch` (`Th_single_ch`, `[6]`) | more (noisier) clips to learn the universal templates from | fewer, cleaner clips; too few and the templates miss shapes |

Thresholds are in whitened σ: they only mean the same thing across recordings if the preprocessing
whitens (it does by default).

## Waveform window

`nt` (61) is in **samples**: 2 ms at 30 kHz, 2.5 ms at 24.4 kHz. Set it to cover the whole waveform;
a waveform cut by the window loses part of its identity, and matching pursuit then matches its later
phases as new spikes. Keep the duration, not the sample count, when the sampling rate changes
(5 ms is 121 samples at 24.4 kHz but 151 at 30 kHz; `nt` must be odd). `nt0min` (where the trough
sits) defaults to `int(20 · nt / 61)` ≈ `nt / 3` and follows `nt`.

A quick check: the share of each unit template's energy in the outer 10% of the window (measured in
`playground/benchmarks/emusort_checks.py`). A few percent or more means the window is too short.

## Probe geometry

The spatial templates are Gaussians around virtual positions; these decide how many channels a
template spans.

| Parameter | Effect |
|---|---|
| `centres.dmin` / `centres.dminx` | spacing of the virtual positions (`dmin` defaults to the median vertical contact spacing; `dminx` 32 µm). Match the probe. |
| `centres.min_template_size` (10 µm), `centres.template_sizes` (5) | widths `10, 20, …, 50 µm`: how far a template reaches. Waveforms spanning more channels need wider templates; far-apart contacts (> ~50 µm) make every template collapse onto one contact. |
| `centres.nearest_chans` (10) | channels per template: the largest number a waveform spans. |
| `centres.nearest_templates` (100) | neighbouring positions in the local-maximum test. Keep it at most the number of positions; on small probes, at most the number of channels. |
| `centres.max_channel_distance` (32 µm) | positions farther from every contact are dropped; sparse probes need it larger. |
| `whitening_range` (32) | channels per local whitening neighbourhood; on probes of a few dozen channels, use all of them. |

## Clustering

| Parameter | Effect |
|---|---|
| `clustering.graph.subset_stride` (`cluster_downsampling`, 20) | spikes per right node of the graph. 1 is exact and slow; larger is faster and coarser. |
| `clustering.graph.neighbours` (`cluster_neighbors`, 10) | graph neighbourhood: larger merges more readily, smaller splits more readily. |
| `clustering.graph.max_subset` (`max_cluster_subset`, 25 000) | caps the right nodes per section, so long recordings keep the same neighbourhood scale. |
| `clustering.section_um` (40 µm) | height of a clustering section; one neuron should fit in one or two. |
| `clustering.modularity_split` (0.2), `clustering.bimodality_split` (0.6) | how readily the merging tree splits (Kilosort4's fixed values). Lower bimodality splits more. |
| `template_merge.min_similarity` (0.9), `max_norm_difference` (0.2) | how similar two learned templates must be to merge before matching. |
| `max_peels` (50) | matching-pursuit rounds per batch; more resolves more overlapping spikes in dense recordings. |

Not implemented yet but part of Kilosort4: `acg_threshold` (0.2) and `ccg_threshold` (0.25), the
refractory criteria for splits and merges.

## Preprocessing and sampling of the recording

| Parameter | Effect |
|---|---|
| `highpass_cutoff_hz` (300 Hz) | lower keeps slower waveform components and more low-frequency noise. |
| `do_car` (`true`) | the common average reference removes signals shared by all channels; turn it off when real signals span most channels (EMG). |
| `nskip` (25) | every `nskip`-th batch fits the whitening and learns the templates; lower uses more data (slower, more representative). On a short recording, one batch may be all it uses. |
| `batch_size` (60 000) | samples per batch (2 s at 30 kHz). Rarely worth changing. |

`reproducible` (default `true`) does not change quality: it makes repeated runs identical.

## How to tell whether a change helped

- **With a reference sorting** (simulated ground truth, or another sorter's curated output): per-unit
  accuracy, matching spikes in time **and** place (time alone matches by chance on dense probes);
  `playground/benchmarks/kilosort4_validation.py` does this against Kilosort4's saved results.
- **Without one:** refractory-period violations per unit (fraction of inter-spike intervals under
  ~1.5–2 ms), presence across the recording, template truncation (above), the number of units with few
  spikes, and visual inspection (`playground/sorters/inspection.py`: a channel's trace with each unit's
  spikes and template).
