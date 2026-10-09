# Kilosort4: parameters

`Kilosort4Config::default()` holds the upstream defaults (`kilosort/parameters.py`). Thresholds
are in whitened σ.

In Python, every setting with its type, default and unit:
[`dsp_kitchen.synapse.ml.kilosort4`](../../../api/python/sorters/kilosort4/index.html) (Python API).

| Field | Upstream name | Default | Used by |
|---|---|---|---|
| `nt` | `nt` | 61 | samples per waveform (odd) and batch padding |
| `nt_ms` | (ours) | `None` | waveform length in ms: `nt = round(nt_ms · fs / 1000)` made odd, per recording |
| `nt0min` | `nt0min` | `int(20 · nt / 61)` = 20 | sample a waveform's peak is aligned to |
| `th_universal` | `Th_universal` (9) | 10 | universal-template detection |
| `th_learned` | `Th_learned` (8) | 9 | learned-template matching |
| `th_single_ch` | `Th_single_ch` | `[6]` | clips for learning universal templates |
| `templates_from_data` | `templates_from_data` | `true` | learn `wPCA` / `wTEMP` (else the predefined `wTEMP.npz`: `RunPlan::templates` / Python `templates=`, or the hub) |
| `n_templates` | `n_templates` | 6 | universal templates learned |
| `n_pcs` | `n_pcs` | 6 | temporal PCs learned |
| `nskip` | (fixed 25 in `spikedetect.run`) | 25 | batch stride for learning |
| `centres.dmin` | `dmin` | `None` (median vertical spacing) | template-centre rows |
| `centres.dminx` | `dminx` | 32 µm | template-centre columns |
| `centres.max_channel_distance` | `max_channel_distance` | 32 µm | drop centres far from contacts |
| `centres.min_template_size` | `min_template_size` | 10 µm | smallest spatial envelope |
| `centres.template_sizes` | `template_sizes` | 5 | spatial envelope widths |
| `centres.nearest_chans` | `nearest_chans` | 10 | contacts per centre |
| `centres.nearest_templates` | `nearest_templates` | 100 | neighbouring centres for local maxima |
| `do_car` | (preprocessing) | `true` | common average reference |
| `do_bandpass` | (ours) | `true` | one Butterworth band-pass (order 3 per edge, forward-backward); off: no Butterworth |
| `bandpass_low_hz` | `highpass_cutoff` | 300 Hz | lower band edge (upstream: the high-pass cutoff) |
| `bandpass_high_hz` | (none upstream) | 6000 Hz | upper band edge, the action-potential band `NeuralBand::Ap`; below Nyquist; `None`: no upper edge, a high-pass at `bandpass_low_hz` (upstream Kilosort4's filter) |
| `do_notch`, `notch_hz`, `notch_q` | (ours; EMUsort's notch) | `false`, 60 Hz, 30 | line-noise notch after the band |
| `whitening_range` | `whitening_range` | 32 | channels per whitening neighbourhood |
| `batch_size` | `batch_size` | 60000 | samples per batch |
| `clustering.section_um` | (40 µm sections, paper) | 40 µm | height of a clustering section |
| `clustering.min_section_spikes` | (fixed upstream) | 1000 | sections with fewer spikes are one unit |
| `clustering.graph.subset_stride` | `cluster_downsampling` | 20 | every n-th spike is a right node of the graph |
| `clustering.graph.neighbours` | `cluster_neighbors` | 10 | neighbours of each spike |
| `clustering.graph.max_subset` | `max_cluster_subset` | 25000 | at most this many right nodes per section |
| `clustering.graph.init_clusters` | (paper: 200) | 200 | k-means++ seeds of the assignment |
| `clustering.modularity_split` | (paper: 0.2) | 0.2 | merging-tree nodes below this γ̂ are split |
| `clustering.bimodality_split` | (fixed upstream) | 0.6 | halves scoring this bimodality are split |
| `clustering.refractory` | `ccg_threshold`, `acg_threshold` (0.25, 0.2 as Kilosort4 runs; probabilities 0.05, 0.2) | `RefractoryOptions::default()` | correlogram tests: CCG of halves in the final clustering and in merges, ACG for labels; 1 ms bins over ±0.5 s, `k` ≤ 10 bins, shoulders beyond 250 ms (ours) |
| `template_merge.min_similarity` | (fixed upstream) | 0.9 | learned templates this correlated are merged… |
| `template_merge.max_norm_difference` | (fixed upstream) | 0.2 | …if their norms differ by less than this |
| `max_peels` | `max_peels` (paper: 50) | 50 | matching pursuit rounds per batch |
| `global_merge.enabled` | (always upstream) | `true` | global merges of the final units |
| `global_merge.min_similarity` | (paper: 0.5) | 0.5 | waveform similarity a pair needs to be tested |
| `duplicate_spike_ms` | `duplicate_spike_ms` | 0.25 ms | a unit's spikes this close to its previous spike are removed |
| `reproducible` | (ours) | `true` | pin result-changing tuned choices: the same input on the same device gives the same sort |

In Python, `kilosort4.Config` exposes these under the upstream names where there is one
(`cluster_downsampling`, `cluster_neighbors`, `max_cluster_subset`, `reproducible`, …), plus
`global_merges`, `merge_similarity`, `duplicate_spike_ms`, `nt_ms`, `do_bandpass`,
`bandpass_low_hz`, `bandpass_high_hz`, `do_notch`, `notch_hz`, `notch_q`. The command line's `dsp-cli sort kilosort4 --help` lists the same settings
with their defaults, and `--show-config` prints every setting of a run.

**Thresholds and the band.** Upstream Kilosort4 high-passes at 300 Hz with no upper edge and detects
at 9 / 8 whitened σ. Our default band stops at 6 kHz (`NeuralBand::Ap`), which lowers the whitened
noise: at 9 / 8 it detected 36% more spikes and 80 more units than Kilosort4 on the test recording.
At 10 / 9 the agreement with Kilosort4's saved results matches that of the high-pass at 9 / 8
(130 225 spikes, 291 units; good units: median accuracy 0.82, 72% ≥ 0.5); 11 / 10 misses spikes.
For upstream's filter, set `bandpass_high_hz = None` and the thresholds to 9 / 8
(`UPSTREAM_TH_UNIVERSAL`, `UPSTREAM_TH_LEARNED`).

## Tuning

Which of these to change to improve a sort, and how to check the result: [Kilosort4 tuning](tuning.md).
