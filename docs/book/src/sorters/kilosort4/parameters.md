# Kilosort4: parameters

`Kilosort4Config::default()` holds the upstream defaults (`kilosort/parameters.py`). Thresholds
are in whitened σ.

In Python, every setting with its type, default and unit:
[`dsp_kitchen.synapse.ml.kilosort4`](../../../api/python/sorters/kilosort4/index.html) (Python API).

| Field | Upstream name | Default | Used by |
|---|---|---|---|
| `nt` | `nt` | 61 | samples per waveform (odd) and batch padding |
| `nt0min` | `nt0min` | `int(20 · nt / 61)` = 20 | sample a waveform's peak is aligned to |
| `th_universal` | `Th_universal` | 9 | universal-template detection |
| `th_learned` | `Th_learned` | 8 | learned-template matching |
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
| `highpass_cutoff_hz` | `highpass_cutoff` | 300 Hz | high-pass filter |
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
| `template_merge.min_similarity` | (fixed upstream) | 0.9 | learned templates this correlated are merged… |
| `template_merge.max_norm_difference` | (fixed upstream) | 0.2 | …if their norms differ by less than this |
| `max_peels` | `max_peels` (paper: 50) | 50 | matching pursuit rounds per batch |
| `reproducible` | (ours) | `true` | pin result-changing tuned choices: the same input on the same device gives the same sort |

In Python, `kilosort4.Config` exposes these under the upstream names where there is one
(`cluster_downsampling`, `cluster_neighbors`, `max_cluster_subset`, `reproducible`, …).

## Tuning

Which of these to change to improve a sort, and how to check the result: [Kilosort4 tuning](tuning.md).
