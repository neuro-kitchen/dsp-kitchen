# EMUsort: parameters

`EmusortConfig::default()` is `Kilosort4Config` with the paper's changes, plus EMUsort's own
settings. Everything not listed keeps its [Kilosort4 default](../kilosort4/parameters.md).

In Python, every setting with its type, default and unit:
[`dsp_kitchen.synapse.ml.emusort`](../../../api/python/sorters/emusort/index.html) (Python API).

| Field | Upstream name | EMUsort | Kilosort4 |
|---|---|---|---|
| `kilosort4.n_pcs` | `n_pcs` | 9 | 6 |
| `kilosort4.n_templates` | `n_templates` | 9 | 6 |
| `kilosort4.th_single_ch` | `Th_single_ch` | `[6, 9, 12, 15]` | `[6]` |
| `kilosort4.nskip` | `nskip` | 2 | 25 |
| `kilosort4.do_car` | `do_CAR` | `false` | `true` |
| (`learn_options().clip_scaling`) | (fork: `clips /= (clips**2).sum(1).std()**.5`) | `Common` | `PerClip` |
| `remove_channel_delays` | `remove_chan_delays` | `true` | — |
| `remove_spike_outliers` | `remove_spike_outliers` | `true` | — |
| `hdbscan_min_cluster_size` | `hdbscan_min_cluster_size` | 20 | — |
| (`max_delay_samples(fs)`) | `fs // 500` | 2 ms | — |

Unchanged and worth knowing: `Th_universal = 9`, `Th_learned = 8`, `nt = 61`, `template_sizes = 5`,
`nearest_chans = 10`.

## Tuning

Which of these to change to improve a sort, and how to check the result: [EMUsort tuning](tuning.md).
