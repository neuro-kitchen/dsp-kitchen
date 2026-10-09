# EMUsort: parameters

`EmusortConfig` lists **every** setting of the run, with EMUsort's defaults: the settings it
shares with Kilosort4 (same names and meaning) and EMUsort's own. There is no nested Kilosort4
config to replace, so a run always uses what is in the EMUsort config (in Python,
`emusort.Config(*, nt=…, do_car=False, …)` shows each default). Below, the settings whose defaults
differ from [Kilosort4's](../kilosort4/parameters.md); all others take Kilosort4's values.

In Python, every setting with its type, default and unit:
[`dsp_kitchen.synapse.ml.emusort`](../../../api/python/sorters/emusort/index.html) (Python API).

| Field | Upstream name | EMUsort | Kilosort4 |
|---|---|---|---|
| `n_pcs` | `n_pcs` | 9 | 6 |
| `n_templates` | `n_templates` | 9 | 6 |
| `th_single_ch` | `Th_single_ch` | `[6, 9, 12, 15]` | `[6]` |
| `nskip` | `nskip` | 2 | 25 |
| `do_car` | `do_CAR` | `false` | `true` |
| `bandpass_high_hz` | (paper: band-pass to 5000 Hz) | 5000 Hz (`NeuralBand::Emg`) | 6000 Hz (`NeuralBand::Ap`) |
| `do_notch` | (paper: 60 Hz notch) | `false` ([why](tuning.md#line-noise-notch-off-by-default)) | `false` |
| `th_universal`, `th_learned` | `Th_universal`, `Th_learned` | 9, 8 (upstream) | 10, 9 |
| (`learn_options().clip_scaling`) | (fork: `clips /= (clips**2).sum(1).std()**.5`) | `Common` | `PerClip` |
| `remove_channel_delays` | `remove_chan_delays` | `true` | — |
| `remove_spike_outliers` | `remove_spike_outliers` | `true` | — |
| `hdbscan_min_cluster_size` | `hdbscan_min_cluster_size` | 20 | — |
| (`max_delay_samples(fs)`) | `fs // 500` | 2 ms | — |

Unchanged and worth knowing: `Th_universal = 9`, `Th_learned = 8`, `nt = 61`, `template_sizes = 5`,
`nearest_chans = 10`.

## Tuning

Which of these to change to improve a sort, and how to check the result: [EMUsort tuning](tuning.md).
