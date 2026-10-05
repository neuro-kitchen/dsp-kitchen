# EMUsort: parameters

`EmusortConfig::default()` is `Kilosort4Config` with the paper's changes, plus EMUsort's own
settings. Everything not listed keeps its [Kilosort4 default](../kilosort4/parameters.md).

| Field | Upstream name | EMUsort | Kilosort4 |
|---|---|---|---|
| `kilosort4.n_pcs` | `n_pcs` | 9 | 6 |
| `kilosort4.n_templates` | `n_templates` | 9 | 6 |
| `kilosort4.th_single_ch` | `Th_single_ch` | `[6, 9, 12, 15]` | `[6]` |
| `kilosort4.nskip` | `nskip` | 2 | 25 |
| `kilosort4.do_car` | `do_CAR` | `false` | `true` |
| `remove_channel_delays` | `remove_chan_delays` | `true` | — |
| `remove_spike_outliers` | `remove_spike_outliers` | `true` | — |
| `hdbscan_min_cluster_size` | `hdbscan_min_cluster_size` | 20 | — |
| (`max_delay_samples(fs)`) | `fs // 500` | 2 ms | — |

Unchanged and worth knowing: `Th_universal = 9`, `Th_learned = 8`, `nt = 61`, `template_sizes = 5`,
`nearest_chans = 10`.

## Tuning notes

- EMUsort's command-line tool sweeps Kilosort parameters (`Th_universal`, `Th_learned`,
  `Th_single_ch`, `nt`, `nt0min`, `min_template_size`, `template_sizes`, `nearest_chans`,
  `nearest_templates`) per dataset; the defaults above are the starting point.
- MUAPs last longer than cortical spikes; at high sample rates `nt` (61 samples = 2 ms at 30 kHz)
  may need to grow, and `nt0min` follows it by default.
- Channel delays are bounded by ±2 ms; arrays with longer conduction paths need a larger bound.
