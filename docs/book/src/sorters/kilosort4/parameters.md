# Kilosort4: parameters

`Kilosort4Config::default()` holds the upstream defaults (`kilosort/parameters.py`). Thresholds
are in whitened σ.

| Field | Upstream name | Default | Used by |
|---|---|---|---|
| `nt` | `nt` | 61 | samples per waveform (odd) and batch padding |
| `nt0min` | `nt0min` | `int(20 · nt / 61)` = 20 | sample a waveform's peak is aligned to |
| `th_universal` | `Th_universal` | 9 | universal-template detection |
| `th_learned` | `Th_learned` | 8 | learned-template detection (not yet implemented) |
| `th_single_ch` | `Th_single_ch` | `[6]` | clips for learning universal templates |
| `templates_from_data` | `templates_from_data` | `true` | learn `wPCA` / `wTEMP` (else load `wTEMP.npz`) |
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

## Tuning notes

- **`Th_universal`** sets how many candidates reach clustering: lower finds smaller units and more
  noise. It is in whitened σ, so preprocessing must whiten.
- **`dmin` / `dminx`** should match the probe's contact spacing; the defaults suit Neuropixels 1.0
  and 2.0. Centres farther than `max_channel_distance` from any contact are dropped, so sparse
  probes may need a larger value.
- **`templates_from_data = false`** uses the predefined 6-template set: faster, but tuned for
  cortical extracellular spikes at 30 kHz.
