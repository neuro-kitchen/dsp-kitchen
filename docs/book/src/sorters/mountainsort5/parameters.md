# MountainSort 5: parameters

Defaults are MountainSort 5's (`Scheme1SortingParameters`, `Scheme2SortingParameters`) and, where
the package leaves them to the caller, SpikeInterface's wrapper (`sorters/external/mountainsort5.py`).
Every setting is a field of `Mountainsort5Config` (Rust) and a keyword of `mountainsort5.Config`
(Python), whose signature shows the default.

| Setting (`Config`) | Upstream name | Default | Source | Used by |
|---|---|---|---|---|
| `scheme` | `scheme` | 2 | wrapper | 1: one pass; 2: train on a stretch, classify everything |
| `do_car` | — | off | ours | common average reference before filtering |
| `do_bandpass`, `bandpass_low_hz`, `bandpass_high_hz` | `filter`, `freq_min`, `freq_max` | on, 300, 6000 Hz | wrapper | band-pass (no upper edge: a high-pass) |
| `do_notch`, `notch_hz`, `notch_q` | — | off, 60 Hz, 30 | ours | line-noise notch |
| `do_whiten` | `whiten` | on | wrapper | whitening ("MountainSort5 expects whitened data") |
| `whitening_chunks`, `whitening_chunk_ms`, `whitening_epsilon` | `num_chunks_per_segment`, `chunk_duration`, `eps` | 20, 500 ms, 10⁻¹⁶ (floored at 10⁻¹²) | SpikeInterface `whiten` (µV data) | the chunks the whitening is fitted on |
| `detect_threshold` | `detect_threshold` | 5.5 | package | detection threshold (whitened units); scheme 2's phase 2 |
| `detect_sign` | `detect_sign` | −1 | package | negative peaks (+1 positive, 0 both) |
| `detect_time_radius_ms` | `detect_time_radius_msec` | 0.5 ms | package | events closer than this compete |
| `scheme1_detect_channel_radius_um` | `scheme1_detect_channel_radius` | 150 µm | wrapper | scheme 1's detection neighbourhood |
| `phase1_detect_channel_radius_um` | `scheme2_phase1_detect_channel_radius` | 200 µm | wrapper | phase 1's detection neighbourhood |
| `phase1_detect_threshold` | `phase1_detect_threshold` | 5.5 | package | phase 1's threshold |
| `phase1_detect_time_radius_ms` | `phase1_detect_time_radius_msec` | 1.5 ms | package | phase 1's time radius |
| `detect_channel_radius_um` | `scheme2_detect_channel_radius` | 50 µm | wrapper | phase 2's detection neighbourhood |
| `snippet_t1`, `snippet_t2` | `snippet_T1`, `snippet_T2` | 20, 20 samples | package | snippet window |
| `snippet_mask_radius_um` | `snippet_mask_radius` | 250 µm | wrapper (package: none) | snippet channels |
| `npca_per_channel` | `npca_per_channel` | 3 | package | first PCA: 3 components per channel |
| `npca_per_subdivision` | `npca_per_subdivision` | 10 | package | PCA of each isosplit6 subset |
| `skip_alignment` | `skip_alignment` | off | package | template alignment step |
| `max_num_snippets_per_training_batch` | `scheme2_max_num_snippets_per_training_batch` | 200 | package | noise snippets, and snippets per unit and channel |
| `classifier_npca` | `classifier_npca` | `max(12, 3 · channels)` | package | classifier PCA components |
| `training_duration_sec` | `scheme2_training_duration_sec` | 300 s | wrapper | training stretch (`None`: the whole recording) |
| `training_sampling` | `scheme2_training_recording_sampling_mode` | `"uniform"` | wrapper | 10 s chunks spread over the recording (`"initial"`: the start) |
| `classification_chunk_sec` | `classification_chunk_sec` | `10⁸ / channels` samples | package | samples per window |
| `isocut_threshold` | isosplit6 `isocut_threshold` | 2.0 | isosplit6 | dip score below this merges |
| `min_cluster_size` | isosplit6 `min_cluster_size` | 10 | isosplit6 | smaller clusters always merge |
| `k_init` | isosplit6 `K_init` | 200 | isosplit6 | initial parcels |
| `max_iterations_per_pass` | isosplit6 `max_iterations_per_pass` | 500 | isosplit6 | iterations per pass |
| `pca_exact_cap` | `pca_solver.py` `cov_cap` | 8000 | package | above this many features PCA is randomized |
| `seed` | `random_state` | 0 | package | the randomized PCA's start |
