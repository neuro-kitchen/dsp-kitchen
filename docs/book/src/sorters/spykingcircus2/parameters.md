# SpyKING CIRCUS 2: parameters

Defaults are SpikeInterface's (`Spykingcircus2Sorter._default_params` and the defaults of the
components it calls). Every setting is a field of `Spykingcircus2Config` (Rust) and a keyword of
`spykingcircus2.Config` (Python), whose signature shows the default.

| Setting (`Config`) | Upstream | Default | Used by |
|---|---|---|---|
| `ms_before`, `ms_after` | `general.ms_before`, `ms_after` | 0.5, 1.5 ms | waveform window (features, templates, prototype) |
| `radius_um` | `general.radius_um` | 100 µm | feature neighbourhood; detection uses half |
| `do_bandpass`, `bandpass_low_hz`, `bandpass_high_hz`, `filter_order` | `filtering` | on, 150, 7000 Hz, 2 | Bessel band-pass |
| `do_common_reference`, `common_reference_min_channels` | `common_reference` when ≥ 32 channels | on, 32 | common median reference |
| `do_whiten`, `whitening_radius_um` | `whitening.mode = "local"`, `radius_um` | on, 100 µm | local whitening |
| `whitening_chunks`, `whitening_chunk_ms`, `whitening_epsilon` | `whiten` random chunks, `eps` | 20, 500 ms, 10⁻¹⁶ | whitening fit |
| `noise_chunks`, `noise_chunk_ms` | `get_noise_levels` | 20, 500 ms | noise levels |
| `detect_threshold` | `detection.detect_threshold` | 5 | prototype and matched filtering |
| `prototype_peaks` | `n_peaks` of the prototype | 10 000 | prototype |
| `matched_filter_chunks` | `random_chunk_kwargs` | 5 | matched filter thresholds |
| `n_peaks_per_channel`, `min_n_peaks` | `selection` | 5000, 100 000 | peaks clustered |
| `svd_components`, `svd_peaks_fit` | `peaks_svd.n_components`, `n_peaks_fit` | 5, 5000 | features |
| `split_radius_um`, `split_depth`, `min_cluster_size`, `split_pca_features` | `split` | 75 µm, 3, 20, 3 | iterative HDBSCAN |
| `sparsify_threshold`, `min_snr`, `max_jitter_ms`, `mean_sd_ratio_threshold` | `cleaning` | 1, 5, 0.2 ms, 3 | template cleaning |
| `merge_similarity`, `merge_num_shifts` | `merge_from_templates` | 0.8, 3 | template merging |
| `min_firing_rate` | `min_firing_rate` | 0.1 Hz | small clusters |
| `omp_min_amplitude`, `omp_max_failures`, `omp_rank`, `omp_vicinity` | circus-omp | 0.6, 5, 5, 2 | matching |
| `final_merges`, `final_merge_max_distance_um`, `final_merge_censor_ms`, `final_merge_sparsity_overlap`, `final_merge_max_lag_ms` | `merging`, `final_cleaning_circus` | on, 50 µm, 3 ms, 0.5, 0.1 ms | final cleaning |
| `chunk_sec` | `job_kwargs.chunk_duration` | 1 s | windows |
| `seed` | `seed` | 42 | shuffles and selections |
