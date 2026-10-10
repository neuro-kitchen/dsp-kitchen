# Tridesclous 2: parameters

Defaults are SpikeInterface's (`Tridesclous2Sorter._default_params` and the defaults of the
components it calls). Every setting is a field of `Tridesclous2Config` (Rust) and a keyword of
`tridesclous2.Config` (Python), whose signature shows the default.

| Setting (`Config`) | Upstream | Default |
|---|---|---|
| `do_bandpass`, `bandpass_low_hz`, `bandpass_high_hz`, `filter_order` | `freq_min`, `freq_max`, Bessel order | on, 150, 6000 Hz, 2 |
| `do_common_reference`, `common_reference_min_channels` | `common_reference` when ≥ 32 channels | on, 32 |
| `do_whiten`, `whitening_radius_um`, `whitening_chunks`, `whitening_chunk_ms`, `whitening_epsilon` | `whiten(mode="local", radius_um=100)` | on, 100 µm, 20, 500 ms, 10⁻¹⁶ |
| `noise_chunks`, `noise_chunk_ms` | `get_noise_levels` | 20, 500 ms |
| `detect_threshold`, `detection_radius_um`, `detection_exclude_sweep_ms` | `detect_threshold`, `detection_radius_um`, sweep | 5, 150 µm, 1.5 ms |
| `n_peaks_per_channel`, `min_n_peaks` | `n_peaks_per_channel`, 20 000 | 5000, 20 000 |
| `clustering_ms_before`, `clustering_ms_after` | `clustering_ms_before`, `clustering_ms_after` | 0.5, 1.5 ms |
| `features_radius_um`, `n_svd_components_per_channel`, `svd_peaks_fit` | `features_radius_um`, `n_svd_components_per_channel` | 120 µm, 5, 5000 |
| `split_radius_um`, `clustering_recursive_depth`, `min_size_split`, `n_pca_features` | `split_radius_um`, `clustering_recursive_depth`, split defaults, `n_pca_features` | 60 µm, 3, 25, 6 |
| `isosplit_n_init`, `isosplit_min_cluster_size`, `isosplit_max_iterations_per_pass`, `isocut_threshold` | isosplit clusterer | 15, 10, 500, 2.0 |
| `clustering_sparsify_threshold`, `clustering_min_snr` | `template_sparsify_threshold`, `template_min_snr_ptp` | 1.5, 3.5 |
| `merge_similarity`, `merge_similarity_lag_ms` | `merge_from_templates`, `merge_similarity_lag_ms` | 0.8, 0.5 ms |
| `min_firing_rate` | `min_firing_rate` | 0.1 Hz |
| `ms_before`, `ms_after`, `template_radius_um` | `ms_before`, `ms_after`, `template_radius_um` | 1.0, 2.5 ms, 100 µm |
| `template_sparsify_threshold`, `template_min_snr_ptp`, `template_max_jitter_ms` | same names | 1.5, 3.5, 0.2 ms |
| `peeler_*` | `tdc-peeler` defaults | sweep 0.8 ms, 80 / 150 / 150 µm, ±2 samples, 0.5 / 0.8 ms, 2 levels, [0.7, 1.4], fine detector on |
| `fine_detector_chunks` | matched filter random chunks | 5 |
| `final_merges`, `final_merge_max_distance_um`, `final_merge_censor_ms`, `final_merge_sparsity_overlap` | `final_cleaning_circus` | on, 50 µm, 3 ms, 0.5 |
| `chunk_sec`, `seed` | `chunk_duration`, `seed` | 1 s, 0 (upstream: none) |
