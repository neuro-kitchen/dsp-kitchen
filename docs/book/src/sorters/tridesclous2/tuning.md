# Tridesclous 2: tuning

| Setting | Raise it | Lower it |
|---|---|---|
| `detect_threshold` (5) | fewer, larger peaks clustered and peeled | more small units, more noise peaks |
| `n_pca_features` (6) | more dimensions per split: finer clusters, slower | coarser clusters |
| `isocut_threshold` (2.0) | fewer splits (more merging in isosplit) | more splits |
| `merge_similarity` (0.8) | fewer template merges | more merges of similar units |
| `peeler_amplitude_min` (0.7) | only spikes close to their template's size | more partial matches |
| `n_peaks_per_channel` (5000) | more peaks clustered: rare units, slower | faster |

Check a change with the labels and composite score of the sorting, and against another sorter on
the same recording (*Benchmarks*).
