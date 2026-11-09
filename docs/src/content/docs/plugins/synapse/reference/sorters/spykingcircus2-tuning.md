---
title: "SpyKING CIRCUS 2: tuning"
description: "Which settings to change to improve a SpyKING CIRCUS 2 sort, and how to check it worked."
sidebar:
  label: "Tuning"
  order: 12
---

:::note[Names on this page]
Names in code font are fields of the sorter's Rust configuration; `--show-config` prints them all with their values. The ones `dspk sort` accepts as flags are on the [settings page](/dsp-kitchen/plugins/synapse/reference/settings/spykingcircus2/). The rest are reachable from Rust only, for now.
:::

| Setting | Raise it | Lower it |
|---|---|---|
| `detect_threshold` (5) | fewer, larger peaks to cluster | more small units, more noise peaks |
| `min_cluster_size` (20) | fewer, larger clusters | more small clusters (also slower) |
| `merge_similarity` (0.8) | fewer merges: units split across templates stay split | more merges: similar units joined |
| `min_snr` (5) | only large templates are matched | small units kept (more false ones) |
| `omp_min_amplitude` (0.6) | only spikes close to their template's size | more small matches (and noise) |
| `n_peaks_per_channel` (5000) | more peaks clustered: rare units found, slower | faster, rare units missed |

Check a change with the unit labels and composite score of the sorting (`to_sorting_output`), and
against another sorter on the same recording (*Benchmarks*).
