---
title: "MountainSort 5: tuning"
description: "Which settings to change to improve a MountainSort 5 sort, and how to check it worked."
sidebar:
  label: "Tuning"
  order: 9
---

:::note[Names on this page]
Names in code font are fields of the sorter's Rust configuration; `--show-config` prints them all with their values. The ones `dspk sort` accepts as flags are on the [settings page](/dsp-kitchen/plugins/synapse/reference/settings/mountainsort5/). The rest are reachable from Rust only, for now.
:::

MountainSort 5 has few settings by design: isosplit6 has no cluster count or tuning parameter.
What to change:

| Setting | Raise it | Lower it |
|---|---|---|
| `detect_threshold` (5.5) | fewer, larger spikes; small units missed | more small spikes and noise events |
| `snippet_mask_radius_um` (250 µm) | more channels per snippet: more context, more memory | fewer channels: faster, less separation of nearby units |
| `training_duration_sec` (300 s) | more spikes to cluster: better rare units, slower phase 1 | faster; rare units may be missed |
| `detect_channel_radius_um` (50 µm) | fewer duplicate detections of one spike | more events per spike on dense probes |
| `npca_per_subdivision` (10) | more dimensions for isosplit6 (slower, may separate similar units) | coarser clusters |
