# MountainSort 5: tuning

MountainSort 5 has few settings by design: isosplit6 has no cluster count or tuning parameter.
What to change:

| Setting | Raise it | Lower it |
|---|---|---|
| `detect_threshold` (5.5) | fewer, larger spikes; small units missed | more small spikes and noise events |
| `snippet_mask_radius_um` (250 µm) | more channels per snippet: more context, more memory | fewer channels: faster, less separation of nearby units |
| `training_duration_sec` (300 s) | more spikes to cluster: better rare units, slower phase 1 | faster; rare units may be missed |
| `detect_channel_radius_um` (50 µm) | fewer duplicate detections of one spike | more events per spike on dense probes |
| `npca_per_subdivision` (10) | more dimensions for isosplit6 (slower, may separate similar units) | coarser clusters |
