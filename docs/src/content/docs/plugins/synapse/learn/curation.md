---
title: Metrics and curation
description: "Check units with metrics and edit them in the Curation workspace."
sidebar:
  order: 5
---

## In the app

**Curation** shows the active view's sorting in views you arrange: each is a dock panel you can drag,
snap or close. The **Views** menu toggles each one, and the **Layout** menu applies a grid.

| View | Shows |
|---|---|
| Unit table | Sort by any column, filter by label |
| Waveforms | The selected units' waveforms |
| Features | PC1 against PC2 |
| Correlograms | Auto-correlogram, and cross-correlograms between selected units |
| ISI | Inter-spike intervals |
| Amplitude | Amplitude over time |
| Template | The lead unit's template |
| Similarity | The units most like it: merge candidates |
| Probe | Units on the probe |

Under them, the bottom strip shows the lead unit's channels over the shared timeline, or a raster.

Charts are interactive: wheel zooms (Shift for x, Alt for y), drag pans, and double-click resets. Each
has its settings under ⋯.

| Action | Keys |
|---|---|
| Label good / mua / noise | Alt+G / Alt+M / Alt+N |
| Merge the selected units | G |
| Split | Lasso on the features or amplitude view |
| Undo / redo | Ctrl+Z / Ctrl+Shift+Z |
| Save | Ctrl+S |

Edits are saved as a log, `curation.json`, inside the `.sorting.zarr`, applied on read. The sorter's
arrays are never rewritten.

## From Python

[`dspk.synapse.metrics`](/dsp-kitchen/plugins/synapse/reference/python/metrics/) has the analyses.
Defaults follow SpikeInterface or the cited papers.

| Question | Function |
|---|---|
| Is the unit refractory? | `isi_violations` (Hill et al., 2011), `llobet_contamination` (Llobet et al., 2022), `acg_refractory` |
| Is it present throughout? | `presence_ratio`, `firing_range` |
| Are spikes missed? | `amplitude_cutoff` |
| Is it well separated? | `silhouette`, `isolation_distance`, `d_prime`, `snr` |
| A single score | `composite_score` (EMUsort's) |
| Should two units merge? | `ccg_refractory`, [`suggest_merges`](/dsp-kitchen/plugins/synapse/reference/python/matching/#suggest_merges) |
| Do two sortings agree? | `compare_trains`, `compare_sortings` |
| Timing | `autocorrelogram`, `crosscorrelogram`, `firing_rate`, `bursts`, `psth`, `stimulus_average` |

```python
sorting = dspk.synapse.io.Sorting.load("sorted")
train = sorting.spike_train(3)
dspk.synapse.metrics.isi_violations(train, sorting.total_samples, sorting.sample_rate)
```
