---
title: Metrics and curation
description: "Check units with metrics and edit them in the Curation workspace."
sidebar:
  order: 5
---

## In the app

![Curation: the unit table, similar units, waveforms, features, correlograms, ISI, amplitude over time and template of one unit](/dsp-kitchen/img/app/18-data-curation.png)

**Curation** shows the active view's sorting in views you arrange, or opens directly on a
`.sorting.zarr` or Phy folder as a standalone workspace without a loaded recording: use the left
**Affinity** rail and **Source** picker to move between units, and arrange each dock panel you can
drag, snap or close. The **Views** menu toggles each one, and the **Layout** menu applies a grid.

| View | Shows |
|---|---|
| Unit table | Sort by any column, filter by label |
| Waveforms | The selected units' waveforms |
| Features | PC1 against PC2 |
| Feature grid | PC1 of each of the lead unit's channels against each other |
| Firing rate | Spikes per second over the recording |
| Unit scatter | Every unit by amplitude and depth: click one to select it |
| Amplitude histogram | How many spikes at each amplitude (a cut-off distribution means missed spikes) |
| Correlograms | Auto-correlogram, and cross-correlograms between selected units |
| ISI | Inter-spike intervals |
| Amplitude | Amplitude over time |
| Template | The lead unit's template |
| Similarity | The units most like it: merge candidates |
| Probe | Units on the probe |

Under them, a stacked **Raster + Traces** bottom strip shows sorted spikes alongside the lead
unit's channels over the shared timeline.

Charts are interactive: wheel zooms (Shift for x, Alt for y), drag pans, and double-click resets. Each
has its settings under ⋯.

| Action | Keys |
|---|---|
| Label good / mua / noise | Alt+G / Alt+M / Alt+N |
| Merge the selected units | G |
| Split | Lasso on the features or amplitude view, then **Split**: every spike of the unit inside it is taken, not only those drawn |
| Unmerge | Select a merged unit, **Unmerge**: its units come back with their spikes and labels |
| Undo / redo | Ctrl+Z / Ctrl+Shift+Z |
| Save | Ctrl+S |

Edits are saved as a log, `dspk-curation.json`, inside the `.sorting.zarr`, applied on read. The
sorter's arrays are never rewritten. Each edit says when it was made (`at`, UTC), where (`via`:
`desktop` or `page`) and by whom (`by`: the person working in the app, see
[People](/dsp-kitchen/app/settings/#people); a page's server names its client; a guest is
`guest`). The file also says when it was saved and what wrote it, and keeps the edits undone and
not replaced, so redo works after a restart.

On a page served by [the container](/dsp-kitchen/deploy/image/#the-app-in-your-browser), the
server computes the waveforms and features (from the signal as the view processes it) and saves
the curation on its disk; a client paired without `--control` may look but not save. Before v0.0.11 the log was `curation.json`: it is still read,
and renamed at the next save. A `curation.json` written by phy or SpikeInterface (merge proposals)
is not touched.

### Coming from phy

The working format is the `.sorting.zarr`; a Phy / Kilosort folder goes in and out of it.

- **In:** *Attach sorting…* (Explore → Sorting) also takes a Phy folder. It is imported once as
  `<folder>.sorting.zarr` next to it; attaching the folder again opens the same store, with your edits.
- **Out:** *To Phy* in Curation writes the curated sorting as phy's own save does:
  `spike_clusters.npy`, `cluster_group.tsv` and `cluster_info.tsv`, the originals kept as `.bak`.
  A sorting that came from a Phy folder goes back into it, so the folder still opens in phy; any
  other sorting goes to a folder you pick.

```bash
dspk convert kilosort4/ rec.sorting.zarr                 # Phy folder → .sorting.zarr
dspk convert rec.sorting.zarr kilosort4/ --curated       # the curation, back into the folder
```

The keys are phy's by default, and can be changed in
[`keybindings.toml`](/dsp-kitchen/app/keyboard/#change-the-keys).

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
