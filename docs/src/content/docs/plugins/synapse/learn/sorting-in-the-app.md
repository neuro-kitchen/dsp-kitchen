---
title: Sorting in the app
description: "Run sorters and browse their units in the Sorting workspace."
sidebar:
  order: 4
---

The **Sorting** workspace runs the installed sorters on the active view, or on a part of it, and shows
what they found, channel by channel.

| Area | What it holds |
|---|---|
| Left | The sorter, the part of the recording (start, duration), the probe, the view's runs |
| Right | The sorter's parameters, each with its default shown; typed values replace it |
| Centre | The units of the selected channel as cards (waveform mean ± sd, ISI, spikes), their clusters (PC1 · PC2), and the channel table |
| Bottom | The channel's traces or a raster over the shared timeline |

## A first sort

![The Sorting workspace: sorter, part of the recording, probe and parameters](/dsp-kitchen/img/app/09_sorting-menu.png)

1. Open a recording, and pick the view to sort.
2. Choose a sorter and a short part of the recording, such as 30 seconds.
3. **Run.** Progress shows like the app's other jobs.
4. Click a run to show it everywhere. Compare two runs side by side: units, spikes, labels and matched
   units.

![A Kilosort4 run in progress, with its stage and steps](/dsp-kitchen/img/app/11_running-kilosort.png)

![The templates of the units a run found, above the traces](/dsp-kitchen/img/app/12_kilosort-output.png)

![The Compare runs tab](/dsp-kitchen/img/app/13_compare-different-runs.png)

The sorting is linked to the view and saved as a `.sorting.zarr`.

No sorter yet? **Install sorters** in the Sorting workspace downloads `dspk-cli-sorter`, the
program that runs them, next to the app.

To sort on a server, open the recording from it
([Settings and servers](/dsp-kitchen/app/settings/)), or use the page the container serves. The
run happens there, with the server's sorters.

A run is named with the time it started (`rec.kilosort4.0-20s.20261117-2032.sorting.zarr`, UTC).
Running the same sorter over the same part again asks first: **Run again** keeps the new run
beside the earlier one (nothing is overwritten), **Open that run** shows the earlier one. On a
server, a sort has the GPU to itself: views in progress finish, and the page's views are processed
on the server's CPU until the sort ends.

Judging and editing units is the [Curation](/dsp-kitchen/plugins/synapse/learn/curation/) workspace's.
