---
title: Synapse
description: "Spike sorting on the GPU: the first dsp-kitchen plugin."
sidebar:
  label: Overview
  order: 0
---

Synapse adds electrophysiology on top of the [core](/dsp-kitchen/learn/): neural file formats, probes,
spike detection, clustering, sorting, metrics and curation.

## Sorters

Five sorters run on the GPU or the CPU, from the raw file to sorted units:

| Sorter | For | Licence |
|---|---|---|
| [Kilosort4](/dsp-kitchen/plugins/synapse/reference/sorters/kilosort4/) | High-density probes (Neuropixels, polytrodes) | GPL-3.0 |
| [EMUsort](/dsp-kitchen/plugins/synapse/reference/sorters/emusort/) | Motor units on high-density intramuscular arrays | GPL-3.0 |
| [MountainSort 5](/dsp-kitchen/plugins/synapse/reference/sorters/mountainsort5/) | Any geometry, tetrodes to dense probes | Apache-2.0 |
| [SpyKING CIRCUS 2](/dsp-kitchen/plugins/synapse/reference/sorters/spykingcircus2/) | Dense probes | MIT or Apache-2.0, with SpikeInterface's MIT notice |
| [Tridesclous 2](/dsp-kitchen/plugins/synapse/reference/sorters/tridesclous2/) | Dense probes | MIT or Apache-2.0, with SpikeInterface's MIT notice |

:::caution
These are reimplementations in progress. Until a sorter reproduces its paper's results on the paper's
data, it isn't a substitute for the original. Drift correction isn't implemented yet. See
[Validation](/dsp-kitchen/plugins/synapse/validation/).
:::

## Where to go

| You want to | Go to |
|---|---|
| Open neural files and describe a probe | [Formats and probes](/dsp-kitchen/plugins/synapse/learn/formats-and-probes/) |
| Find spikes | [Detection](/dsp-kitchen/plugins/synapse/learn/detection/) |
| Sort a recording | [Sorting](/dsp-kitchen/plugins/synapse/learn/sorting/) |
| Sort and browse units in the app | [Sorting in the app](/dsp-kitchen/plugins/synapse/learn/sorting-in-the-app/) |
| Check and edit units | [Metrics and curation](/dsp-kitchen/plugins/synapse/learn/curation/) |
| Work through examples | [Course: Detection](/dsp-kitchen/plugins/synapse/learn/course/detection/threshold-and-scoring/) |
