---
title: App
description: "The desktop app: look at recordings and build pipelines without writing code."
sidebar:
  order: 0
---

`dspk-app` is a desktop workbench for long multi-channel recordings. It draws hours of signal from
[pyramids](/dsp-kitchen/learn/concepts/pyramids/), and previews processing live on your GPU.

![dspk-app showing 385 channels of a Neuropixels recording as traces and a heatmap](/dsp-kitchen/img/dsp-app_main-darkmode.png)

Install it with `cargo install dspk-app` ([Linux needs a few system libraries](/dsp-kitchen/learn/get-started/install/#the-desktop-app-on-linux)).

## Workspaces

The tabs along the top are workspaces. They share one session and one timeline.

| Workspace | What it's for |
|---|---|
| [Explore](/dsp-kitchen/app/explore/) | Look at the recording: traces, heatmaps, channels, timeline |
| [Pipeline](/dsp-kitchen/app/pipeline/) | The processing graph: recordings, operations, views and writes on a canvas |
| [Sorting](/dsp-kitchen/plugins/synapse/learn/sorting-in-the-app/) | Run the installed sorters and compare their units (Synapse plugin) |
| [Curation](/dsp-kitchen/plugins/synapse/learn/curation/) | Judge a sorting: label, merge, split (Synapse plugin) |
| Analytics | Charts across recordings and runs; still to be built |

![The start screen, with Open file, Open folder and Synthetic recording](/dsp-kitchen/img/app/00_main-dark.png)

## Start here

1. **File → Open file…** (Ctrl+O) for a `.bin` or `.cbin`, **Open folder…** (Ctrl+Shift+O) for a Zarr store, or **Synthetic recording** to try it with no data.
2. Follow [Filter and inspect a recording](/dsp-kitchen/app/workflows/).

Press **F1** at any time for the [keyboard and mouse list](/dsp-kitchen/app/keyboard/), and **Ctrl+K**
for the command palette: any action, view, operation or recipe by name.
