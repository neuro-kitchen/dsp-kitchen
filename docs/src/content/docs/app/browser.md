---
title: In your browser
description: "The viewer at /dsp-kitchen/viewer/: open a recording from your disk, nothing to install."
sidebar:
  order: 1
---

**[Open the viewer](/dsp-kitchen/viewer/)**: the same app as the desktop's, built for
the browser. Nothing is installed and nothing leaves your computer: the files you open are read in
the page, and processing runs on your own graphics card.

## Open a recording

**Drop it on the page**, or use the buttons:

- **Open file…**: the file chooser, as for a PDF (no question from the browser): a `.wav`, or a
  `.bin` with its `.meta` (select both).
- **Open project…**: a project, or a Zarr recording (`.zarr`, `.nwb.zarr`). A Zarr recording
  is stored as a folder, so your browser asks once to let the page read it ("View files"):
  nothing is uploaded, the page reads it where it is. Dropping it on the page skips that question.

One recording opens at once; only a project holding several asks which. The viewer reads:

- **Zarr recordings** (`.zarr`, as `dspk convert` and `dspk run` write them: zstd, Blosc-LZ4,
  gzip or uncompressed, sharded or not). Only the chunks a view needs are read.
- **NWB as Zarr** (`.nwb.zarr`): its largest electrical series in `/acquisition` (or its largest
  continuous series), scaled to µV, with the channel names of its electrodes table. NWB in
  HDF5 (`.nwb`) opens in the desktop app and the container.
- **WAV files** (`.wav`): 8-, 16-, 24- or 32-bit PCM, or 32- or 64-bit float, any number of
  channels. Integers are shown scaled to ±1.
- **Raw recordings**: a `.bin` with its `.meta`, dspk's JSON layout:

```json
{ "channels": 32, "sample_rate_hz": 30000.0, "format": "int16", "order": "time_major",
  "gain": 0.195, "unit": "microvolt" }
```

Files are read in pieces, so a recording larger than the browser's memory opens too; its
overview fills in while you look.

## Curate a sorting

**Timeline → + → Sorting…** picks a `.sorting.zarr` folder (or a Phy folder's files) and opens it
in [Curation](/dsp-kitchen/plugins/synapse/learn/curation/), over the recording that is open:
every view, label, merge and split, computed in the page on your CPU's threads. A page has no disk
to write to: **Save** downloads `dspk-curation.json`. Put it into the `.sorting.zarr` folder, where
the desktop app and `dspk` read it.

Not in the viewer yet: SpikeGLX (`.ap.bin` with SpikeGLX's `.meta`), NWB in HDF5, sorting a
recording, and the analytics. Use the [desktop app](/dsp-kitchen/app/) or
[the container](/dsp-kitchen/deploy/image/#the-app-in-your-browser), where the server sorts and
curates.

## What your browser needs

The page checks before it downloads the app (a few MB), and says what is missing:

| | Needed for | Where |
|---|---|---|
| Hardware graphics | drawing (in software the page would crawl) | every recent browser, with graphics acceleration on |
| Threads | drawing on several cores | recent Chrome, Edge, Firefox, Safari (the page turns them on itself, after one reload) |
| WebGPU | a view's processing steps | Chrome and Edge on Windows and macOS; on Linux, Chrome with *Unsafe WebGPU Support* in `chrome://flags` |

Without WebGPU the recording still opens and draws; the processing steps say they need it. The
whitening step is not available in the browser yet.

## The browser, the desktop and a server

| | Recordings | Processing |
|---|---|---|
| This page | files on your computer | your GPU, through the browser |
| [The container](/dsp-kitchen/deploy/image/#the-app-in-your-browser) | the server's | the server's GPU; it sorts, and computes Curation's waveforms and features (with the view's processing), and saves curations there |
| The desktop app | files, servers | your GPU, every runtime |
