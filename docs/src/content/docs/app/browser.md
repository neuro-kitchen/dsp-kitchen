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

**Open folder…** and pick the folder that holds your recordings (or a `.zarr` folder itself).
The viewer reads:

- **Zarr recordings** (`.zarr`, as `dspk convert` and `dspk run` write them: zstd, Blosc-LZ4,
  gzip or uncompressed, sharded or not). Only the chunks a view needs are read.
- **WAV files** (`.wav`): 16- or 32-bit PCM, or 32- or 64-bit float, any number of channels.
  Integers are shown scaled to ±1. A good first file to try: any recording saved as audio.
- **Raw recordings**: a `.bin` with its `.meta`, dspk's JSON layout:

```json
{ "channels": 32, "sample_rate_hz": 30000.0, "format": "int16", "order": "time_major",
  "gain": 0.195, "unit": "microvolt" }
```

One recording in the folder opens at once; with several, the viewer lists them. **Open files…**
picks a `.wav`, or a `.bin` and its `.meta`, by hand. Files are read in pieces, so a recording
larger than the browser's memory opens too; its overview fills in while you look.

Not in the viewer yet: SpikeGLX (`.ap.bin` with SpikeGLX's `.meta`), NWB, sortings, Curation and
Analytics. They need the electrophysiology formats (dspk-synapse) built for the browser. Use the
[desktop app](/dsp-kitchen/app/) or [the container](/dsp-kitchen/deploy/image/#the-app-in-your-browser).

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
| [The container](/dsp-kitchen/deploy/image/#the-app-in-your-browser) | the server's | the server's GPU |
| The desktop app | files, servers | your GPU, every runtime |
