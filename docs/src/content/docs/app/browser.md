---
title: In your browser
description: "The app at /dsp-kitchen/web/: open a recording from your disk, nothing to install."
sidebar:
  order: 1
---

**[Open the app in your browser](/dsp-kitchen/web/)**: the same app as the desktop's, built for
the browser. Nothing is installed and nothing leaves your computer: the files you open are read in
the page, and processing runs on your own graphics card.

## Open a recording

**Open files…** and pick a `.bin` with its `.meta` (both files at once). The `.meta` is dspk's
JSON layout, as `dspk convert` writes it:

```json
{ "channels": 32, "sample_rate_hz": 30000.0, "format": "int16", "order": "time_major",
  "gain": 0.195, "unit": "microvolt" }
```

The file is read in pieces, so a recording larger than the browser's memory opens too; its
overview is built once when it opens. SpikeGLX recordings (`.ap.bin` with SpikeGLX's `.meta`)
open in the [desktop app](/dsp-kitchen/app/) or [the container](/dsp-kitchen/deploy/image/#the-app-in-your-browser)
for now.

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
