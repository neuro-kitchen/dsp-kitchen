---
title: Pyramids
description: "How hours of recording draw fast at any zoom."
sidebar:
  order: 4
---

To draw an hour of a 384-channel recording you can't read every sample per frame. A *min/max pyramid*
stores, for each block of samples, the minimum and the maximum, at several block sizes.

A plain average would hide a spike. A min/max pair keeps every peak visible at any zoom.

To draw a view, dsp-kitchen uses the coarsest level with at most one bucket per pixel column, with
column edges aligned to the buckets. Every sample falls into exactly one column, so no peak is lost.
When you zoom in past the finest level, it reads the raw samples.

## Where a pyramid lives

| Recording size | Pyramid | Finest bucket |
|---|---|---|
| Under 64 MiB | In memory, for this session | 256 samples |
| 64 MiB or more | A file, reused between runs | 64 samples |

The file sits in a `cache/` folder next to the recording, at about 1/16 of the recording's size as
float32. So the folder must be writable the first time.

Pyramids are filled page by page, nearest to where you are looking first, so you can start working
while the rest is still being built.

Servers use the same pyramids: a remote view asks the server for the envelope, not the samples.
