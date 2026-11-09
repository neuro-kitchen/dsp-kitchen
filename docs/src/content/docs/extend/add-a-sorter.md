---
title: Add a sorter
description: "Implement a published sorter so every result stays traceable to its authors."
sidebar:
  order: 4
---

A sorter here is a published algorithm. The rules keep every result traceable to its authors.

## Before writing code

1. **Find the paper and resolve its DOI.** Note the venue, year, authors and the paper's licence.
2. **Find the upstream code**, its licence (SPDX), and the release or commit whose defaults you follow.
3. **Hash every downloaded file** (weights, arrays) with SHA-256, and record its size, before listing it.
   Never ship placeholder hashes or invented shapes.

## Licence decides the crate

There is one crate per upstream project, and it follows that project's licence.

| Upstream | Crate |
|---|---|
| Kilosort4, EMUsort (GPL-3.0) | `dspk-sorter-kilosort` |
| MountainSort 5 (Apache-2.0) | `dspk-sorter-ms5` |
| SpikeInterface's sorters (MIT) | `dspk-sorter-si` |

Code ported from one upstream lives in that upstream's crate, even when that means a copy. If you read
upstream's code to match its behaviour, say so in the crate's `NOTICE.md` and don't claim a clean-room
implementation.

## What a sorter provides

- A config with upstream's names in the docs and upstream's defaults in `Default`.
- A provenance record: paper, code, licence and any file it downloads.
- A runner from the raw recording to sorted units, using the shared window streaming.
- Device kernels beside the code that launches them.

## Wiring it in

Add a subcommand to `plugins/cli/dspk-cli-sorter`, exposing settings as flags with their defaults. The app
lists what `dspk-cli-sorter describe` reports.

## Check it

- Two runs on the same input give identical spikes and units.
- Compare with upstream on the same recording, by time and place, and report the numbers in
  [Validation](/dsp-kitchen/plugins/synapse/validation/).
- Say what isn't implemented. Until a sorter reproduces its paper's results on the paper's data, it's a
  reimplementation in progress.
