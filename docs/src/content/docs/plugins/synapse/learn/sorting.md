---
title: Sorting
description: "Sort a recording from the command line, on your GPU or a server."
sidebar:
  order: 3
---

The sorters are one separate program, `dspk-cli-sorter`, which `dspk sort` runs. It's separate because
Kilosort4 and EMUsort are GPL-3.0, and nothing MIT or Apache-licensed links them.

```sh
dspk sorters install                    # this version's build, SHA-256 checked
dspk sort kilosort4 recording.bin -o sorted --format phy
dspk sort kilosort4 --help              # every setting and its default
```

Start with a short stretch, using `--start-sec` and `--duration-sec`, before committing to a whole
recording. `--runtime cuda` picks the runtime, and adding `--show-config` to a command prints every
setting with its value.

The probe comes from the recording when it carries one (SpikeGLX). Otherwise pass `--probe` or
`--probe-file`.

:::note
From Python, sorts run on a server through [`dspk.remote`](/dsp-kitchen/deploy/remote/). The sorters
aren't bound in the Python package, so that the wheel stays MIT or Apache-2.0.
:::

## What you get

A sorting in the format you asked for. Every sorter's units are labelled good or mua by their
auto-correlogram, and scored with EMUsort's composite score.

Open it in the [app](/dsp-kitchen/plugins/synapse/learn/sorting-in-the-app/), or load it:

```python
sorting = dspk.synapse.io.Sorting.load("sorted")
sorting.units()                # a row per unit
sorting.spike_train(3)
```

`dspk-cli-sorter licenses` prints each sorter's licence and source.

## Where it's installed

The user data folder: `~/.local/share/dspk/sorters/<version>/` on Linux,
`~/Library/Application Support/dspk/…` on macOS and `%LOCALAPPDATA%\dspk\…` on Windows.
`DSPK_DATA_DIR` moves it, and `DSPK_CLI_SORTER_BIN` names another build.

Each sorter has a page on [what it does](/dsp-kitchen/plugins/synapse/reference/sorters/kilosort4/),
how it works and how to tune it.
