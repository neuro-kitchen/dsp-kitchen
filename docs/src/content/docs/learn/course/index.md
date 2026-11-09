---
title: Course
description: "Worked examples, from a first call to whole pipelines, each ending with its figure."
sidebar:
  label: Overview
  order: 0
---

Each page is one script of the repository's `playground/`, a little harder than the one before.
Every script ends with a figure, shown on its page, so you can see what it does before running it.

The recordings are synthetic and their content is known: slow waves, mains hum, a common artefact,
spikes and noise, each kept separately. So every result can be checked, not just looked at.

| Chapter | What you learn |
|---|---|
| Basics | Runtimes and that they agree; the recording every chapter uses; writing and reading files |
| Filtering | Splitting LFP from spikes; zero phase; removing hum; FIR design; non-linear filters |
| Referencing | Removing what all channels share, and why the median survives a bad channel |
| Resampling | Lowering the rate for LFP |
| Spatial | Decorrelating channels; keeping what is local on a grid |
| Pipelines | Chains on the device; streaming; specs in JSON or TOML; branches |
| Channels | Channel maps as pairs, in code and in specs |

Detection and spike sorting continue in the
[Synapse plugin](/dsp-kitchen/plugins/synapse/).

## Run them yourself

From the repository root:

```sh
uv sync
cd bindings/python && uv run --project ../.. maturin develop --profile validate && cd ../..
uv run python playground/01_basics/01_runtimes.py
```

`DSPK_RUNTIME=cpu` works everywhere. Scripts are `# %%` cells, so they also open as notebooks in
VS Code or Jupyter.
