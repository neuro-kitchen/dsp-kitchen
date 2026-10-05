# Adding a sorter

A sorter in `dsp_synapse_ml::sorters` is a published algorithm written here from its paper. The
rules keep every result traceable to its authors.

## Before writing code

1. **Find the paper and resolve its DOI** (`https://doi.org/<doi>`); note the venue, year, authors
   and the paper's license.
2. **Find the upstream code**, its license (SPDX) and the release or commit whose defaults you
   follow. If the license is incompatible with MIT / Apache-2.0 (e.g. GPL), write from the paper
   and use the code only to read facts (parameter names, defaults, the order of steps).
3. **Every downloaded file** (weights, arrays) must be downloaded once, hashed (SHA-256) and sized
   before it is listed. Never ship placeholder hashes or invented shapes.

## Layout

```text
sorters/<name>/
├── mod.rs        <Name>Config (upstream names in the docs, defaults in Default), <name>_provenance(),
│                 <Name> implementing Attributed
└── <stage>.rs    one file per stage; generic DSP goes to dsp-base, generic spike work to dsp-synapse
```

- A fork reuses the sorter it forks (EMUsort reuses `sorters::kilosort4`) and adds only what
  differs; do not copy code between sorters.
- GPU stages take a `ComputeClient<R>`; download only results.
- Name constants and say what they are; follow the upstream defaults and name them as upstream
  does in the field docs.

## Catalog (only for downloaded artifacts)

Add an entry to `catalog/models.json` with the URI, the measured SHA-256 and size, the arrays or
tensor ports, and the same `provenance` record. The catalog test rejects entries without a
64-hex hash, a size, or a DOI.

## Documentation

Add `docs/book/src/sorters/<name>/{intro, pipeline, parameters}.md` (what it is and how to cite
it; its stages, marking what is implemented; its parameters with upstream names and defaults),
list it in `sorters/index.md`, `SUMMARY.md` and [Citations](../reference/citations.md).

## Validation

Compare against the upstream implementation on the paper's released data before calling the
sorter by its name; until then, document it as a reimplementation in progress.
