# Sorters

Spike sorters from the literature, reimplemented in Rust from their papers and published defaults
(`dsp_synapse_ml::sorters`). Each records where it comes from — paper and DOI, code and license,
and any file it downloads — as a `Provenance` you can print or cite.

| Sorter | Domain | Pages |
|---|---|---|
| **Kilosort4** | high-density extracellular probes (Neuropixels, polytrodes) | [Introduction](kilosort4/intro.md) · [Pipeline](kilosort4/pipeline.md) · [Parameters](kilosort4/parameters.md) |
| **EMUsort** | high-density intramuscular arrays (Myomatrix), motor units — a Kilosort4 fork | [Introduction](emusort/intro.md) · [Pipeline](emusort/pipeline.md) · [Parameters](emusort/parameters.md) |

Each sorter has three pages: what it is and how to cite it, its stages (marking what is
implemented), and its parameters with their upstream names and defaults.

**Status.** Both run over whole recordings, from the raw file to detected spikes: preprocessing
fitted on the recording, universal templates learned from it, and universal-template detection,
streamed in halo windows on the device (one shared runner, see the
[Kilosort4 pipeline](kilosort4/pipeline.md)). Units are one
per universal template; drift correction, clustering, learned-template deconvolution and merging
are not implemented yet.

Sorters are written from the papers, not ported: the upstream implementations are GPL-3.0 and
this workspace is MIT / Apache-2.0. Until a sorter reproduces its paper's results on the paper's
data, it is a reimplementation in progress, not a substitute for the original.
