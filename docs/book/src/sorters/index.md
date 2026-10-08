# Sorters

Spike sorters from the literature, reimplemented in Rust from their papers and published defaults
(`dsp_synapse_ml::sorters`). Each records where it comes from — paper and DOI, code and license,
and any file it downloads — as a `Provenance` you can print or cite.

| Sorter | Domain | Pages |
|---|---|---|
| **Kilosort4** | high-density extracellular probes (Neuropixels, polytrodes) | [Introduction](kilosort4/intro.md) · [Pipeline](kilosort4/pipeline.md) · [Parameters](kilosort4/parameters.md) · [Tuning](kilosort4/tuning.md) |
| **EMUsort** | high-density intramuscular arrays (Myomatrix), motor units — a Kilosort4 fork | [Introduction](emusort/intro.md) · [Pipeline](emusort/pipeline.md) · [Parameters](emusort/parameters.md) · [Tuning](emusort/tuning.md) |

Each sorter has four pages: what it is and how to cite it, its stages (marking what is
implemented), its parameters with their upstream names and defaults, and which parameters to tune
to improve a sort (and how to check that it improved).

**Status.** Both run over whole recordings, from the raw file to sorted units, on the device, in
one shared runner (see the [Kilosort4 pipeline](kilosort4/pipeline.md)):

1. preprocessing fitted on the recording (high-pass, common reference, local whitening; EMUsort
   adds channel-delay removal);
2. universal templates learned from it, and universal-template detection;
3. a first graph-based clustering of those spikes into units;
4. learned templates (the units' templates, aligned and deduplicated);
5. learned-template matching with matching pursuit, and the clustering of the matched spikes into
   the final units.

Not yet: drift correction, the refractory-period (CCG) criteria of the clustering and the final
merges, duplicate-spike removal, EMUsort's unit score. Results and processing times on the test
data are in [Benchmarks](benchmarks.md).

Sorters are written from the papers, not ported: the upstream implementations are GPL-3.0 and
this workspace is MIT / Apache-2.0. Until a sorter reproduces its paper's results on the paper's
data, it is a reimplementation in progress, not a substitute for the original.
