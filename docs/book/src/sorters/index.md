# Sorters

Spike sorters from the literature, in Rust on the device (`dsp_synapse_ml::sorters`): written from
their papers and published defaults where the upstream code is GPL-3.0 (Kilosort4, EMUsort), ported
from the source where it is permissively licensed (MountainSort 5, Apache-2.0; SpyKING CIRCUS 2 and
Tridesclous 2, MIT). Each records where it comes from — paper and DOI, code and license,
and any file it downloads — as a `Provenance` you can print or cite.

| Sorter | Domain | Pages |
|---|---|---|
| **Kilosort4** | high-density extracellular probes (Neuropixels, polytrodes) | [Introduction](kilosort4/intro.md) · [Pipeline](kilosort4/pipeline.md) · [Parameters](kilosort4/parameters.md) · [Tuning](kilosort4/tuning.md) |
| **EMUsort** | high-density intramuscular arrays (Myomatrix), motor units — a Kilosort4 fork | [Introduction](emusort/intro.md) · [Pipeline](emusort/pipeline.md) · [Parameters](emusort/parameters.md) · [Tuning](emusort/tuning.md) |
| **MountainSort 5** | any geometry, tetrodes to dense probes; isosplit6 clustering, no cluster count | [Introduction](mountainsort5/intro.md) · [Pipeline](mountainsort5/pipeline.md) · [Parameters](mountainsort5/parameters.md) · [Tuning](mountainsort5/tuning.md) |
| **SpyKING CIRCUS 2** | dense probes; matched-filtering detection, iterative HDBSCAN, orthogonal matching pursuit | [Introduction](spykingcircus2/intro.md) · [Pipeline](spykingcircus2/pipeline.md) · [Parameters](spykingcircus2/parameters.md) · [Tuning](spykingcircus2/tuning.md) |
| **Tridesclous 2** | dense probes; iterative isosplit, template peeling | [Introduction](tridesclous2/intro.md) · [Pipeline](tridesclous2/pipeline.md) · [Parameters](tridesclous2/parameters.md) · [Tuning](tridesclous2/tuning.md) |

Each sorter has four pages: what it is and how to cite it, its stages (marking what is
implemented), its parameters with their upstream names and defaults, and which parameters to tune
to improve a sort (and how to check that it improved).

**Status.** All run over whole recordings, from the raw file to sorted units, on the device.
Kilosort4 and EMUsort share one runner (see the [Kilosort4 pipeline](kilosort4/pipeline.md)):

1. preprocessing fitted on the recording (filters, common reference, local whitening; EMUsort adds
   channel-delay removal);
2. universal templates learned from it, and universal-template detection;
3. a first graph-based clustering of those spikes into units;
4. learned templates (the units' templates, aligned and deduplicated);
5. learned-template matching with matching pursuit, the clustering of the matched spikes into the
   final units (with the refractory cross-correlogram criterion), global merges and duplicate-spike
   removal.

MountainSort 5 has its own runner: detection, isosplit6 clustering of a training stretch,
per-channel classifiers over the whole recording (see its [pipeline](mountainsort5/pipeline.md)).
SpyKING CIRCUS 2 too: matched-filtering detection, iterative HDBSCAN, circus-omp matching (see its
[pipeline](spykingcircus2/pipeline.md)); and Tridesclous 2: iterative isosplit, template peeling (see
its [pipeline](tridesclous2/pipeline.md)). The last two share SpikeInterface's sorting components
(`dsp_synapse_ml::sorters::components`).
Every sorter's units are labelled (good / mua by their auto-correlogram) and scored (EMUsort's
composite score) the same way. Not yet: drift correction. Results and processing times on the
test data are in [Benchmarks](benchmarks.md).

Kilosort4 and EMUsort are written from the papers, not ported: their upstream implementations are
GPL-3.0 and this workspace is MIT / Apache-2.0. Until a sorter reproduces its paper's results on
the paper's data, it is a reimplementation in progress, not a substitute for the original.
