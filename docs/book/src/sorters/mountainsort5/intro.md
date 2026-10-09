# MountainSort 5: introduction and citation

MountainSort 5 sorts spikes from extracellular recordings of any geometry, from tetrodes to dense
probes. It detects threshold crossings that are the largest in their neighbourhood, reduces the
waveforms with PCA, and clusters them with **isosplit6**, a clustering method with no tunable
parameter: two clusters are merged unless their projection on the line between them shows a dip
(a non-parametric unimodality test built on isotonic regression). Scheme 2, the default, clusters
a training stretch of the recording and then labels every spike by its nearest training waveform.

In dsp-kitchen it lives in `dsp_synapse_ml::sorters::mountainsort5`, **written from the source**
(`flatironinstitute/mountainsort5` `v0.5.9`, commit `3008b7a`, and `magland/isosplit6`, both
Apache-2.0) and compared with Kilosort4's output on the test recording (*Benchmarks*). Defaults follow
SpikeInterface's MountainSort 5 wrapper where the package leaves them to the caller (filtering,
whitening, scheme 2's radii).

## What is implemented

| Stage | Status |
|---|---|
| Preprocessing: band-pass 300–6000 Hz, global whitening | implemented |
| Detection: largest crossing in its channel neighbourhood and time window (device) | implemented |
| Snippets, PCA, isosplit6 subdivision clustering, templates, alignment (scheme 1) | implemented |
| Per-channel classifiers trained on a stretch, every spike classified (scheme 2, device) | implemented |
| Export as a `SortingOutput` (`SorterResult`: labels, composite score, Phy / Zarr) | implemented |
| Scheme 3 (long recordings in blocks, units matched across blocks) | not planned yet |

```python
from dsp_kitchen.synapse.ml import mountainsort5

config = mountainsort5.Config()            # every setting, MountainSort 5's defaults
result = mountainsort5.run(recording, probe, config)
sorting = result.to_sorting_output(probe)  # Phy / Zarr export, metrics
```

## Citation

> Chung J. E., Magland J. F., Barnett A. H., Tolosa V. M., Tooker A. C., Lee K. Y., Shah K. G.,
> Felix S. H., Frank L. M., Greengard L. F. *A fully automated approach to spike sorting.*
> Neuron 95(6), 1381–1394.e6 (2017). [doi:10.1016/j.neuron.2017.08.030](https://doi.org/10.1016/j.neuron.2017.08.030)

> Magland J. F., Barnett A. H. *Unimodal clustering using isotonic regression: ISO-SPLIT.*
> arXiv:1508.04841 (2015). [arxiv.org/abs/1508.04841](https://arxiv.org/abs/1508.04841)

| | |
|---|---|
| Code | [flatironinstitute/mountainsort5](https://github.com/flatironinstitute/mountainsort5) (Apache-2.0), [magland/isosplit6](https://github.com/magland/isosplit6) (Apache-2.0) |
| Version followed | MountainSort 5 `v0.5.9` (commit `3008b7a`) |
