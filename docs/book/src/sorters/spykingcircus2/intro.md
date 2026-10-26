# SpyKING CIRCUS 2: introduction and citation

SpyKING CIRCUS 2 is SpikeInterface's rewrite of SpyKING CIRCUS for dense probes. It detects
spikes by **matched filtering** with a waveform prototype learned from the recording, clusters a
uniform sample of them with **iterative HDBSCAN** on local SVD features, cleans and merges the
cluster templates, and then finds every spike of the recording with **circus-omp**, an orthogonal
matching pursuit over the templates, which resolves overlapping spikes.

In dsp-kitchen it lives in `dsp_synapse_ml::sorters::spykingcircus2`, **ported from the source**
(SpikeInterface `sorters/internal/spyking_circus2.py` and `sortingcomponents`, MIT; sorter version
2025.12, main `f08c987`). The building blocks shared with Tridesclous 2 are in
`dsp_synapse_ml::sorters::components`.

## What is implemented

| Stage | Status |
|---|---|
| Preprocessing: Bessel band-pass 150–7000 Hz, common median reference, local whitening | implemented |
| Noise levels, detection prototype, matched-filtering detection (device) | implemented |
| Uniform selection, local SVD features, iterative HDBSCAN splits | implemented |
| Templates from the features, cleaning, merging by template similarity, small clusters | implemented |
| circus-omp template matching (device products, host pursuit) | implemented |
| Export as a `SortingOutput` (`SorterResult`: labels, composite score, Phy / Zarr) | implemented |
| Final cleaning: `auto_merge_units` (cross-contamination presets) | implemented |
| Motion correction (upstream's default on dense probes) | planned (drift correction, S10) |

```python
from dsp_kitchen.synapse.ml import spykingcircus2

config = spykingcircus2.Config()            # every setting, SpyKING CIRCUS 2's defaults
result = spykingcircus2.run(recording, probe, config)
sorting = result.to_sorting_output(probe)
```

## Citation

> Yger P., Spampinato G. L. B., Esposito E., Lefebvre B., Deny S., Gardella C., Stimberg M.,
> Jetter F., Zeck G., Picaud S., Duebel J., Marre O. *A spike sorting toolbox for up to thousands
> of electrodes validated with ground truth recordings in vitro and in vivo.* eLife 7:e34518
> (2018). [doi:10.7554/eLife.34518](https://doi.org/10.7554/eLife.34518)

> Buccino A. P., Hurwitz C. L., Garcia S., Magland J., Siegle J. H., Hurwitz R., Hennig M. H.
> *SpikeInterface, a unified framework for spike sorting.* eLife 9:e61834 (2020).
> [doi:10.7554/eLife.61834](https://doi.org/10.7554/eLife.61834)

| | |
|---|---|
| Code | [SpikeInterface](https://github.com/SpikeInterface/spikeinterface) (MIT) |
| Version followed | sorter 2025.12 (main `f08c987`, 2026-10-09) |
