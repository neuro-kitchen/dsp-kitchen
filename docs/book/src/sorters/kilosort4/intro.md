# Kilosort4: introduction and citation

Kilosort4 sorts spikes from high-density extracellular probes (Neuropixels, polytrodes). It
detects spikes with **universal templates** at many virtual positions on the probe, clusters
them in local feature spaces, learns multi-channel templates, and resolves overlapping spikes by
template deconvolution, with drift correction.

In dsp-kitchen it lives in `dsp_synapse_ml::sorters::kilosort4`, **written from the paper and
the published defaults** (the upstream code is GPL-3.0 and is not ported).

## What is implemented

| Stage | Status |
|---|---|
| Whole-recording run (`Kilosort4::run`, Python `kilosort4.run`): halo windows streamed on the device | implemented |
| Preprocessing fitted on the recording (common reference, 300 Hz high-pass, local whitening) | implemented (device) |
| Universal templates learned from the recording (`wPCA`, `wTEMP`) | implemented (device) |
| Predefined universal templates (`wTEMP.npz`) | implemented |
| Universal-template spike detection, `wPCA` features, position | implemented (device) |
| Graph-based clustering of the detected spikes (bipartite k-NN graph, merging tree, bimodality splits) | implemented (device) |
| Learned templates: the units' templates aligned and near-duplicates merged | implemented (device) |
| Learned-template matching with matching pursuit, background-subtracted features | implemented (device) |
| Clustering of the matched spikes into the final units, with the refractory (CCG) criterion | implemented (device) |
| Export as a `SortingOutput` (one unit per cluster, with its waveform template) | implemented |
| Global merges (waveform similarity + refractory CCG), duplicate-spike removal, good / mua labels | implemented |
| Drift correction | not yet |

## Universal templates

Two arrays drive detection:

- `wPCA` (`n_pcs × nt`): a single-channel temporal basis; features are projections onto it.
- `wTEMP` (`n_templates × nt`): single-channel waveform shapes ("universal templates").

By default (`templates_from_data = true`) **both are learned from each recording**; Kilosort4 also
publishes a predefined set (6 × 61 each) used when `templates_from_data = false`. There are no
other "pretrained weights".

## Citation

> Pachitariu M., Sridhar S., Pennington J., Stringer C. *Spike sorting with Kilosort4.* Nature
> Methods (2024). [doi:10.1038/s41592-024-02232-7](https://doi.org/10.1038/s41592-024-02232-7)

```bibtex
@article{pachitariu2024kilosort4,
  title   = {Spike sorting with {Kilosort4}},
  author  = {Pachitariu, Marius and Sridhar, Shashwat and Pennington, Jacob and Stringer, Carsen},
  journal = {Nature Methods},
  year    = {2024},
  doi     = {10.1038/s41592-024-02232-7}
}
```

| | |
|---|---|
| Code | [MouseLand/Kilosort](https://github.com/MouseLand/Kilosort), GPL-3.0; defaults followed from v4.1.3 |
| Predefined templates | `wTEMP.npz` — [osf.io/download/6807fb5958b763aae139aa60](https://osf.io/download/6807fb5958b763aae139aa60/), 3432 bytes, SHA-256 `cae1c96f8f4150be0a39627515750b70c4bc3548177cf487ae3c013f1ca6abd8` (catalog id `kilosort4/wtemp-v1`) |
| In code | `dsp_synapse_ml::sorters::kilosort4::kilosort4_provenance()` |
