# EMUsort: introduction and citation

EMUsort sorts **motor-unit action potentials** (MUAPs) from high-density intramuscular arrays
such as Myomatrix. It is a fork of [Kilosort4](../kilosort4/intro.md), adapted to what makes
muscle recordings hard: the same unit reaches channels at different times (conduction along the
fibres), waveforms are complex and polyphasic, and units overlap heavily.

In dsp-kitchen it lives in `dsp_synapse_ml::sorters::emusort`, **written from the paper and the
published defaults** (the upstream code is GPL-3.0 and is not ported). Because it is a fork, it
reuses every Kilosort4 stage of `sorters::kilosort4` with its own settings and adds two stages.

## What EMUsort changes in Kilosort4

| Change | Why (paper) |
|---|---|
| **Channel-delay removal** before detection | a MUAP reaches channels up to a few ms apart |
| 9 temporal PCs and 9 universal templates (Kilosort4: 6 and 6) | more complex waveforms |
| Several clip thresholds `[6, 9, 12, 15]` (Kilosort4: one, 6) | later-recruited, larger units were missed by one threshold |
| **HDBSCAN outlier removal** before the templates are clustered | movement artifacts contaminate the templates |
| No common average reference | a MUAP spans many channels; subtracting the average removes signal |
| Template-learning stride `nskip` 2 (Kilosort4: 25) | more MUAP waveforms for the templates |

## What is implemented

| Stage | Status |
|---|---|
| Whole-recording run (`Emusort::run`, Python `emusort.run`), Kilosort4's runner with EMUsort's plan | implemented |
| Channel-delay estimation and removal | implemented (device) |
| Universal templates learned with HDBSCAN outlier removal and several thresholds | implemented (device) |
| Universal-template detection and features | implemented (device, shared with Kilosort4) |
| Everything after detection | as in Kilosort4: not yet |

EMUsort learns its universal templates from every recording; it publishes no weight or template
files.

## Citation

> O'Connell S., Michaels J. A., Wang R., Mamidipaka S., Venkatesh M., Aresh N., Pachitariu M.,
> Pruszynski J. A., Sober S. J., Pandarinath C. *High performance sorting of motor unit action
> potentials with EMUsort.* openRxiv (bioRxiv) (2026), CC-BY 4.0.
> [doi:10.64898/2026.01.06.697952](https://doi.org/10.64898/2026.01.06.697952)

```bibtex
@article{oconnell2026emusort,
  title   = {High performance sorting of motor unit action potentials with {EMUsort}},
  author  = {O'Connell, Sean and Michaels, Jonathan A and Wang, Runming and Mamidipaka, Sahit and
             Venkatesh, Manikandan and Aresh, Nevin and Pachitariu, Marius and Pruszynski, J Andrew
             and Sober, Samuel J and Pandarinath, Chethan},
  journal = {openRxiv},
  year    = {2026},
  doi     = {10.64898/2026.01.06.697952}
}
```

| | |
|---|---|
| Code | [snel-repo/EMUsort](https://github.com/snel-repo/EMUsort) (GPL-3.0, commit `a06bb60`) and its Kilosort4 fork [snel-repo/Kilosort4](https://github.com/snel-repo/Kilosort4) (GPL-3.0, compared against Kilosort4 v4.0.18) |
| Arrays | Myomatrix: Chung et al., *Myomatrix arrays for high-definition muscle recording*, eLife (2023), [doi:10.7554/eLife.88551](https://doi.org/10.7554/eLife.88551) |
| In code | `dsp_synapse_ml::sorters::emusort::emusort_provenance()` |
