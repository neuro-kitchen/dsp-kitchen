---
title: "Tridesclous 2"
description: "Tridesclous 2 is SpikeInterface's rewrite of Tridesclous (Samuel Garcia and Christophe Pouzat) from SpikeInterface's sorting components."
sidebar:
  order: 13
---

Tridesclous 2 is SpikeInterface's rewrite of Tridesclous (Samuel Garcia and Christophe Pouzat) from
SpikeInterface's sorting components. It detects locally exclusive peaks, clusters a uniform sample
of them by **iterative isosplit** splits on local SVD features, cleans and merges the templates, and
then **peels** the recording: spikes are found, assigned to the closest template, fitted and
subtracted, level after level, with a last matched-filter level for the small spikes left.

In dsp-kitchen it lives in `the sorter crates::tridesclous2`, **ported from the source**
(SpikeInterface `sorters/internal/tridesclous2.py`, `clustering/iterative_isosplit.py`,
`clustering/isosplit_isocut.py`, `matching/tdc_peeler.py`, MIT; sorter version 2026.01, main
`f08c987`). Its isosplit is SpikeInterface's own (not isosplit6): ours reproduces it exactly on the
reference cases (`playground/benchmarks/isosplit_si_reference.py`).

## What is implemented

| Stage | Status |
|---|---|
| Preprocessing: Bessel band-pass 150–6000 Hz, common median reference, local whitening | implemented |
| Detection (`locally_exclusive`), uniform selection | implemented |
| Local SVD features, iterative isosplit splits (SpikeInterface's isosplit) | implemented |
| Templates, cleaning, merging, small clusters; mean templates for the peeler | implemented |
| Tridesclous peeler (host, windows in parallel) | implemented |
| Final cleaning (`auto_merge_units`) | implemented |
| Motion correction | not yet (off by default upstream too) |


## Citation

Tridesclous has no dedicated paper; cite SpikeInterface and the code:

> Buccino A. P., Hurwitz C. L., Garcia S., Magland J., Siegle J. H., Hurwitz R., Hennig M. H.
> *SpikeInterface, a unified framework for spike sorting.* eLife 9:e61834 (2020).
> [doi:10.7554/eLife.61834](https://doi.org/10.7554/eLife.61834)

> Magland J. F., Barnett A. H. *Unimodal clustering using isotonic regression: ISO-SPLIT.*
> arXiv:1508.04841 (2015). [arxiv.org/abs/1508.04841](https://arxiv.org/abs/1508.04841)

| | |
|---|---|
| Code | [SpikeInterface](https://github.com/SpikeInterface/spikeinterface) (MIT); [tridesclous](https://github.com/tridesclous/tridesclous) (original) |
| Version followed | sorter 2026.01 (main `f08c987`, 2026-10-09) |


## Run it

```sh
dspk sort tridesclous2 recording.bin -o sorted --format phy
```

Every setting and its default: [Tridesclous 2 settings](/dsp-kitchen/plugins/synapse/reference/settings/tridesclous2/). How it works: [pipeline](/dsp-kitchen/plugins/synapse/reference/sorters/tridesclous2-pipeline/). What to change: [tuning](/dsp-kitchen/plugins/synapse/reference/sorters/tridesclous2-tuning/).
