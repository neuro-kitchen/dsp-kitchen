# dsp-kitchen Python API

Every class and function of the `dsp_kitchen` package: what it does, its arguments with their
**types, defaults and units**, what it returns, and short examples. Generated from the package's
stubs and docstrings, so it matches the installed version.

**New here?** The [guide](../../guide/) explains the concepts (filtering, pipelines, probes, the
sorters and how to tune them); this reference is for looking up an exact signature. Rust users:
the [Rust API](../rust/dsp_base/index.html); all crates are listed on the
[documentation home](../../index.html).

| Module | What it holds |
|---|---|
| [`dsp_kitchen.synapse.ml.kilosort4`](sorters/kilosort4.md), [`…emusort`](sorters/emusort.md) | spike sorters: configuration, `run`, results |
| [`dsp_kitchen.io`](io.md) | recordings (any supported format), memory-mapped and synthetic recordings |
| [`dsp_kitchen.pipeline`](pipeline.md) | stages chained on the device |
| [`dsp_kitchen.filter`](filter/iir.md) | IIR (Butterworth, Chebyshev, notch), FIR, non-linear, template subtraction |
| [`dsp_kitchen.spatial`](spatial.md) | common average reference, surface Laplacian, whitening |
| [`dsp_kitchen.math`](math.md), [`dsp_kitchen.linalg`](linalg.md) | pointwise stages; PCA, PPCA, FastICA |
| [`dsp_kitchen.synapse`](synapse.md) | probes, detection, sorting outputs, metrics, comparison |
| [`dsp_kitchen.runtime`](runtime.md) | choosing the compute device; progress bars |

**Conventions.** Multi-channel data is `[channels, samples]`, `float32`. Sampling rates are in Hz,
distances in µm, sorter thresholds in whitened σ. Lengths given in **samples** depend on the
sampling rate.
