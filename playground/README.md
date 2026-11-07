# playground

A course in `dspk`, from a first call to a sorted recording. Each folder is a chapter, each script
one idea, a little harder than the one before; every script ends with a figure, committed next to
it in `figures/`, so you can see what it shows before running it. Scripts are `# %%` cells: run
them as scripts, or open them as notebooks in VS Code / Jupyter.

Up to sorting, everything runs on a **synthetic recording** whose content is known
(`plating.signals`: LFP, mains hum, a common artefact, spikes, noise), so each result can be
checked, not just looked at. Sorting uses real data, downloaded once by `09_sorting`.

Figures are drawn with [`plating`](../plating/README.md), the repository's small plotting package:
the look lives there, the scripts only say what to plot.

## Setup

From the repository root:

```bash
uv sync                                                             # the environment (installs plating)
cd bindings/python && uv run --project ../.. maturin develop --profile validate && cd ../..
uv run python playground/01_basics/01_runtimes.py
```

Every script runs in seconds on any runtime (`DSPK_RUNTIME=cpu` works everywhere). For timings,
build with `maturin develop --release` (`--profile validate` optimizes the dependencies but not
dspk itself).

## Chapters

| Chapter | Scripts | What you learn |
|---|---|---|
| **01_basics** | `01_runtimes` · `02_synthetic_signal` · `03_files` | runtimes (CUDA, wgpu, CPU) and that they agree; the recording every chapter uses; writing and reading files (int16 with a gain, Zarr) |
| **02_filtering** | `01_lowpass_highpass` · `02_zero_phase_vs_causal` · `03_notch` · `04_fir_design` · `05_median_and_energy` | splitting LFP from spikes; why zero phase (equal to scipy's `sosfiltfilt`); taking the hum out; designing FIR taps; non-linear filters |
| **03_referencing** | `01_car_and_cmr` | removing what all channels share; why the median survives a bad channel |
| **04_resampling** | `01_resample_poly` | lowering the rate for LFP (`resample_poly`, `decimate`, equal to scipy) |
| **05_spatial** | `01_whitening` · `02_laplacian` | decorrelating channels as sorters do; keeping what is local on a grid |
| **06_pipelines** | `01_chain_on_device` · `02_streaming` · `03_specs` · `04_branches` | chaining stages on the device; streaming in pieces equal to the whole; pipelines as JSON / TOML files; one read, several outputs |
| **07_channels** | `01_remap_with_pairs` · `02_maps_in_specs` | channel maps written as pairs; maps in pipeline specs (reorder, drop, name) |
| **08_detection** | `01_threshold_and_scoring` · `02_features_and_clusters` · `03_templates_and_matching` | finding spikes and scoring the finds; snippets, principal components and clusters; templates and matching pursuit |
| **09_sorting** | `00_fetch_kilosort_data` · `01_detect_and_locate` · `02_sorting_quality` | a real Neuropixels recording: every spike located, the drift map; a sorting judged against a reference and unit by unit |
| **10_remote** | README | a GPU server and a laptop: `dspk serve`, `dspk clients`, `dspk remote`, and `dspk.remote` in Python |

Measurements against other tools (time, agreement) are in [`benchmarks/`](../benchmarks/README.md),
not here: the playground teaches, the benchmarks measure.

## Data

Chapters 01–07 need no data. From chapter 09 on, data is local and not in the repository:
`playground/data/` (what `09_sorting/00_fetch_kilosort_data.py` downloads), and the benchmarks'
`<repository>/.data/` (see `benchmarks/README.md`). Both are git-ignored.
