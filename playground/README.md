# playground

Examples of the `dspk` Python package, written as `# %%` cells (run them as scripts or open them as
notebooks in VS Code / Jupyter). Figures and files go to `playground/output/` (git-ignored).

```text
playground/
├── base/       01 IIR filters vs scipy on every runtime · 02 pipelines on the device and streaming · 03 pipelines as data (JSON / TOML specs)
├── synapse/    detection walkthroughs                    (old `dsp_kitchen` API: waits for the synapse bindings)
├── sorters/    Kilosort4 + EMUsort examples             (old `dsp_kitchen` API: waits for the synapse bindings)
├── benchmarks/ sorter validation and resources          (old `dsp_kitchen` API: waits for the synapse bindings)
└── output/     anything the scripts write (git-ignored)
```

## Setup

From the repository root:

```bash
uv sync                                                             # development environment
cd bindings/python && uv run --project ../.. maturin develop --profile validate && cd ../..
uv run python playground/base/01_filters_vs_scipy.py
```

Use `maturin develop --release` instead for timings (`--profile validate` optimizes the dependencies
but not dspk itself).

## Not in the new bindings yet

The previous `base/` scripts (FIR and template filters, PCA and other linear algebra, spatial
whitening, class-style filters, probe layouts) used parts the rewritten bindings do not cover yet;
they are parked in `.refactoring/playground/base-old/` and in git history. Probe layouts belong to
the synapse bindings.

## Data

Test data is local and not in the repository: it lives in `<repository>/data/` (git-ignored), or
in the folder named by `DSP_KITCHEN_DATA` (scripts on the old API).

| Path under `data/` | Used by |
|---|---|
| `nwb/15-25-33_meps.nwb.zarr` (HD-EMG, 4 × 8 grid, `HDEMG` series) | `base/03`, `synapse/01`, `synapse/02`, `sorters/emusort_hdemg`, `benchmarks/emusort_checks`, `benchmarks/sorter_resources` |
| `kilosort4/ZFM-02370_mini.imec0.ap.short.bin` + `.meta` (Neuropixels 1.0, SpikeGLX) | `sorters/kilosort4_universal_templates`, `benchmarks/kilosort4_*`, `benchmarks/sorter_resources` |
| `kilosort4/saved_results/` (Kilosort4's output for that file, Phy folder) | `sorters/kilosort4_universal_templates`, `benchmarks/kilosort4_*` (through `benchmarks/ks4_reference.py`) |

The `base/` scripts need no data (synthetic signals). `synapse/03` and `sorters/00` use
`dsp_kitchen.io.SyntheticRecording`; `synapse/01` falls back to synthetic data when the recording is
absent.

## What the sorter examples show

`kilosort4.run` and `emusort.run` (with a progress bar per stage and the time left; `tqdm` is
used when installed) run the stages implemented so far over a whole recording, in Rust and
on the device (halo windows, bounded memory): preprocessing, EMUsort's channel-delay removal,
universal templates, detection, clustering, learned-template matching and the clustering of the
matches into units. The refractory criteria, final merges and drift correction are not implemented
yet; see the book's *Sorters* pages and *Benchmarks*. Both scripts end with `inspection.inspect`:
one channel of the run's own preprocessed signal with the units on it (Kilosort4's saved units
alongside ours on the Neuropixels data). `sorters/00_sorters_synthetic.py` checks them
without data: channel delays recovered and removed on the device, detection vs ground truth, the
same run on every compiled runtime, export.
