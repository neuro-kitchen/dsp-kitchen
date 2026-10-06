# playground

Examples of the `dsp_kitchen` Python SDK, written as `# %%` cells (run them as scripts or open
them as notebooks in VS Code / Jupyter).

```text
playground/
├── base/       01 filtering · 02 math, common reference, PCA on every runtime · 03 pipelines, probes, recordings
├── synapse/    01 detection walkthrough · 02 streaming detection of a whole recording · 03 detection vs ground truth
├── sorters/    00 Kilosort4 + EMUsort on synthetic ground truth · emusort_hdemg (EMUsort on HD-EMG) · kilosort4_universal_templates (Neuropixels)
└── output/     anything the scripts write (git-ignored)
```

## Setup

```bash
uv pip install maturin && maturin develop     # builds dsp_kitchen (default features: wgpu, hub)
uv pip install matplotlib                     # optional, for the plots
python playground/base/01_filtering_methods.py
```

## Data

Test data is local and not in the repository: it lives in `<repository>/data/` (git-ignored), or
in the folder named by `DSP_KITCHEN_DATA`.

| Path under `data/` | Used by |
|---|---|
| `nwb/15-25-33_meps.nwb.zarr` (HD-EMG, 4 × 8 grid, `HDEMG` series) | `base/03`, `synapse/01`, `synapse/02`, `sorters/emusort_hdemg` |
| `kilosort4/ZFM-02370_mini.imec0.ap.short.bin` + `.meta` (Neuropixels 1.0, SpikeGLX) | `sorters/kilosort4_universal_templates` |
| `kilosort4/saved_results/` (Kilosort4's output for that file, Phy folder) | `sorters/kilosort4_universal_templates` |

`base/01`, `base/02`, `synapse/03` and `sorters/00` need no data (synthetic signals, or
`dsp_kitchen.io.SyntheticRecording` with ground truth). `base/03` and `synapse/01` fall back to
synthetic data when the recording is absent.

## What the sorter examples show

`kilosort4.run` and `emusort.run` run the stages implemented so far over a whole recording, in Rust and
on the device (halo windows, bounded memory): preprocessing, EMUsort's channel-delay removal,
universal templates and universal-template detection. Clustering, deconvolution and merging are not
implemented yet; see the book's *Sorters* pages. `sorters/00_sorters_synthetic.py` checks them
without data: channel delays recovered and removed on the device, detection vs ground truth, the
same run on every compiled runtime, export.
