# playground — review (2026-10-06)

Read-only review; nothing changed. 12 scripts (3.4 k lines, `# %%` cells usable as notebooks),
two benchmark reports, seven figures. `playground/data/` is git-ignored and absent here, so every
script that reads a recording expects files that are not in the repository.

**D** = a decision needed from you.

## Status of each script

The new SDK API is in `dsp_kitchen_py/README.md`.

| Script | What it shows | State | Proposal |
|---|---|---|---|
| `base/01_filtering_methods.py` | Every filter, causal vs zero phase, settling, template subtraction | Stale calls: `median_filter_9p`, `notch_filter(freq=…)`, `fs` positional; otherwise sound | **Update** |
| `base/02_spatial_math_and_pca.py` | Scale / baseline / clamp, CAR, PCA "CPU vs GPU" | `PCA.transform(use_gpu=…)` removed; `Scale` without `alpha` | **Update** |
| `base/03_composable_gpu_pipeline.py` | Probe presets, `Pipeline`, `DspSession`, an NWB series | `DspSession`, `*_layout`, `get_local_path`, `open_nwb_zarr`, `list_nwb_series` removed | **Update** (`ProbeLayout.*`, `io.Recording`, `io.list_sources`) |
| `sorters/01_simple_spike_sorting.py` | Filter → detect → dedup → snippets → templates, SNR, ISI on HD-EMG | `custom_layout`, `open_nwb_zarr`; there is no clustering, so it is detection | **Update**, rename *detection* |
| `sorters/02_streaming_out_of_core_sorting.py` | `sort_recording` on a whole NWB recording | `sort_recording` → `detect_recording`; it is detection | **Update**, rename *streaming detection* |
| `sorters/03_ml_advanced_spike_sorting.py` | Denoisers, VAE / contrastive embedders, quality classifier, ONNX bridge, on **random weights** | None of these classes exist in the SDK; results on random weights mean nothing | **Park** |
| `sorters/04_ground_truth_evaluation.py` | Detection scored against MEArec ground truth (or its own synthetic fallback) | `get_local_path`, `custom_layout`; the method is sound | **Update**; generate the recording and its ground truth with dsp-io `SyntheticRecording` (Rust exposes `spike_times(unit, range)`; it needs a Python binding) instead of the script's own generator |
| `sorters/05_kilosort4_model_hub.py` | "Kilosort4" via `Kilosort4Detector` / `Kilosort4BasisEmbedder` | Built on the **parked** design (fixed `wTEMP` as a matched filter + threshold, which is not Kilosort4) | **Park** |
| `sorters/06_hdemg_nwb_zarr_kilosort4.py` | The same "Kilosort4" on HD-EMG | Same | **Park** |
| `sorters/07_kilosort4_nwb_to_phy.py` | "Kilosort4" → GMM → Phy (CLI script) | Same; its docstring claims a Kilosort4 run | **Park** |
| `sorters/kilosort4_test.py` | Parked "Kilosort4" detector vs Kilosort4's saved results on the IBL `ZFM-02370` sample | Same; but comparing against Kilosort4's own output is the right validation idea | **Park**, reuse the idea (below) |
| `sorters/08_emusort_nwb_to_phy.py` | "EMUsort": 150-sample `wTEMP_EMG`, 12-PC basis, latency aligner, CAR | The **invented** EMUsort design (EMUsort uses 61-sample learned templates, 9 PCs, no CAR, channel-delay removal) | **Park** |

## Other files

- **`figures/*.png` (4.2 MB):** produced by `playground/spike_sorting_evaluation.py`, which no
  longer exists; they show claims such as "99.1% sensitivity" that cannot be reproduced now.
  *Proposal:* park them; scripts that plot write into a git-ignored `playground/output/`. **D**
- **`benchmarks/*.json`:** two reports in the old suite format (a `sorter` field; the suite now
  writes `streaming_detection`) from `playground/benchmarks`, the CLI's old default. *Proposal:*
  park them as a historical baseline; new reports go wherever `--report-dir` points.
- **`.gitignore`** ignores `playground/data/` and `playground/sorters/phy_*`.

## Cross-cutting

- **PG1: data.** Every recording-based script needs files that are not in the repository and not
  described anywhere. *Proposal:* a `playground/README.md` listing each dataset, where to download
  it (with its license), and the path each script expects. Known so far:
  - `nwb/15-25-33_meps.nwb.zarr`: a lab HD-EMG recording, not public? **D**
  - `kilosort4/ZFM-02370_mini.imec0.ap.short.bin`: the IBL sample used by Kilosort's tutorial.
  - `mearec_32ch_10s.*`: a MEArec simulation, regenerable.
- **PG2: units.** Scripts label values µV (`peak_uv`, `sigmas_uv`, "µV" axes) and use `0.195`-style
  constants. *Proposal:* read units from `Recording.units`.
- **PG3: paths.** `get_local_path()` / `resolve_data_path()` are gone from the SDK. *Proposal:* each
  script takes its data path as an argument or a constant at the top, relative to `playground/`.
- **PG4: names.** "sorting" is used for detection-only pipelines. *Proposal:* call them detection,
  keeping "sorter" for the sorters in `dsp_kitchen.synapse.ml`.
- **PG5: new examples for the real sorters** (replacing 05–08 and `kilosort4_test`):
  - `sorters/kilosort4_universal_templates.py`: on the IBL sample, learn `wPCA` / `wTEMP` from
    the data, load the predefined `wTEMP.npz` from the hub, compare them, and run universal-template
    detection. Then compare the detected spike times with Kilosort4's saved results (the useful
    idea of `kilosort4_test.py`).
  - `sorters/emusort_channel_delays.py`: channel-delay estimation and removal on HD-EMG, and
    EMUsort's templates (several thresholds, HDBSCAN outlier removal) vs Kilosort4's.
  - Both print the sorter's `provenance().citation()`.

## Proposed layout

```text
playground/
├── README.md            datasets (source, license, expected path), how to run (`maturin develop`)
├── base/                01 filtering · 02 spatial, math, PCA · 03 pipelines, probes, recordings
├── synapse/             01 detection walkthrough · 02 streaming detection · 03 ground-truth evaluation
├── sorters/             kilosort4_universal_templates · emusort_channel_delays
└── output/              (git-ignored) figures and reports the scripts write
```

The parked scripts, figures and reports would go to `refactoring/playground/` with a README.
