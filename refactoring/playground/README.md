# playground — changes (2026-10-06)

Review: `REVIEW.md`. User decisions:
- implement the new layout;
- the HD-EMG data is available, and its sorter example uses **EMUsort**;
- **delete** the figures (they must be regenerated);
- test data stays local and git-ignored, and lives **outside** `playground/`, in
  `<repository>/data/` or the folder in `DSP_KITCHEN_DATA`.

## Parked

| Parked | Why |
|---|---|
| `sorters/03_ml_advanced_spike_sorting.py` | ML models and an ONNX bridge on random weights; the classes do not exist in the SDK. |
| `sorters/05_kilosort4_model_hub.py`, `06_hdemg_nwb_zarr_kilosort4.py`, `07_kilosort4_nwb_to_phy.py`, `kilosort4_test.py` | "Kilosort4" through the parked fixed-template detector, which is not Kilosort4. `kilosort4_test`'s comparison with Kilosort4's saved results is reused in `sorters/kilosort4_universal_templates.py`. |
| `sorters/08_emusort_nwb_to_phy.py` | The invented EMUsort design. |
| `benchmarks/*.json` | Old suite format (`sorter` field); kept as a historical baseline. |

**Deleted:** `figures/*.png`, made by a removed script and showing claims that cannot be reproduced.

## New layout

| Now | Was | Change |
|---|---|---|
| `base/01_filtering_methods.py` | same | New call forms (`fs` keyword, positional cutoffs, `median_filter(x, 9)`); Chebyshev I added. |
| `base/02_spatial_math_and_pca.py` | same | `Scale(STEP_UV)`, `SubtractBaseline(…)`, `Clamp(lo, hi)` (named constants); PCA fitted and projected on **every compiled-in runtime**, compared (replaces `use_gpu`). |
| `base/03_pipelines_probes_recordings.py` | `03_composable_gpu_pipeline.py` | `ProbeLayout` presets; `io.list_sources` + `io.Recording` (units from the recording); `DspSession` section removed. |
| `synapse/01_detection.py` | `sorters/01_simple_spike_sorting.py` | Detection (no clustering); the grid comes from `ProbeLayout.hdemg_grid` with a named pitch; per-channel σ passed to detection; settings are named constants; units from the recording. |
| `synapse/02_streaming_detection.py` | `sorters/02_streaming_out_of_core_sorting.py` | `syn.detect_recording` with the library's defaults; lazy `slice_time` window. |
| `synapse/03_ground_truth_evaluation.py` | `sorters/04_ground_truth_evaluation.py` | Ground truth from `io.SyntheticRecording` (new binding: dsp-io's procedural recording with `spike_times`), scored with `syn.compare_spike_trains` (per unit and overall); the hand-written generator and matcher are gone. |
| `sorters/emusort_hdemg.py` | new | EMUsort on the HD-EMG recording: preprocessing without common reference, channel-delay removal, universal templates learned at 6–15 σ with HDBSCAN outlier removal, universal-template detection, citation. |
| `sorters/kilosort4_universal_templates.py` | new | Kilosort4 on the IBL Neuropixels sample: probe from the SpikeGLX metadata, learned vs predefined (hub) templates by cosine similarity, detection, and recall / precision against Kilosort4's saved results. |
| `sorters/_frontend.py` | new | Shared Kilosort4-style preprocessing (high-pass, optional common reference, local whitening) and padded batches; `DATA_DIR`. |
| `README.md` | new | Layout, setup, the expected data files under `data/`. |

**Supporting changes:**
- **Bindings:** `dsp_kitchen.io.SyntheticRecording` (`buffer/synthetic.rs`), with `unit_count`,
  `spike_times(unit, start, end)` and `recording()`.
- **`.gitignore`:** `data/` (was `playground/data/`) and `playground/output/`.

**Not run:** the extension is not built here. Every script parses, and every `dsp_kitchen` import
was checked against the SDK's exports.

## Open

- `_frontend.WHITENING_EPSILON = 1e-6` is taken as Kilosort4's `whitening_from_covariance` value;
  verify it against upstream.
- Batch spikes in the padding are dropped in the examples (as Kilosort4 does). `detect_universal`
  could do this itself (an `emit` range, like dsp-synapse's device detection).
- dsp-io `SyntheticRecording` has unnamed literals (waveform shape, rate and amplitude ranges);
  name them in the dsp-io pass.
