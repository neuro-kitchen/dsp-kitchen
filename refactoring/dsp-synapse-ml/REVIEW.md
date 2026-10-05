# dsp-synapse-ml review notes (working file)

Started 2026-10-05 on `versions/v0.15`. Read-only review; no code changed. Findings tagged by area.

## Inventory (~4.9k lines)
| Area | Files | What |
|---|---|---|
| `hub/` | cache, catalog, downloader, manifest, providers, pytorch_remap, registry, safetensors, mod | Model catalog, download + SHA-256 verification, local cache, safetensors reader, PyTorch weight remapping. |
| `runtime/` | burn_engine, onnx_engine, kernels/projection, mod | Burn tensor helpers, an ONNX graph evaluator over `onnx-ir`, CubeCL basis projection / template filter kernels, device selection. |
| `models/` | dartsort, emusort (+ alias `myomatrix`), kilosort4, spikenet2, unitrefine | Model-family wrappers around catalog entries. |
| `catalog/models.json` | 10 entries | Model metadata: URLs, hashes, sizes, I/O specs. |
| `examples/emusort_nwb_zarr.rs`, `build.rs` | | Example pipeline; build script watching an `onnx/` folder. |

Dependencies: dsp-core (compute), dsp-base, dsp-synapse, burn-tensor, burn-flex / burn-wgpu
(optional), onnx-ir, safetensors, memmap2, cubecl, bytemuck, serde(_json), thiserror, anyhow,
tracing, sha2, hex, dirs, **reqwest** (network). dsp-io only as a dev-dependency.

## Findings

### Catalog (`catalog/models.json`)
- CAT1 **Fabricated integrity data.** 7 of 10 `sha256` values are hashes of placeholder inputs:
  `spikenet2/ieeg-detector-v1` = SHA-256 of `""`; `unitrefine/curation-v1` = **SHA3**-256 of `""`;
  `dartsort/singlechan-denoiser-v1` = SHA-256(`"test"`); `dartsort/spatiotemporal-vae-v1` =
  SHA-256(`"4"`); `cebra/…` = SHA-256(`"6"`); `cascade/…` = SHA-256(`"7"`); `deepinterpolation/…` =
  SHA-256(`"8"`). The other 3 (bombcell, both Kilosort4) are unverified. Any "verified" download
  either fails or verifies nothing.
- CAT2 **URLs / artifacts unverified**: custom schemes (`hf://`, `gh://`, `zenodo://`) pointing at
  repositories / files that may not exist (e.g. `hf://bdsp-core/spikenet2-onnx`, ONNX exports of
  PyTorch / Keras models whose upstreams do not publish ONNX). Sizes look invented except the two
  Kilosort `.npy` (1592 B = a `6 × 61` f32 array + header).
- CAT3 Kilosort4 I/O specs disagree with the artifacts: `wPCA.npy` is `6 × 61` but the spec says
  input `[-1, 61, 8]` → output `[-1, 48]`; `wTEMP.npy` spec `[-1, 61]` → `[-1, 6]`.
- CAT4 Entries with no model code: `cebra/*`, `cascade/*`, `deepinterpolation/*` (catalog only).

### hub/
- HUB1 `download_and_verify` **skips verification when `sha256` is empty** (`!expected.is_empty() &&`):
  an entry without a hash is trusted silently. Combined with CAT1 the integrity story is hollow.
- HUB2 `size_bytes` is never checked.
- HUB3 `HubCache::status` re-hashes the whole file on every call (`ModelHub::list` hashes every cached
  model, e.g. 184 MB for deepinterpolation); `index.json` stores the hash but is not used for it.
- HUB4 `gh://` resolves to `raw.githubusercontent.com`: Git-LFS files and release assets are not
  served there.
- HUB5 Blocking `reqwest` network client inside the inference library (pulled at `from_hub` time,
  including in unit tests — CAT1 makes them fail). Fetching artifacts is a separate concern from
  running models.
- HUB6 `registry.rs` (`ProbePreset`, `ModelPresetConfig`: per-probe `k_neighbors`, snippet lengths,
  latent dims, sample rates) has **no users**; values unexplained.
- HUB7 `pytorch_remap.rs` (`PyTorchWeightAdapter`, rules, `transpose_2d_slice`, `flip_conv1d…`,
  `permute_kio_to_oik`) has **no users**; `transpose_2d_slice` duplicates `dsp_base::core::layout`;
  the default stripped prefixes include `"model."` (strips a real top-level module name).
- HUB8 `safetensors.rs`: F32 only, copies every tensor into a `Vec` (doc says "zero-copy").
- Good: streaming SHA-256 while downloading, `.part` file + rename, provider URI resolvers tested.

### runtime/
- RT1 **Per-operation host round trips.** `burn_linear_2d` / `burn_conv1d` upload the input *and
  the weights* from host slices, compute one op and download the result — every layer of every
  call. Weights are never kept on the device; the GPU path is likely slower than the CPU one.
- RT2 Burn runs on its **own device**, separate from the CubeCL `ComputeClient` the rest of the
  workspace uses: model inputs produced on the device by dsp-synapse (snippets in VRAM) must go
  through the host to reach a model.
- RT3 `ComputeTarget` other than WGPU (CUDA, HIP, CPU) **silently falls back to Burn-Flex on the
  CPU** (`_ =>` arm).
- RT4 `default_compute_target` (`ComputeTarget::from_env`) chooses the device inside the library;
  every model constructor falls back to it (`target: Option<ComputeTarget>`). Same rule break as
  in dsp-synapse (fixed there).
- RT5 Uses `DspError::Model`, removed from dsp-core; re-exports dsp-core compute types
  (`ComputeError, ComputeTarget, ComputeTask, LaunchGeometry`) as its own API (alias).
- ONNX1 `onnx_engine.rs` is a hand-written interpreter over `onnx-ir`: element-wise ops on the host,
  Gemm / MatMul / Conv1d through RT1 per node; constant initializers re-converted to `f32` on every
  node evaluation (`resolve_arg`).
- ONNX2 **Broadcasting is wrong** except for equal sizes, scalars or a trailing-axis repeat
  (`i % b.len()`): e.g. `[N, C, T] + [C, 1]` (per-channel bias) is computed incorrectly; the output
  shape is taken from the larger operand.
- ONNX3 Conv1d **ignores the right padding** (`Explicit(l, _r)` → `l`): asymmetric pads give the
  wrong length / values.
- ONNX4 GELU uses the tanh approximation; ONNX `Gelu` default is exact (erf). Softmax ignores its
  `axis` (always last). MatMul only 2-D (no batched). Small op set; `run` uses only the first input
  / output. `burn-import` / `burn-onnx` (ONNX → Burn code generation) exist for this.
- K1 `kernels/projection.rs`: `temporal_basis_project` / `_reconstruct` are matrix products
  (dsp-base `linalg` matmul / `DeviceProjection` already exist); `universal_template_filter` is a
  bank of FIR correlations + rectify + L2 (dsp-base FIR). f32 only, raw `client.empty(n * 4)`,
  host slices in and out on every call, a new client per call via `ComputeTarget::run`.

### models/
- MOD1 **Fabricated "pretrained" EMUsort models.** `EmusortBasisEmbedder::from_canonical` builds a
  12-row basis from hand-picked Hermite-Gaussian polynomials and sines (Gram-Schmidt);
  `EmusortTemplateMatcher::from_canonical` builds 6 hand-written "MUAP" shapes. Documented as
  "pretrained or canonical"; model ids `emusort/temporal-basis-150-12pc-v1`,
  `emusort/universal-muap-templates-v1` **do not exist in the catalog**. Only `from_npy` loads real
  data. The example (`emusort_nwb_zarr.rs`) runs on the synthetic templates.
- MOD2 `EmusortSortConfig` defaults (150 samples, 6.5 σ, 60-sample refractory, 6000 µm radius,
  2.5–6 m/s, 12 PCs, 3–10 clusters) are unsourced; nothing reads `min/max_conduction_velocity`,
  `num_temporal_pcs`, `min/max_clusters`, `dedup_window_samples`, `probe_kind`.
- MOD3 `EmusortLatencyAligner`: another lag cross-correlation (per-lag normalized, integer only;
  dsp-base `math::xcorr` now exists); its conduction-velocity fields are unused.
- MOD4 Kilosort4: `Kilosort4BasisEmbedder` = projection onto `wPCA` (fine, matches KS4's use of a
  6 × 61 temporal basis). `Kilosort4TemplateMatcher` is **not Kilosort4's detection** (KS4 thresholds
  per-template projections with spatial templates): it is a universal-template energy detector
  (`√Σ max(0, ⟨x, tᵢ⟩)²` vs `k · σ(trace)` — energy compared with a voltage σ). Centre offset from
  template 0 only, applied to all.
- MOD5 **Two more peak pickers** (KS4 matcher with "replace if larger", EMUsort matcher greedy) —
  `dsp_base::peaks` now covers both; the EMUsort and KS4 matchers are near-copies of each other.
- MOD6 DARTsort denoiser hard-codes an architecture (`conv1` → ReLU → `fc`) with key names that are
  very likely not upstream's (DARTsort's single-channel denoiser has more conv layers and ships as
  a PyTorch checkpoint); output length is not checked against the snippet length. VAE: upstream
  ONNX export unverified.
- MOD7 UnitRefine: upstream models are scikit-learn classifiers (skops), not ONNX logit networks;
  the catalog's 8-metric input is unverified; softmax applied to outputs that may already be
  probabilities. `UNITREFINE_BOMBCELL_MODEL_ID` exported, never used.
- MOD8 SpikeNet2: no preprocessing (the spec's `z_score_per_channel` is not applied), no resampling
  to the model's 128 Hz, not a `SpikeDetector`.
- MOD9 Backward-compatibility aliases: module `myomatrix`, types `Myomatrix*`,
  `Kilosort4Detector`, `EmusortDetector`; constants `MYOMATRIX_*`.
- MOD10 Every model takes `Option<ComputeTarget>` and stores it (RT4); none takes a client or
  device buffers; `SnippetBatch` literals miss `peak_index` (synapse step 2).
- MOD11 Tests depend on the network (`from_hub`) and on the fabricated catalog; use moved synapse
  items (`dsp_synapse::tetrode`, `waveform_peak_to_peak`).

### examples/, build.rs, Cargo.toml
- EX1 `emusort_nwb_zarr.rs`: a 373-line application pipeline (filtering, whitening, detection,
  dedup, localization, clustering, NWB export) on the synthetic EMUsort templates; **writes
  `/units` into the input NWB store in place**; hard-codes WGPU; uses moved / removed synapse items.
- BLD1 `build.rs` watches an `onnx/` folder that does not exist and emits nothing.
- DEP1 Unused dependencies: `memmap2`, `thiserror`, `tracing`; `anyhow` in a library API (hub) next
  to `DspResult` elsewhere.

## Summary (read first)
1. **Integrity / provenance**: the catalog's hashes are placeholders (CAT1), URLs and I/O specs
   unverified or wrong (CAT2, CAT3); the EMUsort models are synthetic (MOD1); several wrappers
   assume architectures / formats upstream does not publish (MOD6, MOD7). What is real today: the
   Kilosort4 `wPCA` / `wTEMP` arrays (if their URL resolves) and any user-supplied `from_npy` /
   `from_onnx_file` / `from_safetensors_file`.
2. **Performance model is inverted**: per-op host round trips with weights re-uploaded (RT1), Burn on
   a separate device (RT2), silent CPU fallback (RT3).
3. **Correctness of the ONNX interpreter** (ONNX2–4).
4. **Duplicates of dsp-base** (K1, MOD3, MOD5, HUB7) and **dead code** (HUB6, HUB7, MOD2 fields,
   DEP1, BLD1).
5. **Rule breaks** shared with pre-cleanup synapse: device chosen inside (RT4), aliases (MOD9,
   RT5), network-dependent tests (MOD11).

## Decisions / context from the user
- 2026-10-05: EMUsort is a fork of Kilosort (same base, diverged for EMG / Myomatrix). Shared
  machinery should be shared; what is EMUsort-specific (conduction-velocity constraints, latency
  alignment, MUAP defaults, EMG array geometries) stays a variant of it, not a copy (today the
  EMUsort and KS4 matchers are near-duplicates, MOD5).
- 2026-10-05: intent — dsp-synapse-ml distributes neuro models (Kilosort-like sorters, denoisers,
  curators) and runs them.
- 2026-10-05: structure — `kilosort4/` and `emusort/` stay separate modules named after the
  published sorters; their shared parts become neutral dsp-synapse components
  (`features::TemporalBasisEmbedder`, `detection::TemplateBankDetector`); EMUsort-specific stages
  (latency alignment, conduction-velocity bounds via `estimate_hdemg_conduction_velocity`) live in
  its module.
- 2026-10-05: catalog — keep only verifiable entries, mark the rest unavailable; remove the
  synthetic EMUsort models.
- 2026-10-05: hub — new crate **`dsp-synapse-hub`**: fetch / verify (hash + size, no empty hash) /
  cache by URI, generic manifest; all network code there. dsp-synapse-ml owns its model catalog,
  loads from paths, `hub` feature resolves catalog ids through dsp-synapse-hub; dsp-cli / app /
  Python use both.
- 2026-10-05: runtime — `burn-onnx` with device-resident weights on the shared CubeCL client,
  replacing the hand-written interpreter and per-op Burn calls.

Plan: 1 catalog → 2 dsp-synapse-hub → 3 shared components in dsp-synapse + kilosort4 / emusort →
4 burn-onnx runtime → 5 cleanup (aliases, dead code, dependencies, build.rs).

## EMUsort upstream (checked online 2026-10-05)
- Paper: O'Connell, Michaels, …, Pachitariu, Pruszynski, Sober, Pandarinath, *High performance
  sorting of motor unit action potentials with EMUsort*, bioRxiv 2026 (CC-BY 4.0; PMC12803147).
- Code: `snel-repo/EMUsort` (Python CLI on SpikeInterface, **GPL-3.0**) + `snel-repo/Kilosort4`
  (modified KS4 fork; comparisons against KS4 v4.0.18). Datasets released openly.
- **No pretrained weights**: EMUsort learns templates per recording (`templates_from_data = True`:
  temporal PCs from detected waveforms, spatial components during matching). So there is nothing to
  download — the `from_canonical` "models" stand in for something that does not exist upstream.
- Changes vs KS4 (paper Methods): cross-channel delay removal (cross-correlation over ±2 ms,
  reference = channel maximizing summed correlation; `remove_chan_delays`); `n_pcs` 6 → 9;
  `n_templates` 6 → 9; `Th_single_ch` a list `[6, 9, 12, 15]`; HDBSCAN outlier rejection before
  K-means (`remove_spike_outliers`); `do_CAR = False`; `nskip` 25 → 2. Unchanged:
  `Th_universal = 9`, `Th_learned = 8`, `nt = 61`; `template_sizes = 5`, `nearest_chans = 10`.
- Workspace is MIT OR Apache-2.0: reimplement from the paper; do not port GPL code.
- 2026-10-05: sorters built from papers (Kilosort4, EMUsort) stay in dsp-synapse-ml, under
  `src/sorters/`; pretrained-weight models stay under `src/models/`.

## Kilosort4 / EMUsort `wPCA` / `wTEMP` (checked online 2026-10-05)
- Upstream KS4 (`MouseLand/Kilosort`, `kilosort/utils.py::template_path`): **one file `wTEMP.npz`**
  with arrays `wTEMP` and `wPCA`, from `https://osf.io/download/6807fb5958b763aae139aa60/`, cached
  in `~/.kilosort/`. Downloaded and checked: **3432 bytes, SHA-256
  `cae1c96f8f4150be0a39627515750b70c4bc3548177cf487ae3c013f1ca6abd8`**; `wTEMP.npy` `<f4 (6, 61)`
  C order, `wPCA.npy` `<f4 (6, 61)` **Fortran order**. The catalog's
  `gh://MouseLand/Kilosort@main/kilosort/models/{wPCA,wTEMP}.npy` do not exist (CAT2 confirmed).
- EMUsort fork (`snel-repo/Kilosort4`, parent MouseLand/Kilosort): same `wTEMP.npz`, but downloaded
  from `https://www.kilosort.org/downloads/wTEMP.npz` (TLS certificate does not match the host
  today). Only used when `templates_from_data = False`.
- Default in both: `templates_from_data = True` → `extract_wPCA_wTEMP` learns them per recording:
  single-channel clips → `TruncatedSVD(n_pcs)` → `wPCA`; `KMeans(n_templates, n_init = 10)` on clips
  (EMUsort: after HDBSCAN outlier removal) → `wTEMP`, each row L2-normalized. KS4: 6 / 6; EMUsort
  config: 9 / 9 — so **EMUsort has no fixed arrays of its own**; its 9-component sets exist only when
  learned from the recording.
- Format note: `.npz` is a zip of `.npy`; dsp-io reads `.npy` only (Fortran order supported).
- 2026-10-05: every sorter and model carries its provenance (upstream version, license, where data
  was downloaded from, paper link with DOI) so authorship is visible and explorable.
- 2026-10-05: split — `sorters/` = algorithms reimplemented from papers (`kilosort4`, `emusort`);
  `models/` = pretrained networks run as-is (`dartsort` denoiser / VAE, `spikenet2`, `unitrefine`;
  `cebra`, `cascade`, `deepinterpolation` catalog-only, unavailable).

## Provenance checks (2026-10-05, DOIs resolved through doi.org)
| Entry | Paper (verified) | Code / license |
|---|---|---|
| kilosort4 | Pachitariu, Sridhar, Pennington, Stringer, *Spike sorting with Kilosort4*, Nature Methods 2024, `10.1038/s41592-024-02232-7` | `MouseLand/Kilosort` GPL-3.0, latest v4.1.3 |
| emusort | O'Connell et al., *High performance sorting of motor unit action potentials with EMUsort*, openRxiv 2026, `10.64898/2026.01.06.697952` (CC-BY 4.0) | `snel-repo/EMUsort` GPL-3.0 (HEAD `a06bb60`), fork `snel-repo/Kilosort4` GPL-3.0 |
| dartsort | Boussard, Windolf, Hurwitz, Lee …, *DARTsort: …*, openRxiv 2023, `10.1101/2023.08.11.553023` | `cwindolf/dartsort` |
| unitrefine | Jain, Greene, Halcrow, Swann …, *UnitRefine: …*, openRxiv 2025, `10.1101/2025.03.30.645770` | `SpikeInterface/UnitRefine` |
| bombcell | Fabre, Van Beest, Peters, Carandini …, Zenodo 2023, `10.5281/zenodo.8172821` | `Julie-Fabre/bombcell` |
| cebra | Schneider, Lee, Mathis, Nature 2023, `10.1038/s41586-023-06031-6` | `AdaptiveMotorControlLab/CEBRA` |
| cascade | Rupprecht et al., Nature Neuroscience 2021, `10.1038/s41593-021-00895-5` | `HelmchenLabSoftware/Cascade` |
| deepinterpolation | **catalog DOI is another paper** (Li et al., calcium imaging); correct: Lecoq et al., *Removing independent noise in systems neuroscience data using DeepInterpolation*, Nature Methods 2021, `10.1038/s41592-021-01285-2` | `AllenInstitute/deepinterpolation` (no SPDX license on GitHub) |
| spikenet2 | `10.1001/jamaneurol.2019.3542` **unverified** (not resolvable via DOI content negotiation) | `bdsp-core/SpikeNet2` (no SPDX license on GitHub) |

### docs/sorters (added 2026-10-02, commit 1a83454) — to rewrite in the docs phase
- DOC1 `emusort/*` describes the invented design (150-sample `wTEMP_EMG`, 12-PC `wPCA_EMG`, cBSS /
  density clustering, CAR on) instead of the paper's (9 PCs, 9 templates, `Th_single_ch` list, CAR
  off, channel-delay removal, HDBSCAN); cites only the Myomatrix array paper (eLife 2023), not
  EMUsort's own paper.
- DOC2 `kilosort4/01_intro.md` presents `wTEMP.npy` / `wPCA.npy` as pretrained weights; by default
  KS4 learns them per recording and the predefined set is one `wTEMP.npz`.

## Kilosort4 / EMUsort algorithm facts used for `sorters/` (read from upstream 2026-10-05; written here, not ported)
Data `X` = preprocessed (CAR, 300 Hz high-pass, local whitening over 32 channels) batch
`[channels, batch_size = 60000 + 2·nt padding]`; thresholds in whitened σ.

**Universal templates from data** (`templates_from_data = True`, default):
1. Clips: every `nskip`-th batch (KS4 passes 25; EMUsort config 2). Per threshold `Th` of
   `Th_single_ch` (KS4: 6; EMUsort: list `[6, 9, 12, 15]`): peaks where `|X|` equals its max over
   ±4 samples × ±5 channel indices and `|X| > Th`; keep isolated ones (exactly one such peak within
   ±6 channel indices × ±`nt // 2` samples); drop the first / last `nt` samples; union over
   thresholds without duplicates. Clip = `X[ch, t − nt0min .. t − nt0min + nt]`. At most 500 000.
2. Scale all clips by one factor: `1 / sqrt(std over clips of ‖clip‖²)`.
3. `wPCA` = top `n_pcs` right singular vectors of the (uncentred) clip matrix (TruncatedSVD).
4. EMUsort only (`remove_spike_outliers`; fork default False, EMUsort config True): HDBSCAN
   (`min_cluster_size = hdbscan_min_cluster_size = 20`, Euclidean) on the clips when there are
   ≥ max(20, min_cluster_size); drop label −1.
5. `wTEMP` = k-means centres (`n_templates`, 10 initialisations) of the kept clips, rows L2-normalized.
Predefined alternative: `wTEMP.npz` (6 × 6 × 61, see above).

**Universal-template detection**:
- Template centres per shank: `y` from min to max in steps of `dmin / 2` (`dmin` = median vertical
  contact spacing unless set), `x` = `round((xmax − xmin) / (dminx / 2)) + 1` points (`dminx = 32`
  µm); meshgrid. Each centre: `nearest_chans = 10` nearest contacts (squared distances `ds`); keep
  centres whose nearest contact is within `max_channel_distance = 32` µm; neighbour centres: the
  `nearest_templates = 100` nearest centres.
- Spatial weights `w[s, c, centre] = exp(−ds / σ_s²)`, `σ_s = min_template_size · (s + 1)`
  (`min_template_size = 10` µm, `template_sizes = 5`), L2-normalized over `c`.
- `B[ch, k, t]` = correlation of `X[ch]` with `wTEMP[k]` (centred, padding `nt // 2`).
  `A[s, k, centre, t] = Σ_c w[s, c, centre] · B[iC[c, centre], k, t]`; `As[centre, t] = max_{s,k}
  |A|` (keep the arg-max and its sign).
- `Amax[centre, t]` = max of `As` over the neighbour centres, zero in the first / last `nt`
  samples, then max-pooled over ±`nt0min` samples. Spikes: `As == Amax` and `As > Th_universal = 9`.
- Per spike: amplitude `As`, template id, centre; `y` = Σ relu(B at the arg-max template over the
  centre's channels)-weighted contact `y`; features `X[iC[·, centre], t − nt // 2 ..= t + nt // 2] @
  wPCAᵀ` → `[nearest_chans, n_pcs]`.

**EMUsort channel delays** (`remove_chan_delays`): `max_lag = fs // 500` (2 ms). Over every
`nskip`-th batch: each channel std-normalized then `|·|`; `CC[a, b, lag] += mean_t(roll(X_a, lag) ·
X_b)` on the unpadded part; averaged. `best = argmax_b Σ_a max_lag CC[a, b, ·]`; delays = arg-max
lags of row `best`; applied after whitening as `X[i] ← roll(X[i], −delay_i)` per batch.

Defaults also used: `nt = 61`, `nt0min = int(20 · nt / 61)`, `Th_learned = 8`, `highpass_cutoff
= 300` Hz, `whitening_range = 32`, `batch_size = 60000`.
