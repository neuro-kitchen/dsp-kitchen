# Changelog

All notable changes, newest first. Versions follow [Semantic Versioning](https://semver.org); in
0.0.x any release may change the API. The crates, the Python package and `dspk-sorters` share one
version number.

## Unreleased

### Added

- **[Documentation](https://neuro-kitchen.github.io/dsp-kitchen/)**: Learn (get started, concepts,
  Python, a course generated from the playground), the App, Operations, Deploy, GPU lessons,
  Validation, Extend, a Lab notebook and the Synapse plugin's guides and sorter pages. CLI, sorter
  settings and pipeline-spec references are generated from the programs.
- Release notes on GitHub come from this file.

## 0.0.8 — 2026-11-07

### Added

- **Documentation site** (`docs/`, Starlight): entry points by purpose as header tabs (Learn, App,
  Operations, Reference, Plugins, More), each with its own sidebar of small chapters; Synapse as
  the first plugin.
- **Python API reference**, generated from the package's source and stubs (no build needed): one
  page per module, numpy style. Every public function and class has Parameters, Returns, Raises
  and Examples; the examples run as doctests on the CPU. Defaults are shown with their values;
  pipeline specs and channel maps in JSON and TOML tabs.
- **`dspk.custom`**: your own Python functions over a recording, window by window (`measure`,
  `transform` with a halo, `apply`, `write` to Zarr; the next window is read on another thread).
  **`dspk.io.ZarrWriter`** writes a store block by block.
- **`dspk.remote`**: a client of `dspk serve` (`Session`: info, files, open a recording read on
  the server, sort / status / wait / stop / logs, fetch; mutual TLS with the enrolled identity).
- **Python `dspk.synapse`**: `clustering` (kmeans, hdbscan, gmm with BIC, density_peaks,
  kde_merge, subsample_peaks), `matching` (template_similarity, suggest_merges,
  matching_pursuit), `spatial` (localize with four methods, drift).
- **Spike locations from the streaming detector** (`locate`; `dspk detect --locate`,
  `detect_recording(locate=True)`): every spike's centre of mass over its K neighbours.
- **Pipeline ops**: `laplacian` (surface Laplacian of an electrode grid), `whiten` (global ZCA
  fitted before the run), per-sample `rectify`, `square`, `sqrt`, `log10`; also as
  `dspk.ops.stage` stages.
- **App**: Add view or container (Rerun-style: tabs, horizontal, vertical, grid; traces, heatmap,
  raster), keyboard-driven; one Inspector for the focused view; views bound to session views and
  named by their source; the Pipeline's Channels family (remap tables, map files in TOML or JSON),
  remaps anywhere in a view's chain previewed as the pipeline runs them; the full operation
  catalog in Explore's Add op; whitening fitted in the background; an Analytics workspace tab
  (placeholder).
- **Playground**: chapters 08 (detection) and 09 (sorting a Neuropixels recording).

### Changed

- The Python stubs state every constant's value, so editors and the docs show defaults, not only
  names.
- A spatial stage made for another channel count is an error with the reason (was a panic).
- IIR autotune benchmarks at most 2^20 steps; pinned passes split time into blocks by channel
  count. App previews never stop to autotune.

### Fixed

- App: views closed or switched to another source no longer leave source and view nodes in the
  Pipeline; icons used by Add view and the Channels family are bundled (a test checks every icon
  the sources name).
- App: a preview read by a branch computes only what it asks for (no slack growing per level).

## 0.0.7 — 2026-11-06

### Changed (breaking)

- **dspk-synapse: one way to call every operation** (`.tasks/30`). Names start with their verb;
  functions on device buffers start with `execute_`; data comes first with its shape, then
  `sample_rate`, then one settings struct; spike times are samples.
  - Renamed: `isi_histogram`, `refractory_stats`, `acg_refractory`, `ccg_refractory`,
    `composite_score`, `sort_composite_score`, `firing_range` → `compute_*`; `kmeans` /
    `hdbscan` → `cluster_kmeans` / `cluster_hdbscan`; `kmeans_points*`, `hdbscan_points*`,
    `cluster_sums`, `bipartite_clustering`, `nearest_neighbours` → `execute_*`;
    `detect_spikes_multichannel` / `_with_sigma` → `detect_spikes_threshold` / `_with_sigma`;
    `extract_snippets_multichannel` / `extract_snippet_batch_multichannel` → `extract_snippets` /
    `extract_snippet_batch`; `compute_neo_energy_1d` → `compute_neo_energy`;
    `haar_dwt_multilevel_1d` → `compute_haar_dwt`; `deduplicate_spikes_spatial_gpu` →
    `deduplicate_spikes_spatial_device`.
  - Settings structs: detectors take `(data, channels, samples, sample_rate, &settings)` with
    `ThresholdSpikeDetector` / `NeoSpikeDetector` / `MatchedFilterSpikeDetector` (refractory in
    ms); `MatchingPursuitMatcher` and `GridConvolutionLocalizer` are their functions' settings;
    new `HdbscanOptions` (replaces `cluster_hdbscan_allow_single`), `DensityPeaksOptions`
    (replaces `cluster_density_peaks_capped`), `KdeMergeOptions`, `AmplitudeCutoffOptions`
    (replaces `compute_amplitude_cutoff_with`), `PsthOptions`, `MepOptions`, `DriftOptions`,
    `SnippetOptions`.
  - Samples, not seconds: `compute_psth`, `compute_instantaneous_firing_rate`,
    `estimate_rigid_drift`, `estimate_nonrigid_drift` take spike samples and `sample_rate`;
    `compute_isi_violations` takes the recording's total samples.
  - Device kernels (`*_kernel`) are no longer public (sorting's stay, hidden, for the sorter crates).
- The kernel benchmarks moved from the crates' `tests/` into `benchmarks/kernels` (crate
  `dspk-benchmarks`; the Kilosort ones in `dspk-benchmarks-sorters`, GPL-3.0); every benchmark
  writes a JSON record.
- dspk-app starts on the start screen; reopening the last recording is a setting (off by default).

### Added

- **Channel maps as pairs** (`[remap] base`, `pairs = [{ from, to, name }]`, TOML or JSON):
  `dspk_core::ChannelMap` (compose, check), `RemappedSource`, `dspk_io::read_channel_map`.
  Pipeline node `remap`: right after `read` it folds into the read; between processing nodes it
  reorders on the device (`PipelineStage::ReorderChannels`); before `write` it may drop channels
  (gathered on the device, per branch). Zarr outputs keep channel names.
- `dspk_ops::core::layout::execute_gather_channels`.
- Probe files: probeinterface and Kilosort 4 probe JSON read as a channel map plus geometry
  (`.json` only); `SensorLayout::remap`, `depth_order`. CLI: `sort` / `detect`
  `--probe-file`, `--channel-map`; `dspk probe --file … --write-map …`.
- **Python:** `dspk.synapse` — `io` (`Probe`, `Sorting`: Phy / `.sorting.zarr` / NWB units),
  `metrics` (22 unit and train metrics, correlograms, PSTH, stimulus averages, sorting
  comparison), `detection` (threshold, NEO, matched filter, dedup, `detect_recording`),
  `extraction` (snippets), `features` (morphology, Haar wavelet, conduction velocity);
  `dspk.ops.layout` (channel maps, `remap`, `gather_channels`), `stage.reorder_channels` /
  `stage.remap`; branched pipeline specs run from Python.
- Branched pipeline specs compute the stages all branches start with once per window.
- dspk-app: servers in Settings (Ctrl+,), Open from server (browse, open recordings remotely,
  fetch), recordings on servers as `dspk://` paths sorted on the server; the Pipeline canvas
  shows the path of a node on hover and each node's run status; recipes as group cards; recent
  entries removable.
- dspk-stream: `RemoteSource` / `RemoteSignal` (a server's recording read like a local one);
  stopped or failed jobs remove their partial output; `dspk serve --runtime`.
- `deploy/smoke-test.sh` (the server image end to end with Podman).
- `plating` (the figures' look, a synthetic recording with known components, the benchmark JSON);
  the playground rebuilt as a course in chapters (01 basics … 07 channels, 10 remote), each
  script with its committed figure.

### Fixed

- dspk-app: adding plot views in Curation no longer panics.
- Python: branched specs (one read, several writes) were refused.

### Security

- `dspk serve` refuses `probe-file` and `channel-map` as job parameters (paths a client could point
  outside the data root).

## 0.0.6 — 2026-11-06

### Security

- **dspk-stream: only enrolled clients connect (mutual TLS).** A server keeps its own
  certificate authority (`Authority`: CA key mode 0600, `clients.json`) and lets in only clients
  holding a certificate it issued and has not revoked (checked at every connection);
  `--allow-anonymous` adds read-only access without one, with a warning. `dspk clients enroll
  <name> [--control]` writes a client's identity file (its certificate and key, and the
  authority, mode 0600); `dspk clients revoke <name>`, `dspk clients list`; `dspk receive
  --identity <file>`. Changed API: `Server::accept` returns the connection and its `Peer` (name,
  permission); new `server_config_with_clients`, `client_config_with`, `ClientIdentity`,
  `ServerTrust::from_identity`.
- dspk-stream: a client could make a server allocate without bound (a view of any width or with
  repeated channels, many such requests at once, unlimited connections, 64 MiB messages). Views
  are now checked (`checked_view`: channels in range and once, at most 65,536 columns and 8 Mi
  values), client messages are limited to 64 KiB, each client to 8 requests at once, servers to
  32 clients (`Server::with_max_connections`, `dspk serve --max-clients`), idle connections close
  after 30 s, and a client proves its address (QUIC Retry) before any handshake work.
- `dspk serve` stops on SIGTERM (container and service managers), not only Ctrl+C.
- Supply chain: `cargo-deny` (`deny.toml`: no known vulnerability, no yanked crate, allowed
  licences only, crates.io only) on every pull request, weekly, and before every release.

### Changed

- **dspk-stream protocol 2** (ALPN `dspk-stream/2`; version 1 peers are refused): the hello is
  answered with `ServerInfo` (server, the client's name and permission, device state, files);
  requests (views, `ListFiles`, `Open`) are `Request` → `Response` on streams of their own;
  refusals come back as `StreamError::Server`. A server can serve the recordings under a data
  root (`files::DataRoot`: absolute paths, `..`, hidden files and links out of it refused;
  `RecordingOpener`, `Device`, `serve_session`), each client opening its own;
  `Session::recording()` replaces `info()`, with `list_files` and `open`.
  `dspk serve --data <dir>`, `dspk receive --open <path>`. Identity files and keys
  (`*.dspk-identity.pem`, `*.key`) are never listed or served.

### Added (server jobs)

- **Sorts on a server.** A control client configures a job (`Configure`: sorter, recording,
  output, parameters, part), starts and stops it; any client reads `Status` and `Logs`. One job
  at a time (Stopped ⇄ Running, Error); every request is checked (known sorter, paths under the
  data root, a new output never overwritten, plain parameter names, none the server sets) and
  every control request audited (`audit.jsonl`, 0600). `dspk-stream::jobs` (`Jobs`,
  `JobRunner`); `dspk serve --data` runs `dspk-cli-sorter`.
- **Fetch** files and directories (a `.sorting.zarr`) under the data root (`Session::fetch`,
  never overwriting, names from the server checked).
- **`dspk remote <addr> --identity <file>`**: `info`, `files`, `sort … [--wait]`, `status
  [--follow]`, `logs`, `stop`, `fetch`. `dspk clients enroll --out -` writes to standard output.

### Added

- `deploy/`: dspk as a server in a Podman container (targets `cpu`, `nvidia`, `mesa`), hardened
  (non-root, read-only root, no capabilities), with a systemd Quadlet unit; tested with CUDA and
  Vulkan on an NVIDIA GPU.
- `dspk serve --name`: the host names clients use, on the self-signed certificate.
- `dspk doctor` says in one line why a runtime is not usable, and warns when the selected one
  has no device.

- **`dspk-app` on crates.io** (`cargo install dspk-app`), with its core `dspk-app-core` and the
  electrophysiology plugin `dspk-app-synapse`. Four workspaces on one timeline:
  - Explore: traces and heatmaps from min/max pyramids, epochs, a timeline of sources.
  - Pipeline: an infinite canvas of the session's graph (sources, ops by family and type,
    views, writes), layers, recipes (reusable groups of ops), a live device preview per op.
  - Sorting: runs the sorters through `dspk-cli-sorter`; units, templates (multi-select) and
    channels; the sorter on the left, its parameters on the right.
  - Curation: correlograms, ISI, templates, amplitudes, waveforms, features and the probe as
    zoomable charts with axes; merge, split (lasso), label, undo, save to `.sorting.zarr`.
- The session is one JSON document (`current-session.json`) changed only by commands, with undo
  and redo. Side panels are resizable and collapsible.
- dspk-pipeline: `PipelineSpec::branches`, several outputs from one read of the recording.

## 0.0.5 — 2026-11-05

### Changed (install)

- The sorter program `dspk-sorters` is now `dspk-cli-sorter`, in `plugins/cli/`: a plugin the
  command line (`dspk sort`) runs as a separate program (GPL-3.0, never linked). Release files
  `dspk-cli-sorter-<target>` and `dspk-cli-sorter-manifest.json`; environment variables
  `DSPK_CLI_SORTER_BIN` and `DSPK_CLI_SORTER_RELEASE_URL` (were `DSPK_SORTERS_*`).
  `dspk sorters install` installs it.

### Added

- Python: `dspk.ops.math.channel_mad` (median absolute deviation per channel, on the device).
- dspk-pipeline: `RunReport::timings` (`StageTimings`: read wait, device, write wait, encode,
  disk), in the CLI output, `dspk run --json` and the Python reports.
- dspk-io: `ZarrWriter`, a Zarr recording written from blocks of columns in order.
- dspk-ops: `core::buffer::{fill, copy}`, `core::layout::transpose_strided`.

### Changed

- **Synapse ready for publication, one crate per upstream lineage** (none of these names was on
  crates.io before):
  - `dspk-synapse-io` merged into `dspk-synapse` (`dspk_synapse::io`).
  - `dspk-synapse-hub` → `dspk-hub` (`hub/`): generic verified downloads with data catalogs per
    domain (feature `synapse`: Kilosort4's `wTEMP.npz`); `dspk_hub::pull(id)`. The
    `dspk-sorters` installer moved into the CLI. `dspk hub` lists the hub catalog.
  - `dspk-synapse-ml`: manifests only say how to run a model; downloads come from the hub
    catalog (no `ModelHub`; `DSPK_MODEL_CATALOG` overrides model manifests, `DSPK_HUB_CATALOG`
    downloads). The CLI no longer depends on it.
  - Sorters: `dspk-ks4` → `dspk-sorter-kilosort` (features `ks4`, `emusort`; templates through
    `dspk-hub`, no ML dependency; provenance text says upstream's code was read);
    `dspk-ms5` → `dspk-sorter-ms5` (with its own isosplit6); `dspk-sc2` + `dspk-tdc2` →
    `dspk-sorter-si` (features `sc2`, `tdc2`; SpikeInterface's isosplit and waveform cutting, no
    MS5 dependency). isosplit is duplicated into its two owners, so `dspk-synapse` is MIT OR
    Apache-2.0 with no Apache-only part.
  - `dspk` (umbrella): features `synapse`, `hub`.
- **Breaking (Rust):** dspk-ops has one error type, `OpsError` / `OpsResult` (was `FilterError`
  for filters and `dspk_core::DspResult` for stored-sample unpacking); `From<OpsError> for
  DspError` keeps `?` working in code returning `DspResult`.
- **Breaking (Rust):** every operation on device buffers is `execute_<op>`, data first:
  `execute_filter(client, input, output, channels, samples, sample_rate, spec)` (was the filter
  first), `resample_poly[_with]` → `execute_resample_poly[_with]`, `decimate` →
  `execute_decimate`. Parameter names of `execute_scaling` / `execute_clamp`: `len`, `scale`,
  `offset` / `min`, `max`.
- Vectorized kernels run on every row length: rows the vector width does not divide (385
  channels, 66 234-sample zero-phase windows) used to fall back to scalar code; FIR, running
  median, `resample_poly`, CAR and CMR now store their last vector lane by lane, and the IIR's
  time-major lanes are padded to the width. Still bit-identical to the scalar kernels. Autotune
  keys renamed, so cached choices are measured again once.
- `dspk run` / `dspk.pipeline.run`: a writer thread takes downloaded windows through a bounded
  queue, so writing overlaps reading and the device; Zarr output is written chunk by chunk as the
  run goes instead of converted from a temporary raw file at the end.
- **Parallel, compressed writing (default on):** Zarr chunks are compressed and stored on several
  threads, raw output frames built on several threads. New `write` node settings `compression`
  (`{"codec": "zstd", "level": 3}` default, `blosc_lz4`, `none`), `chunk_samples` (Zarr chunk
  length, default 5000, independent of the processing windows) and `parallel` (default `true`);
  the same in `dspk_io::ZarrOptions` and Python `dspk.io.write_zarr(..., compression="zstd",
  level=None, parallel=True)` (its `chunk_samples` default is now 5000, `ZARR_CHUNK_SAMPLES`).
  Output is float32 whatever the input's stored type. **Breaking (Rust):** `dspk_io::write_zarr`
  and `ZarrWriter::create` take `&ZarrOptions` instead of a chunk length.
- Dependencies: pyo3 / numpy 0.30, gpui-kit 0.9 (dspk-app), tonic 0.14.9, toml 1.2, maturin 1.16.

### Sorters (speed)

- MountainSort 5 on 2 s × 384 channels (CUDA): **117 s → 51 s**.
  - The per-channel classifiers are fitted on several threads (each with its own device stream;
    same arithmetic per channel; byte-identical output) and their PCA is exact through the
    snippets' Gram matrix (`TopComponentsOptions::dual_max_rows`, MS5
    `classifier_dual_max_rows = 1024`): classifiers 84 s → 14 s. Exact like upstream's, so
    results can differ from 0.0.4's randomized fit (one spike of 115 on the test excerpt).
  - The isosplit6 subdivision runs the two halves of each split on two threads (same labels).
  - `Mountainsort5Config::parallel` (default `true`) switches both off.
- dspk-ops: `TopComponentsOptions::dual_max_rows` (default 0, off): exact PCA of few rows
  through their Gram matrix.
- CLI (`dspk sort`, `dspk-sorters`): a recording with more channels than its probe has sites
  (SpikeGLX's sync channel) is read on the probe's channels, with a note, instead of refused.

### Project

- CI runs only the checks a change can affect; documentation alone runs none.
- `CITATION.cff`, issue forms, pull request template, `CONTRIBUTING.md`, `SECURITY.md`, this
  changelog.

## 0.0.4 — 2026-10-29

### Added

- `dspk-sorters`: every spike sorter (Kilosort4, EMUsort, MountainSort 5, SpyKING CIRCUS 2,
  Tridesclous 2) in one program, GPL-3.0 as a whole, attached to each GitHub release.
  `dspk sort …` runs it as a separate process; `dspk sorters install` downloads and verifies it
  (SHA-256 manifest); `dspk-sorters licenses` prints every licence.
- `dspk-pipeline` is published on crates.io.
- dspk-ops: `math::execute_channel_mad` (NumPy median and median absolute deviation per channel,
  on the device); `reduce::row_kth` (percentiles) and `row_abs_dev_kth`.
- `benchmarks/signal-operations/scipy_timing.py`: time and agreement against scipy / NumPy.
- An example pipeline spec, `playground/base/pipelines/neuropixels_preprocess.json`.

### Changed

- Vectorized kernels (`Vector<F, N>`), chosen per device by the autotuner and bit-identical to the
  scalar ones: FIR, `resample_poly`, running median, common median and common average reference,
  IIR pass. On CubeCL's CPU runtime 2–10× faster (e.g. FIR 175 → 33 ms, common median reference
  1561 → 239 ms on 384 × 60 000); several GPU kernels faster too.
- Common median reference by bitonic sort on GPUs (autotuned against rank counting).
- The `dspk` command line no longer links any sorter: it is MIT OR Apache-2.0.
- Synapse split into one crate per sorter, each with its licence and notice: `dspk-ks4`
  (GPL-3.0), `dspk-ms5` (Apache-2.0), `dspk-sc2`, `dspk-tdc2`; shared parts in `dspk-synapse`.
- SpyKING CIRCUS 2 and Tridesclous 2 compute their noise levels on the device.

### Fixed

- A process could abort at exit after its work ("thread local panicked on drop", upstream
  tracel-ai/cubecl#1981): only wgpu devices are shut down at exit now.
- The Python sdist carries its licence files (PyPI rejected the 0.0.3 sdist).
- Release workflow: a re-run finishes a partial crates.io publish; crates new to crates.io are
  skipped with a warning; PyPI receives only wheels and the sdist.

## 0.0.3 — 2026-10-28

Pipeline engine v2 and `dspk run`; the full set of Python operations; recordings uploaded in their
stored integer type and converted on the device; a faster CPU runtime (contiguous work per thread).

## 0.0.2 — 2026-10-28

Python bindings (`pip install dspk`), a fix to the dspk-ops CPU runtime, CI and tag-driven releases
with trusted publishing.

## 0.0.1 — 2026-10-28

First release of the crates on crates.io.
