# dsp-app — parked code and rewiring (2026-10-06)

The app is the UI only: it decides what to show and draws what the backend answers. Data access,
pyramids and envelopes are dsp-view's `SignalBackend` (`LocalSignal` today; a dsp-stream remote
backend later).

## Parked: Curation (the sorter views)

Parked while the main view (Explore) is rewired. Moved here with paths relative to `src/`:

| Parked | What it was |
|---|---|
| `engine/curation/` (`mod.rs`, `derived.rs`, `filter.rs`, `history.rs`) | Sorting data (Phy folders), derived per-cluster data (waveforms, PCA features, correlograms, amplitudes, ISI, firing rate), cluster filter, merge / split history |
| `engine/compute.rs` | `ComputeCache` of derived curation data and its work-pool slots |
| `viewmodels/curation.rs` | Curation view model, sorting-folder detection (`is_sorting`) |
| `views/curation.rs` | Clusters, similar, waveforms and correlogram panels |
| `views/plot.rs` | Plot kit used only by Curation (axes, polylines, lasso via `point_in_polygon`) |

Also removed with it: `SpikeEventStore::from_sorting` (events from a sorting), the
`Open sorting…` menu item, `DspApp::curation`, `Services::cache`, the UI test
`a_phy_folder_opens_in_curation`. The Curation tab shows "planned". `Session::recent_sortings`
and `push_recent_sorting` stay (persisted field).

**To un-park:** sorting files are dsp-io's now — `PhySorting` → `dsp_io::neuro::phy::PhyFolder`
(`write_curation`), `ClusterId` from dsp-io, format detection `dsp_io::neuro::detect_sorting`
(replaces the app's own `has_array` checks); `extract_waveform_pca(client, …)` needs a compute
device: the app chooses one at startup (`--runtime` / `DSP_KITCHEN_RUNTIME`) as the CLI and
Python do. The waveform window's literal 61 becomes a named constant or the sorting's own.

## Rewiring of Explore

- `Dataset` = a `SignalBackend` + display fields (`sample_rate` Hz from the exact `SampleRate`,
  `start_time_sec` from `RationalTime`, `probe` from `dsp_io::probe_of`). It is no longer a
  `RecordingSource`. Summary / cache file / `cache_after_summary` / `Summarizer` removed: one
  pyramid per source in dsp-view, one read of the recording (was summary, then a second read to
  write the file).
- `SourceSet::load`: `CachedRecording` → `LocalSignal::open(source, Some((path, id)))`; the pyramid
  policy (`AUTO_LOD_BYTES`) and `build_caches` moved to dsp-view (`FILE_PYRAMID_MIN_BYTES`, files
  for large recordings automatically). Menu item `Build min/max caches` removed.
- Renderer: `RenderRequest { signal }` (was `source`, `lod`, `summary`); `envelope` asks
  `signal.view(View)` and spreads `Envelope::Samples` / copies `Columns`; its own streamed raw
  read (`stream_raw`) moved into dsp-view (`View::read_into`). Frames report `complete`; a view
  redraws on `AppEvent::PyramidProgress` (was `SummaryProgress`, tied to a fixed base) only while
  its last frame was incomplete. `Store::summarize` → `focus_pyramid`, `summary_label` →
  `pyramid_label`.
- Hover: one sample through the backend; the reader's own chunk copy removed (the source's chunk
  cache does that).
- Units: `NOMINAL_UV = 80` → `nominal_amplitude(unit)` (80 µV in the source's voltage unit, else
  `NOMINAL_OTHER = 1`); `TimeView::set_source` takes a `SignalUnit`.
- `--snapshot`: `prepare(start, end)` builds the visible range, then renders (exact at any zoom).
- `math::ticks` (`nice_step`, `ticks`) un-parked from `refactoring/dsp-base/math/` into
  `engine/ticks.rs`; `math/geometry.rs` (`point_in_polygon`) stays parked with Curation's lasso.
- Named: `PROCEDURAL_CHANNELS_PER_UNIT = 4`, `PROCEDURAL_MAX_UNITS = 64`.
- Test fixture `SpikeEventStore::detect`: new dsp-synapse detection API (`SpikePolarity`,
  `SpikeSpacing`: the deeper of two troughs within 2 ms stays; expectations updated).

`cargo check --workspace --tests`: clean (first time dsp-app builds since the cleanup). Tests
(including the headless GPUI ones) not run — end pass.
