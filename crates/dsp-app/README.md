# `dsp-app` — Electrophysiology Workbench (GPUI)

`dsp-app` is the interactive desktop workbench for `dsp-kitchen`, built on `gpui` and `gpui-kit`. It provides docked workspaces for exploring multi-channel recordings (`Explore`) and curating spike-sorting output (`Curation`).

## Architecture

`dsp-app` is structured into four strict layers:

1. **`engine/` (UI-agnostic)**:
   - Holds zero GPUI types and can run headlessly (`--snapshot` CLI mode and unit tests).
   - `engine/axis.rs`: 1–2–5 nice-step tick calculation.
   - `engine/canvas.rs` & `engine/palette.rs`: BGRA pixel buffer (`Frame`), primitives, and dark/light plot `Palette`.
   - `engine/render_pool.rs`: Worker pool where each view has a `u64` key; a newer request for the same key replaces any unstarted request so panning/zooming never builds a backlog.
   - `engine/data/`: `Dataset`, `SourceSet`, background `Summarizer` (fills `MinMaxSummary` nearest the visible window first), and `SpikeEventStore`.
   - `engine/time/`: `TimelineState`, `TimeView`, `WaveformRenderer` (traces & viridis heatmap), and background `HoverReader`.
   - `engine/compute.rs`: Background compute service and invalidation cache for derived curation data (waveforms, PCA features, correlograms, ISIs, firing rates).
   - `engine/curation/`: Phy / Kilosort / Zarr / NWB sorting loader, cluster metrics, command history (`Undo` / `Redo`), and safe partial saver (`spike_clusters.npy`, `cluster_group.tsv`, `cluster_info.tsv` with `.bak` backups).

2. **`store.rs` & `session.rs` (Application State)**:
   - `Store` is a single GPUI `Entity<Store>` holding the open recording, shared `TimelineState`, per-source channel selections, active `Workspace`, theme preferences, and `CurationStore`.
   - Mutations emit fine-grained `AppEvent`s (`RecordingChanged`, `WindowMoved`, `PlaybackChanged`, `SelectionChanged`, `WorkspaceChanged`, `PaletteChanged`, `SummaryProgress`, `CurationChanged`, `Status`) rather than blanket notifications.
   - `Session` persists recent recordings, recent sorting folders, theme preferences, and workspace layouts to `$XDG_CONFIG_HOME/dsp-app/workbench.json`.

3. **`viewmodels/` (Per-View Reactive State)**:
   - `ExploreVm` and `TraceVm` subscribe to `Store` events, coalesce render requests via `cx.defer`, receive rendered `RenderImage` frames over `async_channel`, and retire replaced textures one frame later via `cx.drop_image`.

4. **`views/` & `widgets.rs` (GPUI Elements & Dock Panels)**:
   - `app.rs`: Title bar menus, workspace switcher, start screen, status bar, and help/about modals.
   - `views/explore.rs`, `views/trace.rs`, `views/panels.rs`: `DockArea` with center `TracePanel`s and collapsible `ChannelsPanel`, `SettingsPanel`, and `TimelinePanel`.
   - `views/plot/`: Reusable plot kit (axes, ticks, pan/zoom, polygon lasso selection, small-multiple grids).
   - `views/curation/`: Spike-sorting curation workspace (Clusters & Similar tables, Waveforms, Features, Correlograms, Amplitudes, ISI, Firing Rate, Templates, Raster, Cluster Map, Template Features, and Probe views).
   - `widgets.rs`: Stateless reusable UI building blocks (`MenuSelect`, `Section`, `EmptyState`, `PanelHeader`, `rail`).

## Running & Testing

```bash
# Launch the workbench
cargo run -p dsp-app

# Open a synthetic 32-channel recording
cargo run -p dsp-app -- --synthetic 5m --channels 32

# Render a headless PNG snapshot
cargo run -p dsp-app -- --synthetic 10s --snapshot /tmp/traces.png

# Run all unit and headless GPUI tests
cargo test -p dsp-app
```
