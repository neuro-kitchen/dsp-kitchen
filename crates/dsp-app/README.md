# `dsp-app`: electrophysiology workbench (GPUI)

Desktop app of `dsp-kitchen`, built on GPUI and gpui-kit: workspaces (Explore · Sorting ·
Pipeline · Curation) of docked views on one shared timeline. Explore is built; Curation's data
layer is built and tested, its views come next; Sorting and Pipeline show what they will hold.

## What lives where

The app keeps UI, state and orchestration. Signal maths and sorting I/O live in the library
crates, so other tools (CLI, Python bindings) share them:

| Crate | What the app uses from it |
|---|---|
| `dsp-base` | min/max envelopes, the background `Summarizer` and `MinMaxCache` (`resampler`); percentiles, histograms, axis ticks, point-in-polygon (`math`) |
| `dsp-synapse` | `PhySorting` / `load_spikes` (phy / Kilosort folders, `.sorting.zarr`, NWB `/units`, curation save); `.npy` I/O; correlograms, ISI histograms, firing rates, ISI violations; PCA; `read_snippets` |
| `dsp-io` | recording sources (SpikeGLX, raw + JSON sidecar, NWB Zarr) |

## Layers

1. **`engine/`** (no GPUI type; runs headless, e.g. `--snapshot` and unit tests)
   - `canvas.rs`, `palette.rs`: BGRA frames (uploaded to GPUI as is) and light / dark plot palettes.
   - `work_pool.rs`: shared background threads; jobs keyed by `(view, kind)`, newest wins, stale
     ones cancelled. Renders and curation computations both run here.
   - `data/`: `Dataset` (a recording with its summary and cache), `SourceSet` (the sources of a
     file), `SpikeEventStore`, the timeline overview's `activity`.
   - `time/`: `TimelineState`, `TimeView`, the trace / heatmap rasterizer, hover readouts.
   - `curation/`: the curation state over a `PhySorting` (`mod.rs`: cluster table, open / save),
     its edit history (`history.rs`: merge, split, labels, undo / redo), the data each view draws
     (`derived.rs`: waveforms, features, correlograms, amplitudes, ISI, firing rate; from real
     samples only), the cluster filter (`filter.rs`).
   - `compute.rs`: cache of derived curation data, invalidated per cluster.
2. **`store.rs`, `session.rs`**: the shared state (recording, timeline, per-source channel
   selection, workspace, theme) as one entity emitting fine-grained `AppEvent`s; what is
   remembered between runs (`$XDG_CONFIG_HOME/dsp-app/workbench.json`).
3. **`viewmodels/`**: `ExploreVm`, `TraceVm` (render on change, frames over the view's own
   channel, replaced images released a frame later), `Services` (work pool, hover reader, cache).
4. **`views/`, `app.rs`, `widgets.rs`**: the shell (title bar, workspaces, start screen, status
   bar, help / about), the Explore dock (`TracePanel`, Channels, View settings, Timeline), the plot
   kit (`plot.rs`) for Curation, small widgets.

## Running and testing

```bash
cargo run -p dsp-app                                    # reopens the last recording
cargo run -p dsp-app -- --synthetic 5m --channels 32    # a procedural recording
cargo run -p dsp-app -- --synthetic 10s --snapshot /tmp/traces.png   # headless PNG
cargo test -p dsp-app                                   # unit and headless GPUI tests

# With local data (ignored by default)
DSP_KS4_FOLDER=<kilosort4 saved_results> cargo test -p dsp-app --release -- --ignored
DSP_APP_BENCH_FILE=<recording> cargo test -p dsp-app --release bench -- --ignored --nocapture
```
