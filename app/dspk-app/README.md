<p align="center">
  <img alt="neuro-kitchen" width="120" src="https://raw.githubusercontent.com/neuro-kitchen/dsp-kitchen/main/assets/img/neuro-kitchen-logo.png">
</p>

# `dspk-app`: DSP App, the desktop workbench of dspk

A desktop app (GPUI, gpui-kit) for long multichannel recordings, with electrophysiology built in.
Four workspaces share one timeline:

- **Explore**: traces and heatmaps of every channel, drawn from min/max pyramids, so hours of
  Neuropixels data scroll smoothly. Includes epochs and a timeline of sources.
- **Pipeline**: an infinite canvas of the session's graph (sources, ops, views, writes), with
  layers, reusable recipes and a live device preview of each op.
- **Sorting**: runs a spike sorter (Kilosort4, MountainSort 5, SpyKING CIRCUS 2, …) through the
  separate `dspk-cli-sorter` program, then shows its units, templates and channels.
- **Curation**: correlograms, ISI, templates, amplitudes, waveforms, features and the probe, as
  zoomable charts. Merge, split (lasso), label, undo and save to `.sorting.zarr`.

The app is `dspk-app-core` (any multichannel signal) plus the electrophysiology plugin
`dspk-app-synapse` (feature `synapse`, on by default).

## Install

```bash
cargo install dspk-app
```

On Linux, GPUI builds against these system libraries:

```bash
sudo apt-get install libfontconfig-dev libfreetype-dev libwayland-dev libxkbcommon-dev libxkbcommon-x11-dev libx11-xcb-dev
```

Processing runs on the GPU through CubeCL (wgpu, CUDA) or on the CPU. All three runtimes are
compiled in by default (features `wgpu`, `cuda`, `cpu`).

Sorting needs the sorter program `dspk-cli-sorter`, of the same version as the app. Download it
from the [GitHub release](https://github.com/neuro-kitchen/dsp-kitchen/releases): the file
`dspk-cli-sorter-<target>`. Rename it `dspk-cli-sorter` and put it next to `dspk-app`
(`~/.cargo/bin`), or point `DSPK_CLI_SORTER_BIN` at it. `dspk sorters install` from the
command line does the same.

## Run

```bash
dspk-app                                    # reopens the last recording
dspk-app recording.ap.bin                   # SpikeGLX .bin/.cbin, raw .bin + JSON .meta, Zarr
dspk-app --synthetic 5m --channels 32       # a procedural recording
dspk-app --synthetic 10s --snapshot traces.png   # one view to a PNG, no window
```

The session (sources, ops, views, epochs) is one JSON document, changed only through commands,
with undo and redo.

## Licence

MIT OR Apache-2.0. The GPL-3.0 sorters are never linked: the app runs `dspk-cli-sorter` as a
separate program.
