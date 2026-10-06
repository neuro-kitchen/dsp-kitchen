# dsp-app

![Explore, dark theme](../assets/img/dsp-app_main-darkmode.png)

![Explore, light theme](../assets/img/dsp-app_main-lightmode.png)

*Explore on a Neuropixels recording (385 channels, 30 kHz): channel list (left), traces and a
heatmap of every channel (centre), view settings (right), the timeline with the recording's
activity (bottom). Dark and light themes; plots can stay dark in the light theme.*

## Intent

The desktop workbench (GPUI): workspaces of docked views on one timeline. **The app is the UI
only:** it decides what to show (views, timeline, channel selection, layout) and draws what the
backend answers. Opening files is dsp-io, preparing signals for viewing (pyramids, envelopes,
the build in the background) is dsp-view's [`SignalBackend`](dsp-view.md), and analysis is the
library crates; a remote backend (dsp-stream) can replace the local one without touching the
views.

### Owns
- Views, view models and the shared store (timeline, selection, workspace, session).
- Rasterizing an `Envelope` into pixels (traces, heatmap, grid, scale bar, spike ticks), axis
  ticks, palettes.
- The glue that lists a file's sources (dsp-io) and opens each as a backend.

### Must not contain
- Data-access policy or envelope logic (dsp-view), file formats (dsp-io), DSP or sorting.

## Data flow (Explore)

```text
open path ── dsp_io::sources ──► SourceSet (one entry per signal of the file)
                                     │ first view of a source
                                     ▼
          CachedRecording (dsp-io chunk cache) ─► LocalSignal (dsp-view: pyramid + builder)
                                     │                       ▲ focus(window start), watch(progress)
                                     ▼                       │
   TraceVm ── RenderRequest ──► work pool ── signal.view(View) ──► Envelope ──► pixels
       ▲                                                   (complete = false: redraw on progress)
       └── AppEvent::PyramidProgress (store) ◄── progress ──┘
```

- Each opened source is a `Dataset`: the backend plus display fields (exact rate as Hz, start
  time in seconds, probe geometry from `dsp_io::probe_of`).
- A frame reports whether every column was available; a view redraws on pyramid progress only
  while its last frame was incomplete.
- Hover readouts read one sample through the backend (the chunk cache keeps decoded chunks).
- Amplitudes follow the source's unit: the fixed (not auto-scaled) amplitude is 80 µV expressed
  in that unit for voltages (`nominal_amplitude`), 1 otherwise.
- `--snapshot` renders one view to a PNG after `prepare` builds the visible range.

## Module map

```text
dsp-app/src/
├── main.rs, app.rs, store.rs, session.rs, workspace.rs, actions.rs, widgets.rs, assets.rs
├── engine/            no UI types: data (SourceSet, Dataset, spike events), time (timeline, view
│                      state, renderer, hover), canvas, palette, ticks, work pool
├── viewmodels/        explore, trace, services (work pool, hover reader)
└── views/             explore, panels, trace
```

## Status

- Explore (traces, heatmap, channels, timeline) is wired to dsp-view.
- Curation (sorting folders: clusters, waveforms, correlograms) is parked in
  `refactoring/dsp-app/` while the app is rewired; Sorting and Pipeline are planned.
- The app builds; its tests (headless GPUI) run in the end pass.
