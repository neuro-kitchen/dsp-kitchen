---
title: Add an app plugin
description: "Bring a domain to the app."
sidebar:
  order: 5
---

The core app works on any multi-channel signal. A plugin brings a domain. `dspk-app-synapse`, which adds
electrophysiology, is the working example.

| Extension | The core asks | In Synapse |
|---|---|---|
| `SourceOpener` | Open this file, before the generic formats | SpikeGLX and NWB, with probes |
| `ChannelLayout` | How channels group, and what to call them | Shanks, site positions |
| `AppPlugin::synthetic` | A recording made up on demand | Drifting units, noise, hum |
| `AppPlugin::trace_overlay` | What to draw over the traces of a window | Spikes coloured by unit |
| `AppPlugin::workspace` | The body of a workspace tab | Sorting, Curation |
| `AppPlugin::panel`, `AppPlugin::view_section` | A panel on the right side bar, for the active view | The linked sorting and its units |
| Per-view data | `views[i].plugins.<id>` in the session, any JSON | The linked `.sorting.zarr` |

Plugins are compiled in, since Rust has no stable plugin ABI. The `dspk-app` binary registers the ones it's
built with before the app starts.

A program under another licence stays a separate process that the plugin runs, and is never linked. That's
how the GPL sorters are used.

For UI work, the app follows the design rules of [designparser](https://github.com/designparser/designparser-mcp).

## In these docs

A plugin gets a folder, `docs/src/content/docs/plugins/<name>/`, with the same shape as Synapse's:
overview, learn, reference and validation. Add it to `src/data/nav.mjs` and `astro.config.mjs`.
