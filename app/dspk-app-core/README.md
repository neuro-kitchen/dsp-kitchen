# `dspk-app-core`: the core of DSP App

The workbench of [`dspk-app`](https://crates.io/crates/dspk-app) for any multichannel signal:

- open, view, process and write recordings;
- the session document (graph of sources, ops, views and writes; commands; undo);
- the Explore and Pipeline workspaces;
- the plugin interface that domain plugins extend (electrophysiology: `dspk-app-synapse`).

It is built on GPUI through gpui-kit. Signal processing comes from `dspk-ops` and
`dspk-pipeline`; the app only draws.

To use the app, install `dspk-app`; this crate is its library.

MIT OR Apache-2.0.
