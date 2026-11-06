# `dspk-app-synapse`: electrophysiology in DSP App

The plugin of [`dspk-app`](https://crates.io/crates/dspk-app) for neural recordings:

- SpikeGLX and NWB with probe geometry and shanks;
- sortings (`.sorting.zarr`, phy / Kilosort folders) with spike highlight;
- the Sorting workspace: runs sorters through the separate `dspk-cli-sorter` program;
- the Curation workspace: correlograms, ISI, templates, amplitudes, waveforms, features, probe;
  merge, split, label, undo and save.

The analysis (correlograms, ISI, similarity, PCA) comes from `dspk-synapse`; the plugin only
draws it.

To use it, install `dspk-app` (feature `synapse`, on by default).

MIT OR Apache-2.0; never links the GPL-3.0 sorters.
