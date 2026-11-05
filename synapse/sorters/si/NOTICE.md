# Notice

`dspk-sorter-si` is a port of **SpyKING CIRCUS 2** and **Tridesclous 2**, the sorters
SpikeInterface ships internally, to Rust and GPU kernels (CubeCL).

- Upstream: <https://github.com/SpikeInterface/spikeinterface>:
  `spikeinterface/sorters/internal/spyking_circus2.py`, `spikeinterface/sorters/internal/tridesclous2.py`,
  the `sortingcomponents` they use, and `sortingcomponents/clustering/isosplit_isocut.py`
  (SpikeInterface's port of isosplit6; here `shared/isosplit.rs`), licensed under the MIT License,
  "Copyright (c) 2019 SpikeInterface". Its licence text, which must accompany this port, is in
  `LICENSE-SPIKEINTERFACE` (an exact copy of upstream's `LICENSE`).
- SpyKING CIRCUS 2 follows the design of SpyKING CIRCUS (Yger et al., eLife 2018); Tridesclous 2
  that of Tridesclous (Garcia & Pouzat).
- Shared components (detection, peelers, splits, merges, templates) live in `dspk-synapse`
  (`sorter::components`, same notice).

The stages follow upstream's; the computation is rewritten for the GPU. This crate is
distributed under MIT OR Apache-2.0 (`LICENSE-MIT`, `LICENSE-APACHE`).
