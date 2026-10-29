# Notice

`dspk-tdc2` is a port of **Tridesclous 2**, the sorter SpikeInterface ships internally, to Rust and
GPU kernels (CubeCL).

- Upstream: <https://github.com/SpikeInterface/spikeinterface>, `spikeinterface/sorters/internal/tridesclous2.py` and the
  `sortingcomponents` it uses, licensed under the MIT License, "Copyright (c) 2019
  SpikeInterface". Its licence text, which must accompany this port, is in
  `LICENSE-SPIKEINTERFACE` (an exact copy of upstream's `LICENSE`).
- Shared components (detection, peelers, splits, merges, templates) live in `dspk-sort`
  (`components`, same notice); MountainSort 5 pieces come from `dspk-ms5` (Apache-2.0).

The stages follow upstream's; the computation is rewritten for the GPU. This crate is
distributed under MIT OR Apache-2.0 (`LICENSE-MIT`, `LICENSE-APACHE`).
