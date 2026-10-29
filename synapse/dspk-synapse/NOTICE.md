# Notice

`dspk-synapse`'s isosplit6 clustering (`src/sorting/isosplit.rs`) is ported from
<https://github.com/magland/isosplit6> by Jeremy Magland (`isosplit6.cpp`, `isocut6.cpp`,
`jisotonic5.cpp`, `isosplit5.cpp`), licensed under the Apache License 2.0; that file stays under
the Apache License 2.0 (`LICENSE-APACHE`). Paper: Magland & Barnett, "Unimodal clustering using
isotonic regression: ISO-SPLIT", arXiv:1508.04841.

`src/sorter/components/` (matched-filter and locally exclusive detection, the circus-omp and tdc
peelers, cluster splits, auto-merging, template cleaning and similarity) is ported from
SpikeInterface's `sortingcomponents`, <https://github.com/SpikeInterface/spikeinterface>,
licensed under the MIT License, "Copyright (c) 2019 SpikeInterface". Its licence text, which
must accompany this port, is in `LICENSE-SPIKEINTERFACE` (an exact copy of upstream's `LICENSE`).

The rest of the crate is written for dspk and distributed under MIT OR Apache-2.0
(`LICENSE-MIT`, `LICENSE-APACHE`); the crate as a whole is `(MIT OR Apache-2.0) AND Apache-2.0`.
