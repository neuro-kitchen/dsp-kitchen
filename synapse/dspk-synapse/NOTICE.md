# Notice

`src/sorter/components/` (matched-filter and locally exclusive detection, the circus-omp and tdc
peelers, cluster splits, auto-merging, template cleaning and similarity) is ported from
SpikeInterface's `sortingcomponents`, <https://github.com/SpikeInterface/spikeinterface>,
licensed under the MIT License, "Copyright (c) 2019 SpikeInterface". Its licence text, which
must accompany this port, is in `LICENSE-SPIKEINTERFACE` (an exact copy of upstream's `LICENSE`).

The rest of the crate is written for dspk. The crate is distributed under MIT OR Apache-2.0
(`LICENSE-MIT`, `LICENSE-APACHE`). Ports of other projects' code live in the sorter crates that
use them, under those projects' licences (e.g. isosplit6, Apache-2.0, in `dspk-sorter-ms5`).
