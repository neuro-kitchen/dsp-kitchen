# dspk-sorter-si

SpikeInterface's internal spike sorters for dspk, ported to GPU kernels through CubeCL: any GPU
(wgpu, CUDA) or the CPU.

| Feature | Sorter | Upstream |
|---|---|---|
| `sc2` | **SpyKING CIRCUS 2** (matched-filter detection, iterative HDBSCAN splits, circus-omp matching) | SpikeInterface `sorters/internal/spyking_circus2.py` |
| `tdc2` | **Tridesclous 2** (locally exclusive detection, iterative isosplit splits, the tdc peeler) | SpikeInterface `sorters/internal/tridesclous2.py` |

Both are on by default. `shared/` holds what they share beyond `dspk-synapse` (whose
`sorter::components` are SpikeInterface's `sortingcomponents`): SpikeInterface's isosplit
(`shared/isosplit.rs`) and waveform cutting (`shared/waveforms.rs`).

- **Licence: MIT OR Apache-2.0**; ported from SpikeInterface (MIT), whose licence text is in
  `LICENSE-SPIKEINTERFACE` (see `NOTICE.md`).
