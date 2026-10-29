# dspk-tdc2

**Tridesclous 2**, SpikeInterface's internal sorter (iterative isosplit clustering, the tdc
peeler), ported to GPU kernels through CubeCL.

- `lib.rs` (settings, sorter), `runner/` (the run); shared components in `dspk-sort`.
- **Licence: MIT OR Apache-2.0**; ported from SpikeInterface (MIT), whose licence text is in
  `LICENSE-SPIKEINTERFACE` (see `NOTICE.md`).
