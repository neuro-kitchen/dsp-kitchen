# dspk-sc2

**SpyKING CIRCUS 2**, SpikeInterface's internal sorter (matched-filter detection, iterative
clustering, circus-omp matching), ported to GPU kernels through CubeCL.

- `lib.rs` (settings, sorter), `runner/` (the run); shared components in `dspk-synapse` (`sorter::components`).
- **Licence: MIT OR Apache-2.0**; ported from SpikeInterface (MIT), whose licence text is in
  `LICENSE-SPIKEINTERFACE` (see `NOTICE.md`).
