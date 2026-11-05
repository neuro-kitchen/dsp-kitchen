# dspk-ks4

**Kilosort4** (Pachitariu et al., Nature Methods 2024) and **EMUsort** (O'Connell et al., eLife
2026, a Kilosort4 fork for motor units in high-density intramuscular arrays), on GPU kernels
through CubeCL: any GPU (wgpu, CUDA) or the CPU.

- Stages: `detect/`, `templates/`, `learned/`, `clustering/`, `matching/`, driven by `runner/`;
  EMUsort's settings and channel-delay alignment in `emusort/`.
- Origin: written from the papers and upstream's published defaults, upstream behaviour compared
  against its code (see `NOTICE.md`).
- **Licence: GPL-3.0** (`LICENSE`), like Kilosort4 and EMUsort. A program that includes this crate
  and is distributed must be GPL-3.0 too. The rest of dspk is MIT OR Apache-2.0.
