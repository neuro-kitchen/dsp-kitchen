# dspk-sorter-kilosort

The **Kilosort family** of spike sorters for dspk, on GPU kernels through CubeCL: any GPU (wgpu,
CUDA) or the CPU.

| Feature | Sorter | Upstream |
|---|---|---|
| `ks4` | **Kilosort4** (Pachitariu et al., Nature Methods 2024) | [MouseLand/Kilosort](https://github.com/MouseLand/Kilosort) v4.1.3 |
| `emusort` | **EMUsort** (O'Connell et al., eLife 2026), a Kilosort4 fork for motor units in high-density intramuscular arrays | [snel-repo/EMUsort](https://github.com/snel-repo/EMUsort) `a06bb60` |

Both are on by default. The stages are shared and always compiled: `detect/`, `templates/`,
`learned/`, `clustering/`, `matching/`, `delays/` (channel delays, used by EMUsort), driven by
`runner/`; the entry points are `ks4/` and `emusort/`. With feature `hub`, Kilosort4's predefined
universal templates (`wTEMP.npz`) are downloaded from upstream through `dspk-hub` and checked by
SHA-256 (by default the templates are learned from the recording).

- Origin: written from the papers and upstream's published defaults, upstream's behaviour matched
  by reading its code; no code copied (see `NOTICE.md`).
- **Licence: GPL-3.0** (`LICENSE`), like Kilosort4 and EMUsort. A program that includes this crate
  and is distributed must be GPL-3.0 too. The rest of dspk is MIT OR Apache-2.0.
