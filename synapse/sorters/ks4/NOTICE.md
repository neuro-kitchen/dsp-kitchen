# Notice

`dspk-ks4` implements **Kilosort4** and **EMUsort** (a Kilosort4 fork for motor units) in Rust
and GPU kernels (CubeCL). Both upstreams are licensed under the GNU General Public License v3,
and so is this crate (`LICENSE`, the GPL-3.0 text as shipped by Kilosort).

- Kilosort4: <https://github.com/MouseLand/Kilosort>, v4.1.3, by Marius Pachitariu and
  contributors (HHMI Janelia). Paper: Pachitariu, Sridhar, Pennington, Stringer, "Spike sorting
  with Kilosort4", Nature Methods 2024.
- EMUsort: <https://github.com/snel-repo/EMUsort>, commit `a06bb60` (Kilosort4 fork
  `snel-repo/Kilosort4`, base v4.0.18), by Sean O'Connell and Jonathan A. Michaels. Paper:
  O'Connell et al., eLife 2026 (preprint bioRxiv doi:10.64898/2026.01.06.697952).
- Kilosort4's predefined universal templates (`wTEMP.npz`, OSF project `tx3pu`) are not
  included: with the `hub` feature they are downloaded from their upstream location and
  checked by SHA-256. No licence is stated for that file; it is never redistributed.

## How this code relates to upstream

Written from the papers and upstream's published defaults, with upstream's behaviour compared
against its code (named functions, defaults and stage order in the comments); no upstream source
is copied. The computation is new (GPU kernels per stage, windows streamed through `dspk-sort`).
Licensing it under GPL-3.0 keeps it within upstream's terms either way.
