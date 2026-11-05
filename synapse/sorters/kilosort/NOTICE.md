# Notice

`dspk-sorter-kilosort` implements **Kilosort4** and **EMUsort** (a Kilosort4 fork for motor units) in Rust
and GPU kernels (CubeCL). Both upstreams are licensed under the GNU General Public License v3,
and so is this crate (`LICENSE`, the GPL-3.0 text as shipped by Kilosort).

- Kilosort4: <https://github.com/MouseLand/Kilosort>, v4.1.3, by Marius Pachitariu and
  contributors (HHMI Janelia). Paper: Pachitariu, Sridhar, Pennington, Stringer, "Spike sorting
  with Kilosort4", Nature Methods 2024.
- EMUsort: <https://github.com/snel-repo/EMUsort>, commit `a06bb60` (Kilosort4 fork
  `snel-repo/Kilosort4`, base v4.0.18), by Sean O'Connell and Jonathan A. Michaels. Paper:
  O'Connell et al., eLife 2026 (preprint bioRxiv doi:10.64898/2026.01.06.697952).
- Kilosort4's predefined universal templates (`wTEMP.npz`, OSF project `tx3pu`) are not
  included: with the `hub` feature they are downloaded from their upstream location through the
  `dspk-hub` catalog and checked by SHA-256. No licence is stated for that file; it is never redistributed.

## How this code relates to upstream

Written from the papers and upstream's published defaults. While writing it, upstream's code was
read to match its behaviour: named functions, defaults and stage order are cited in the comments
(e.g. `spikedetect.extract_wPCA_wTEMP`, `kilosort/parameters.py`; for EMUsort, `ks4mods` and its
channel-delay removal). No upstream source is copied; the computation is new (GPU kernels per
stage, windows streamed through `dspk_synapse::sorter`). Because upstream's code was read, this
crate is distributed under upstream's licence, GPL-3.0.
