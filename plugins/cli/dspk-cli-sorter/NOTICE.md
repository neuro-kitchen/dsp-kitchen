# dspk-cli-sorter: notice

`dspk-cli-sorter` is one program with every spike sorter of dspk, run on the GPU or the CPU through
CubeCL. It contains Kilosort4 and EMUsort, which are licensed under the GNU General Public License
v3, so the program as a whole is distributed under **GPL-3.0** (`LICENSE`). The dspk libraries,
its command line and its Python package are MIT OR Apache-2.0 and do not contain it: `dspk sort`
runs this program as a separate process.

Source: <https://github.com/neuro-kitchen/dsp-kitchen>, at the tag of this version (`v<version>`,
`dspk-cli-sorter --version`): `plugins/cli/dspk-cli-sorter` and the crates below.

| Sorter | Crate | Upstream | Upstream licence |
|---|---|---|---|
| Kilosort4 | `dspk-sorter-kilosort` (`ks4`) | <https://github.com/MouseLand/Kilosort> v4.1.3 (Pachitariu et al., Nature Methods 2024) | GPL-3.0 |
| EMUsort | `dspk-sorter-kilosort` (`emusort`) | <https://github.com/snel-repo/EMUsort> `a06bb60` (O'Connell et al., eLife 2026) | GPL-3.0 |
| MountainSort 5 | `dspk-sorter-ms5` | <https://github.com/flatironinstitute/mountainsort5> v0.5.9; isosplit6 <https://github.com/magland/isosplit6> (Jeremy Magland, Flatiron Institute) | Apache-2.0 |
| SpyKING CIRCUS 2 | `dspk-sorter-si` (`sc2`) | SpikeInterface, `sorters/internal/spyking_circus2.py` | MIT, "Copyright (c) 2019 SpikeInterface" |
| Tridesclous 2 | `dspk-sorter-si` (`tdc2`) | SpikeInterface, `sorters/internal/tridesclous2.py` | MIT, "Copyright (c) 2019 SpikeInterface" |

Shared sorter components (detection, peelers, merges, templates) are in `dspk-synapse`, under
MIT OR Apache-2.0 (ported from SpikeInterface, MIT). isosplit6 is in `dspk-sorter-ms5`
(Apache-2.0), SpikeInterface's port of it in `dspk-sorter-si` (MIT). Each crate's `NOTICE.md`
describes how its code relates to upstream. Kilosort4's predefined universal templates
(`wTEMP.npz`) are not included: they are downloaded from their upstream location through
`dspk-hub` when asked for and checked by SHA-256.

The full licence texts follow (`dspk-cli-sorter licenses`).
