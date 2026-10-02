# Task 08: Sorter Output Persistence — Phy/Kilosort (`.npy`/`.tsv`), Zarr Analyzer (`.sorting.zarr`), and NWB `/units`

## Objective
Implement `crates/dsp-synapse/src/storage/` to save and load `SortingOutput` across the three community-accepted formats:
1. **Phy / Kilosort Flat `.npy` + `.tsv` Directory** (`storage/npy.rs`, `storage/phy.rs`)
2. **Self-Contained Zarr Sorting Store (`.sorting.zarr`)** (`storage/zarr_analyzer.rs`)
3. **NWB `/units` DynamicTable inside `.nwb.zarr`** (`storage/nwb_units.rs`)

## Scope & Files

1. **Zero-Dependency NumPy `.npy` v1.0 Reader/Writer (`crates/dsp-synapse/src/storage/npy.rs`)**:
   - Read and write C-contiguous little-endian `.npy` arrays for `<u8` (`u64`), `<i8` (`i64`), `<i4` (`i32`), `<u4` (`u32`), `<f4` (`f32`), and `<f8` (`f64`) with arbitrary $N$-D shapes.
2. **Phy / Kilosort4 Directory Reader & Writer (`crates/dsp-synapse/src/storage/phy.rs`)**:
   - `save_phy_folder(sorting: &SortingOutput, dir: &Path)`:
     - Writes `spike_times.npy` (`u64`), `spike_clusters.npy` (`i32`), `amplitudes.npy` (`f32`), `spike_positions.npy` (`[N, 3] f32`), `templates.npy` (`[U, T, C] f32`), `templates_std.npy` (`[U, T, C] f32`), `templates_se.npy` (`[U, T, C] f32`), `channel_map.npy` (`i32`), `channel_positions.npy` (`[C, 2] f32`), `cluster_group.tsv`, `cluster_info.tsv`, and `params.py`.
   - `load_phy_folder(dir: &Path) -> DspResult<SortingOutput>`:
     - Reads any Kilosort 1–4 / Phy directory (`spike_times.npy`, `spike_clusters.npy`, optional `amplitudes.npy`, `templates.npy`, `templates_se.npy`, `channel_positions.npy`, `cluster_group.tsv` / `cluster_KSLabel.tsv`, `params.py`) into `SortingOutput`.
3. **Zarr Sorting Analyzer Store (`crates/dsp-synapse/src/storage/zarr_analyzer.rs`)**:
   - `save_sorting_zarr(sorting: &SortingOutput, dir: &Path)` & `load_sorting_zarr(dir: &Path) -> DspResult<SortingOutput>`:
     - Stores root `zarr.json` metadata (`dsp_sorting_format = "zarr_analyzer_v1"`, `sorter_name`, `sample_rate_hz`, `total_samples`, unit quality table, probe layout, drift estimate) alongside binary/NPY-backed or Zarr v3 array chunks for spikes, templates (`mean`, `std`, `se`), and locations.
4. **NWB `/units` Zarr DynamicTable (`crates/dsp-synapse/src/storage/nwb_units.rs`)**:
   - `save_nwb_units(sorting: &SortingOutput, nwb_zarr_dir: &Path)` & `load_nwb_units(nwb_zarr_dir: &Path) -> DspResult<SortingOutput>`:
     - Writes and reads the standard NWB `/units` group (`id`, ragged `spike_times` in seconds + `spike_times_index`, `waveform_mean`, `waveform_sd`, `waveform_se`, `snr`, `firing_rate`, `quality`) inside an existing or new `.nwb.zarr` directory.
5. **Auto-Detecting Unified Dispatch (`crates/dsp-synapse/src/storage/mod.rs`)**:
   - `save_sorting(sorting: &SortingOutput, path: &Path)` and `load_sorting(path: &Path) -> DspResult<SortingOutput>` dispatching automatically based on path extension/contents (`.nwb.zarr`, `.sorting.zarr` / `.zarr`, or Phy/Kilosort folder).

## Verification
- Round-trip unit tests in `crates/dsp-synapse` verifying lossless save & load across Phy/Kilosort folders, `.sorting.zarr`, and NWB `/units` in `.nwb.zarr`.

## Git Commit
```bash
git add crates/dsp-synapse/ .tasks/synapse/
git commit -m "feat(dsp-synapse): add Phy/Kilosort, Zarr SortingAnalyzer, and NWB /units storage"
```
