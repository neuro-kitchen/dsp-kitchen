# Task 09: `dsp_kitchen_py` Storage & Sorter Comparison Bindings

## Objective
Expose `SortingOutput`, `save_sorting`, `load_sorting`, `export_to_phy`, `read_kilosort`, `save_nwb_units`, `load_nwb_units`, and `compare_sortings` in `dsp_kitchen_py` with `.pyi` type stubs and end-to-end Python SDK tests.

## Scope & Files

1. **PyO3 Storage & Comparison Bindings (`dsp_kitchen_py/src/synapse/storage.rs` & `comparison.rs`)**:
   - `PySortingOutput` (`SortingOutput`):
     - Properties: `sorter_name`, `sample_rate`, `total_samples`, `num_units`, `total_spikes`.
     - Methods: `unit_ids()`, `spike_train(unit_id)`, `unit_template(unit_id)` (with `mean`, `std`, `se`), `unit_metrics(unit_id)`, `summary_table()`, `save(path, format=None)`, `staticmethod load(path)`.
     - Converters: `StreamingSortResult.to_sorting_output(...)`, `SortingOutput.from_clusters(spike_samples, labels, sample_rate_hz, ...)`, `SortingOutput.from_cbss(cbss_units, sample_rate_hz, ...)`.
   - Top-level functions:
     - `save_sorting(sorting, path, format=None)`
     - `load_sorting(path)`
     - `export_to_phy(sorting, folder)`
     - `read_kilosort(folder)`
     - `save_nwb_units(sorting, nwb_zarr_path)`
     - `load_nwb_units(nwb_zarr_path)`
     - `compare_sortings(sorting_a, sorting_b, delta_time_ms=0.4, agreement_threshold=0.5)`
     - `compare_spike_trains(train_a, train_b, sample_rate_hz=30000.0, delta_time_ms=0.4)`
2. **Python SDK Exports, Stubs & Tests**:
   - Update `dsp_kitchen/_bindings.py`, `dsp_kitchen/__init__.py`, `dsp_kitchen/synapse/__init__.py`, and `dsp_kitchen_bindings/dsp_kitchen_bindings.pyi`.
   - Add round-trip storage (Phy/Kilosort, `.sorting.zarr`, NWB `/units`) and multi-sorter comparison tests in `dsp_kitchen_py/tests/test_sdk.py`.

## Verification
- `PYO3_NO_PYTHON=1 PYO3_CROSS_PYTHON_VERSION=3.13 cargo build -p dsp_kitchen_py --features extension-module`
- `.venv/bin/python dsp_kitchen_py/tests/test_sdk.py` passes 100%.

## Git Commit
```bash
git add dsp_kitchen_py/ .tasks/synapse/
git commit -m "feat(dsp_kitchen_py): expose SortingOutput storage (Phy, Zarr, NWB /units) and compare_sortings"
```
