# refactoring/dsp-synapse

Items removed from `crates/dsp-synapse`.

- Branch / commit at removal: `versions/v0.14` @ `0f7e600`
- Build state: **intentionally broken** until the final fix-up pass.

## Moved to other crates (not parked)

| Original path | New path | When | Reason |
|---|---|---|---|
| `src/storage/npy.rs` | `crates/dsp-io/src/container/npy/mod.rs` | 2026-10-05 (dsp-io phase 2) | `.npy` is a storage container; one copy for the workspace. |
| `src/storage/zarr_store.rs` | `crates/dsp-io/src/container/zarr/mod.rs` | 2026-10-05 (dsp-io phase 2) | Zarr helpers are a storage container; merged with dsp-io's own copies. |
| `zarr_store::infer_nwb_sample_rate` | `crates/dsp-io/src/neuro/nwb/units.rs` | same | NWB-specific; not part of the generic Zarr container. |

## Lines removed from files that stayed

| File | Removed |
|---|---|
| `src/storage/mod.rs` | `pub mod npy;`, `pub mod zarr_store;` |

## Downstream breakage (to fix at the end)

- `dsp-synapse` does **not** depend on `dsp-io` yet: add `dsp-io` (features `zarr`, `neuro`) and
  drop its direct `zarrs` dependency once nothing else uses it.
- Path changes: `storage::npy::…` → `dsp_io::container::npy::…`;
  `storage::zarr_store::…` → `dsp_io::container::zarr::…`;
  `zarr_store::infer_nwb_sample_rate` → `dsp_io::neuro::nwb::infer_nwb_sample_rate`.
- Users: `src/storage/{mod.rs, phy.rs, phy_sorting.rs, nwb_units.rs, zarr_analyzer.rs}`,
  `dsp-app/src/viewmodels/curation.rs` (5 sites).
- These storage files are themselves due to split in dsp-io phase 3 (schema → `dsp-io/neuro`,
  conversion to `SortingOutput` stays in synapse).
