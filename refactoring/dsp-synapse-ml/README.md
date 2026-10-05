# refactoring/dsp-synapse-ml

Items removed from `crates/dsp-synapse-ml`.

- Branch / commit at removal: `versions/v0.14` @ `0f7e600`
- Build state: **intentionally broken** until the final fix-up pass.

## Parked files

| Parked path | Original path | Removed on | Reason | Replacement |
|---|---|---|---|---|
| `hub/npy.rs` (`NpyTensorF32`) | `crates/dsp-synapse-ml/src/hub/npy.rs` | 2026-10-05 (dsp-io phase 2) | Second `.npy` reader (f32 only, `anyhow` errors). Duplicates the complete reader now in `dsp-io`. | `dsp_io::container::npy::read_npy::<f32>(path)` → `NpyArray { data, shape }` (C order, any dtype converted, Fortran order handled). |

## Lines removed from files that stayed

| File | Removed |
|---|---|
| `src/hub/mod.rs` | `pub mod npy;`, `pub use npy::NpyTensorF32;` |
| `src/lib.rs` | `NpyTensorF32` from `pub use hub::{…}` |

## Downstream breakage (to fix at the end)

`NpyTensorF32::from_file` callers (all only use `from_file`; `from_bytes` had no external users):
- `src/models/kilosort4/basis.rs`, `src/models/kilosort4/matcher.rs`
- `src/models/emusort/basis.rs`, `src/models/emusort/matcher.rs`

`dsp-synapse-ml` already depends on `dsp-io`.
