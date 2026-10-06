# dsp-view — changes and plan

New crate (user, 2026-10-06): everything that prepares signals for viewing lives in one crate,
for a local app or a remote viewer, CPU and GPU, instead of being split across dsp-base,
dsp-stream and dsp-app.

### Owns
Min/max envelopes, envelope pyramids (in memory or in a file), their background builder, GPU
envelope kernels, and the view request / result types that dsp-stream carries.

### Must not contain
Rendering or UI (dsp-app), network transport (dsp-stream), file formats (dsp-io). It must not
**depend on dsp-base** either (user, 2026-10-06): only dsp-core and CubeCL. A reduction that
viewing needs lives here, not in dsp-base.

## Origin (2026-10-06)

The display code parked from dsp-base on 2026-10-05 (`refactoring/dsp-base/resampler/`), moved
first into `dsp-stream/src/decimation/` and then here. The code is unchanged, apart from the
`peak_to_peak` removal (a duplicate of `dsp_base::math::peak_to_peak`) and module paths and docs.

| Was (dsp-base `resampler/`) | Now |
|---|---|
| `minmax.rs` | `src/envelope/fold.rs` |
| `decimate.rs` | `src/envelope/buckets.rs` |
| `summary.rs` (`MinMaxSummary`) | `src/pyramid/memory.rs` |
| `cache.rs` (`MinMaxCache`) | `src/pyramid/file.rs` |
| `summarize.rs` (`Summarizer`) | `src/pyramid/builder.rs` |
| `mod.rs` (`read_pipelined`) | `src/read.rs` (private) |

The dependencies are dsp-core, rayon, memmap2 and tracing. `cargo check -p dsp-view --all-targets`
passes with no warnings. Tests have not been run (end pass).

**Downstream:** dsp-app `dsp_base::resampler::…` → `dsp_view::…` (types unchanged for now).

## Step 1 — one pyramid (2026-10-06, done)

`MinMaxSummary` (paged, in memory, filled where views look) and `MinMaxCache` (a global pyramid in
a file, built front to back) were two designs of one structure. They are replaced by one
`Pyramid`.

- **Layout:** one layout, the file's (`[level][channel][bucket]`, buckets doubling per level);
  `pyramid/layout.rs`.
- **Storage:** memory is an anonymous mapping (the OS commits only the pages that are filled),
  a file is a file mapping; `pyramid/storage.rs`. The file header is unchanged (`VERSION` 1), so
  existing `.minmax` files of the same base still open.
- **Filling:** pages of ≈ `PAGE_SEC` = 1 s, a power-of-two number of level-0 buckets, filled in
  any order.
  - Each page is claimed atomically, so concurrent fills never build a page twice.
  - Levels coarser than a page are merged lock-free by the thread that finishes their last
    child.
  - A file is marked complete when the last page is built.
- **Envelope:** the old file version's exact one (bucket-aligned columns). A column is NaN until
  all of its buckets are built.
- **Bases:** `MEMORY_BASE = 256`, `FILE_BASE = 64` (named, with the size trade-off documented).
- **Builder:** `Summarizer` → `PyramidBuilder` over `Pyramid`. With the focus at sample 0 it is the
  old sequential file build.

| Was | Now |
|---|---|
| `MinMaxSummary::new(source)` | `Pyramid::in_memory(source, MEMORY_BASE)` |
| `MinMaxCache::{open, create}` | `Pyramid::{open_file, create_file}` |
| `MinMaxCache::open_or_create(source, rec, base)` | `Pyramid::open_or_create(source, rec)` (file at `FILE_BASE`, falls back to memory) |
| `MinMaxCache::temporary` | removed (memory storage replaces temporary files) |
| `MinMaxCache::build(source, chunk, cancel, progress)` | `Pyramid::fill(source, start, end, block_values, stop)` or `PyramidBuilder::run` (focus 0) |
| `MinMaxSummary::envelope` (always drew, level 0 when finer than base) | `Pyramid::envelope` → `Ok(false)` when finer than base (read raw) |
| `ready_samples()` | `covers(start, end)` / `is_complete()` |
| `CacheIdentity`, `cache_path` | `PyramidIdentity`, `pyramid_path` |
| `Summarizer` | `PyramidBuilder` |

**Trade-offs:**
- Pages are no longer aligned to the source's storage chunks (they are a power-of-two number of
  buckets), so a compressed store may decode a chunk twice at block edges. Reads group
  consecutive pages to limit this.
- On Windows an anonymous mapping counts against the commit limit even before it is filled.

`cargo check` / `clippy` / `doc` for dsp-view: clean, except one pre-existing clippy note in
`envelope/fold.rs` (`chunks_exact` with a constant size). Tests (the ported file and memory
cases, plus filling in any order and concurrent fills) were written but not run (end pass).

## Step 2 — envelopes on the device (2026-10-06, done)

`envelope/device.rs` (feature `device`):
- **`envelope_kernel`:** one cube per `(channel, column)`. Units stride the column, then a
  pairwise min / max merge in shared memory. NaN is skipped by comparison, as on the host.
- **`envelope_on_device::<R, F>(client, input, channels, samples, first, columns)`:** for any
  `Columns` (even pixel columns or pyramid buckets). Column edges are built on the host as
  `u32` offsets (no 64-bit integers, which WebGPU lacks), clipped to the buffer. Only the
  columns are downloaded; empty or all-NaN columns are NaN, as on the host. It returns an error
  past 32-bit device indexing.
- **Constants:** `PAIR` (values per column) is named and used inside the kernel.
- **Replaces** dsp-base's `row_min_max` (removed there; it had no users).
- **Features:** `device` (`cubecl`, `dsp-core/compute`), and `wgpu` (default) / `cpu` / `cuda` /
  `hip`, which forward to dsp-core's runtimes.
- **Dependencies:** `cargo tree` shows dsp-core, cubecl, memmap2, rayon and tracing, and no
  dsp-base.

`cargo check` (with and without default features), `clippy` and `doc` for dsp-view are clean,
except the pre-existing `envelope/fold.rs` note. The test (`device_envelope_matches_host_fold`)
covers even, sub-range, bucket and single columns, with NaN samples and an all-NaN column, on
every compiled-in runtime. It was written but not run (end pass).

## Step 3 — view types (2026-10-06, done)

`src/view.rs`:
- **`View { channels, start, end, width }`**, with `samples`, `columns` and `validate`.
- **`Envelope`:** `Samples(Vec<f32>)` or `Columns { values, complete }`.
- **`View::read(source, pyramid)`** picks the cheapest exact answer:
  - fewer samples than columns: the samples themselves;
  - columns at or above the pyramid's base: from the pyramid, with `complete` = the window is
    built;
  - otherwise: a raw read folded into even columns.

They are plain Rust types; dsp-stream translates them into its protocol messages. Tests were
written (samples path, pyramid vs raw extremes, invalid views) but not run.

## Step 4 — book page (2026-10-06, done)

`docs/book/src/crates/dsp-view.md`, listed in `SUMMARY.md` and in the introduction's crate table.

## Open

- **dsp-stream:** the session protocol (header, `SignalFrame`, `EnvelopeFrame`, `View` request)
  carrying these types (dsp-stream REVIEW ST1–ST3).
- **dsp-app:** imports from `dsp_view` (see step 1's table).
- **End pass:** run the dsp-view tests (layout, storage, pyramid incl. concurrency, builder,
  device kernel on every runtime, view).
