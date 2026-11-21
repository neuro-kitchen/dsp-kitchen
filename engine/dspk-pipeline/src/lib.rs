//! dspk pipelines as data: a versioned node-graph spec ([`PipelineSpec`]) and an executor
//! ([`run`]) that streams a recording through dspk-ops stages on the device.
//!
//! ```json
//! {"dspk_pipeline": 1,
//!  "nodes": {"raw": {"op": "read", "path": "recording.bin"},
//!            "hp":  {"op": "iir", "band": "highpass", "cutoff": [300.0], "mode": "forward"},
//!            "out": {"op": "write", "path": "filtered.zarr", "format": "zarr"}},
//!  "edges": [["raw", "hp"], ["hp", "out"]]}
//! ```
//!
//! - JSON is canonical; TOML is accepted for hand-written files ([`PipelineSpec::load`]); the JSON
//!   Schema of the spec is [`PipelineSpec::json_schema`].
//! - The executor reads windows with read-ahead, uploads each window once (integer recordings as
//!   stored integers, scaled on the device), runs every stage on the device, downloads once and
//!   hands the result to a writer thread (raw, or Zarr chunks directly); the canonical spec is
//!   written next to the output as provenance (`<output>.pipeline.json`). The report counts the
//!   bytes moved at each step and the time spent in each ([`StageTimings`]).
//! - One spec runs over many inputs (globs, lists) and may read several recordings as one (merge).
//! - [`plan`] says what a run would do (mode, halos, windows, bytes) without touching the device;
//!   [`run_with`] reports progress after each window and can cancel.
//! - [`chain`] plans a chain of nodes once for every use: a view's preview, a batch run, a live
//!   device (`live`), each refusing only what it cannot run, with the reason. `view` (feature
//!   `view`) is a view's processed signal: the desktop app, `dspk serve` and a browser page use it.

pub mod chain;
mod fit;
mod prepare;
#[cfg(all(feature = "live", not(target_family = "wasm")))]
pub mod live;
mod run;
mod spec;

pub use fit::{fit_whitening, unfitted_whitening, WhitenFit};
pub use prepare::{default_open, plan, plan_with, ExecutedMode, Plan};
#[cfg(not(target_family = "wasm"))]
pub use run::{events_frame, read_events_zarr, write_events_zarr};
pub use run::{events_csv, run, run_with, write_events, EVENTS_VERSION, DataMovement, Progress, RunReport, StageTimings, PROVENANCE_SUFFIX, RUN_SUFFIX, RUN_RECORD_VERSION, run_record_path, WRITE_QUEUE_WINDOWS};
pub use spec::{
    Band, Chain, Compression, Design, Direction, Edge, ExecMode, Execution, Input, MergePart, Mode, Node, OutputFormat, PipelineSpec, Start, Upload,
    DEFAULT_BLOSC_LEVEL, DEFAULT_CHUNK_SAMPLES, DEFAULT_ZSTD_LEVEL, SPEC_VERSION, WHITEN_DEFAULT_CHUNKS, WHITEN_DEFAULT_FIT_S,
};

/// Why a pipeline could not be read, checked or run.
#[derive(Debug, thiserror::Error)]
pub enum PipelineError {
    #[error("pipeline spec: {0}")]
    Parse(String),
    #[error("invalid pipeline: {0}")]
    Invalid(String),
    #[error("{0}")]
    Io(String),
    #[error(transparent)]
    Dsp(#[from] dspk_core::DspError),
    #[error("compute runtime: {0}")]
    Compute(String),
    #[error("cancelled")]
    Cancelled,
}
#[cfg(feature = "view")]
pub mod view;
