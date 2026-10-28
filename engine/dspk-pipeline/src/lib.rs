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
//!   appends the result to the output; the canonical spec is written next to the output as
//!   provenance (`<output>.pipeline.json`). The report counts the bytes moved at each step.
//! - One spec runs over many inputs (globs, lists) and may read several recordings as one (merge).
//! - [`plan`] says what a run would do (mode, halos, windows, bytes) without touching the device;
//!   [`run_with`] reports progress after each window and can cancel.

mod prepare;
mod run;
mod spec;

pub use prepare::{default_open, plan, plan_with, ExecutedMode, Plan};
pub use run::{run, run_with, DataMovement, Progress, RunReport, PROVENANCE_SUFFIX};
pub use spec::{
    Band, Chain, Design, Edge, ExecMode, Execution, Input, MergePart, Mode, Node, OutputFormat, PipelineSpec, Start, Upload,
    DEFAULT_CHUNK_SAMPLES, SPEC_VERSION,
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
