//! neuro-convert: read neurophysiology recordings into one neutral model ([`model::Session`])
//! and convert them to publication formats.
//!
//! - `inputs/<format>/` read files into a `Session` (one folder per format, one file per
//!   concern, version branches in their own files).
//! - `outputs/<target>/` write a `Session` out (NWB first).
//! - Everything streams in bounded chunks, so multi-hour recordings never sit in memory.

pub mod common;
pub mod error;
pub mod inputs;
pub mod metadata;
pub mod model;
pub mod options;
pub mod outputs;
pub mod registry;

pub use error::{Error, Result};
pub use model::Session;
pub use options::OpenOptions;
pub use registry::{detect, inputs as input_formats, open, Detection, InputFormat};
