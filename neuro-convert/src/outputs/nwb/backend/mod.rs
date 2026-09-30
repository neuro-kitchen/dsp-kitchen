//! Storage behind NWB: groups, attributes and typed datasets at HDMF paths.
//!
//! The NWB types only speak this trait, so HDF5 can be added next to Zarr without touching
//! them. Attribute conventions follow hdmf-zarr (`_DTYPE`, `_ARRAY_DIMENSIONS`, `_LINKS`,
//! `_REFERENCE`), which HDF5 backends translate to native links and references.

pub mod zarr;

use serde_json::{Map, Value};

use crate::error::Result;
use crate::model::SampleType;

pub type Attrs = Map<String, Value>;

/// Streams a large 1-D or 2-D dataset in pieces (safe to share between threads).
pub trait RowSink: Send + Sync {
    /// Writes rows `start..start + rows` (the first dimension) from row-major little-endian
    /// bytes of the dataset's sample type.
    fn write_rows(&self, start: u64, rows: u64, bytes: &[u8]) -> Result<()>;
}

pub trait Backend {
    fn group(&self, path: &str, attrs: Attrs) -> Result<()>;
    /// Scalar string dataset.
    fn string(&self, path: &str, value: &str, attrs: Attrs) -> Result<()>;
    /// 1-D string dataset; `dim` names its dimension.
    fn strings(&self, path: &str, values: &[String], dim: &str, attrs: Attrs) -> Result<()>;
    /// Scalar float64 dataset.
    fn f64_scalar(&self, path: &str, value: f64, attrs: Attrs) -> Result<()>;
    /// Float64 dataset of `shape` (row-major values), dimensions named `dims`.
    fn f64s(&self, path: &str, values: &[f64], shape: &[u64], dims: &[&str], attrs: Attrs) -> Result<()>;
    fn i64s(&self, path: &str, values: &[i64], dim: &str, attrs: Attrs) -> Result<()>;
    /// Starts a large dataset of `shape` and type `ty`, chunked by `chunk_rows` along dimension 0.
    fn stream(&self, path: &str, shape: &[u64], chunk_rows: u64, ty: SampleType, dims: &[&str], attrs: Attrs) -> Result<Box<dyn RowSink>>;

    /// Small 1-D float32 dataset.
    fn f32s(&self, path: &str, values: &[f32], dim: &str, attrs: Attrs) -> Result<()> {
        let n = values.len() as u64;
        let sink = self.stream(path, &[n], n.max(1), SampleType::F32, &[dim], attrs)?;
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        if n > 0 { sink.write_rows(0, n, &bytes) } else { Ok(()) }
    }
}
