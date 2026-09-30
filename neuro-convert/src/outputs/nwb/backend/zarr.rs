//! Zarr v3 backend in the layout hdmf-zarr writes (read by pynwb / hdmf-zarr and DANDI).
//!
//! Every array carries `_DTYPE` (and `_ARRAY_DIMENSIONS` unless scalar) attributes; strings use
//! the `vlen-utf8` codec; data is optionally gzip-compressed (pure Rust, readable everywhere).

use std::path::Path;
use std::sync::Arc;

use serde_json::{json, Value};
use zarrs::array::codec::bytes_to_bytes::gzip::GzipCodec;
use zarrs::array::builder::ArrayBuilderFillValue;
use zarrs::array::{data_type, Array, ArrayBuilder, DataType};
use zarrs::filesystem::FilesystemStore;
use zarrs::group::GroupBuilder;
use zarrs::storage::{ReadableWritableListableStorage, ReadableWritableListableStorageTraits};

use super::{Attrs, Backend, RowSink};
use crate::error::{Error, Result};
use crate::model::SampleType;

fn err(context: &str, e: impl std::fmt::Display) -> Error {
    Error::format("nwb-zarr", format!("{context}: {e}"))
}

pub struct ZarrBackend {
    store: ReadableWritableListableStorage,
    /// gzip level for datasets (`None` = uncompressed).
    gzip: Option<u32>,
}

impl ZarrBackend {
    /// Creates a new store at `path` (which must not exist unless `overwrite`).
    pub fn create(path: &Path, gzip: Option<u32>, overwrite: bool) -> Result<Self> {
        if path.exists() {
            if !overwrite {
                return Err(Error::Unsupported(format!("{} already exists (use --overwrite)", path.display())));
            }
            std::fs::remove_dir_all(path).map_err(|e| Error::io(path, e))?;
        }
        std::fs::create_dir_all(path).map_err(|e| Error::io(path, e))?;
        let store = FilesystemStore::new(path).map_err(|e| err("store", e))?;
        Ok(Self { store: Arc::new(store), gzip })
    }

    fn array(
        &self,
        path: &str,
        shape: Vec<u64>,
        chunks: Vec<u64>,
        dtype: DataType,
        fill: impl Into<ArrayBuilderFillValue>,
        dtype_attr: &str,
        dims: &[&str],
        mut attrs: Attrs,
    ) -> Result<Array<dyn ReadableWritableListableStorageTraits>> {
        attrs.insert("_DTYPE".into(), json!(dtype_attr));
        if !dims.is_empty() {
            attrs.insert("_ARRAY_DIMENSIONS".into(), json!(dims));
        }
        let mut b = ArrayBuilder::new(shape, chunks, dtype, fill);
        b.attributes(attrs);
        if let Some(level) = self.gzip {
            b.bytes_to_bytes_codecs(vec![Arc::new(GzipCodec::new(level).map_err(|e| err(path, e))?)]);
        }
        let array = b.build(self.store.clone(), path).map_err(|e| err(path, e))?;
        array.store_metadata().map_err(|e| err(path, e))?;
        Ok(array)
    }
}

/// Chunk length that keeps small 1-D datasets in one chunk.
fn whole(len: usize) -> Vec<u64> {
    vec![(len as u64).max(1)]
}

impl Backend for ZarrBackend {
    fn group(&self, path: &str, attrs: Attrs) -> Result<()> {
        let mut g = GroupBuilder::new().build(self.store.clone(), path).map_err(|e| err(path, e))?;
        *g.attributes_mut() = attrs;
        g.store_metadata().map_err(|e| err(path, e))
    }

    fn string(&self, path: &str, value: &str, attrs: Attrs) -> Result<()> {
        let a = self.array(path, vec![], vec![], data_type::string(), "", "str", &[], attrs)?;
        a.store_array_subset(&a.subset_all(), vec![value.to_string()]).map_err(|e| err(path, e))
    }

    fn strings(&self, path: &str, values: &[String], dim: &str, attrs: Attrs) -> Result<()> {
        let dtype = attrs.get("_DTYPE").and_then(Value::as_str).unwrap_or("str").to_string();
        let a = self.array(path, vec![values.len() as u64], whole(values.len()), data_type::string(), "", &dtype, &[dim], attrs)?;
        if values.is_empty() {
            return Ok(());
        }
        a.store_array_subset(&a.subset_all(), values.to_vec()).map_err(|e| err(path, e))
    }

    fn f64_scalar(&self, path: &str, value: f64, attrs: Attrs) -> Result<()> {
        let a = self.array(path, vec![], vec![], data_type::float64(), 0.0f64, "float64", &[], attrs)?;
        a.store_array_subset(&a.subset_all(), vec![value]).map_err(|e| err(path, e))
    }

    fn f64s(&self, path: &str, values: &[f64], shape: &[u64], dims: &[&str], attrs: Attrs) -> Result<()> {
        let chunks = shape.iter().map(|&s| s.max(1)).collect();
        let a = self.array(path, shape.to_vec(), chunks, data_type::float64(), 0.0f64, "float64", dims, attrs)?;
        if values.is_empty() {
            return Ok(());
        }
        a.store_array_subset(&a.subset_all(), values).map_err(|e| err(path, e))
    }

    fn i64s(&self, path: &str, values: &[i64], dim: &str, attrs: Attrs) -> Result<()> {
        let a = self.array(path, vec![values.len() as u64], whole(values.len()), data_type::int64(), 0i64, "int64", &[dim], attrs)?;
        if values.is_empty() {
            return Ok(());
        }
        a.store_array_subset(&a.subset_all(), values).map_err(|e| err(path, e))
    }

    fn stream(&self, path: &str, shape: &[u64], chunk_rows: u64, ty: SampleType, dims: &[&str], attrs: Attrs) -> Result<Box<dyn RowSink>> {
        let mut chunks = shape.to_vec();
        chunks[0] = chunk_rows.clamp(1, shape[0].max(1));
        let (shape, name) = (shape.to_vec(), ty.name());
        let array = match ty {
            SampleType::I8 => self.array(path, shape.clone(), chunks, data_type::int8(), 0i8, name, dims, attrs)?,
            SampleType::I16 => self.array(path, shape.clone(), chunks, data_type::int16(), 0i16, name, dims, attrs)?,
            SampleType::U16 => self.array(path, shape.clone(), chunks, data_type::uint16(), 0u16, name, dims, attrs)?,
            SampleType::I32 => self.array(path, shape.clone(), chunks, data_type::int32(), 0i32, name, dims, attrs)?,
            SampleType::I64 => self.array(path, shape.clone(), chunks, data_type::int64(), 0i64, name, dims, attrs)?,
            SampleType::F32 => self.array(path, shape.clone(), chunks, data_type::float32(), 0.0f32, name, dims, attrs)?,
            SampleType::F64 => self.array(path, shape.clone(), chunks, data_type::float64(), 0.0f64, name, dims, attrs)?,
        };
        Ok(Box::new(ZarrRows { array, ty, cols: shape.get(1).copied(), path: path.to_string() }))
    }
}

struct ZarrRows {
    array: Array<dyn ReadableWritableListableStorageTraits>,
    ty: SampleType,
    cols: Option<u64>,
    path: String,
}

impl ZarrRows {
    fn store<T: zarrs::array::Element>(&self, rows: std::ops::Range<u64>, values: Vec<T>) -> Result<()> {
        let res = match self.cols {
            Some(c) => self.array.store_array_subset(&[rows, 0..c], values),
            None => self.array.store_array_subset(&[rows], values),
        };
        res.map_err(|e| err(&self.path, e))
    }
}

impl RowSink for ZarrRows {
    fn write_rows(&self, start: u64, rows: u64, bytes: &[u8]) -> Result<()> {
        macro_rules! typed {
            ($t:ty, $n:expr) => {
                self.store(start..start + rows, bytes.chunks_exact($n).map(|b| <$t>::from_le_bytes(b.try_into().unwrap())).collect::<Vec<$t>>())
            };
        }
        match self.ty {
            SampleType::I8 => typed!(i8, 1),
            SampleType::I16 => typed!(i16, 2),
            SampleType::U16 => typed!(u16, 2),
            SampleType::I32 => typed!(i32, 4),
            SampleType::I64 => typed!(i64, 8),
            SampleType::F32 => typed!(f32, 4),
            SampleType::F64 => typed!(f64, 8),
        }
    }
}
