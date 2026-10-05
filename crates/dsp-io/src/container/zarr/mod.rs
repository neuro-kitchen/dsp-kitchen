//! Zarr v3 groups and arrays on the filesystem (`zarrs`), shared by every Zarr-based format.
//!
//! Writes arrays with `_DTYPE` and `_ARRAY_DIMENSIONS` attributes (the hdmf-zarr convention), and
//! [`read_array`] falls back to legacy `<node>.npy` files when reading older directories.

use std::path::Path;
use std::sync::Arc;

use dsp_core::{DspError, DspResult};
use serde_json::{Map, Value, json};
use zarrs::array::{Array, ArrayBuilder, DataType, data_type};
use zarrs::filesystem::FilesystemStore;
use zarrs::group::GroupBuilder;
use zarrs::storage::{ReadableStorageTraits, ReadableWritableListableStorage};

use crate::container::npy::{NpyArray, NpyElement, read_npy};

fn zarr_err(path: &Path, context: &str, err: impl std::fmt::Display) -> DspError {
    DspError::Io(format!("zarr {} ({context}): {err}", path.display()))
}

/// Opens the Zarr v3 filesystem store at `dir`.
pub fn open_store(dir: &Path) -> DspResult<ReadableWritableListableStorage> {
    let store = FilesystemStore::new(dir).map_err(|e| zarr_err(dir, "open store", e))?;
    Ok(Arc::new(store))
}

/// Opens a read-write Zarr v3 filesystem store at `dir`, creating directories if needed.
pub fn open_rw_store(dir: &Path) -> DspResult<ReadableWritableListableStorage> {
    std::fs::create_dir_all(dir).map_err(|e| zarr_err(dir, "mkdir", e))?;
    open_store(dir)
}

/// Every element of the array at `path` in `store`; `None` when it is missing or unreadable as `T`.
pub fn read_all<T: zarrs::array::ElementOwned>(store: &Arc<dyn ReadableStorageTraits>, path: &str) -> Option<Vec<T>> {
    let a = Array::open(store.clone(), path).ok()?;
    a.retrieve_array_subset::<Vec<T>>(&a.subset_all()).ok()
}

/// Writes a Zarr v3 group at `node_path` (e.g. `"/"`, `"/units"`, `"/spikes"`) with `attributes`.
pub fn write_group(
    store: &ReadableWritableListableStorage,
    dir: &Path,
    node_path: &str,
    attributes: Map<String, Value>,
) -> DspResult<()> {
    let mut group = GroupBuilder::new()
        .build(store.clone(), node_path)
        .map_err(|e| zarr_err(dir, node_path, e))?;
    *group.attributes_mut() = attributes;
    group
        .store_metadata()
        .map_err(|e| zarr_err(dir, node_path, e))
}

/// Reads the `zarr.json` document of `node_path` (e.g. `""` or `"/units/spike_times"`).
pub fn read_node_json(dir: &Path, node_path: &str) -> Option<Value> {
    let rel = node_path.trim_start_matches('/');
    let json_path = if rel.is_empty() {
        dir.join("zarr.json")
    } else {
        dir.join(rel).join("zarr.json")
    };
    let text = std::fs::read_to_string(json_path).ok()?;
    serde_json::from_str(&text).ok()
}

/// Reads the `attributes` object from `<node_path>/zarr.json`.
pub fn read_node_attributes(dir: &Path, node_path: &str) -> Option<Map<String, Value>> {
    read_node_json(dir, node_path)?
        .get("attributes")?
        .as_object()
        .cloned()
}

/// Returns `true` if `node_path` exists in `dir` as either a Zarr v3 array (`<node>/zarr.json`)
/// or a legacy `.npy` file (`<node>.npy`).
pub fn has_array(dir: &Path, node_path: &str) -> bool {
    let rel = node_path.trim_start_matches('/');
    dir.join(rel).join("zarr.json").is_file() || dir.join(format!("{rel}.npy")).is_file()
}

#[allow(clippy::too_many_arguments)]
fn build_and_store<T: zarrs::array::Element + Copy>(
    store: &ReadableWritableListableStorage,
    dir: &Path,
    node_path: &str,
    shape: &[usize],
    dtype: DataType,
    fill: zarrs::array::FillValue,
    dtype_attr: &str,
    dims: &[&str],
    mut attrs: Map<String, Value>,
    data: &[T],
) -> DspResult<()> {
    let expected: usize = shape.iter().product();
    if expected != data.len() {
        return Err(DspError::InvalidConfig(format!(
            "{}:{node_path}: {} values do not match shape {shape:?}",
            dir.display(),
            data.len()
        )));
    }
    attrs.insert("_DTYPE".into(), json!(dtype_attr));
    if !dims.is_empty() {
        attrs.insert("_ARRAY_DIMENSIONS".into(), json!(dims));
    }
    let z_shape: Vec<u64> = shape.iter().map(|&s| s as u64).collect();
    let z_chunks: Vec<u64> = shape.iter().map(|&s| (s as u64).max(1)).collect();
    let mut builder = ArrayBuilder::new(z_shape, z_chunks, dtype, fill);
    builder.attributes(attrs);
    if !dims.is_empty() {
        builder.dimension_names(dims.iter().copied().map(Some).collect::<Vec<_>>().into());
    }
    let array = builder
        .build(store.clone(), node_path)
        .map_err(|e| zarr_err(dir, node_path, e))?;
    array
        .store_metadata()
        .map_err(|e| zarr_err(dir, node_path, e))?;
    if !data.is_empty() {
        array
            .store_array_subset(&array.subset_all(), data)
            .map_err(|e| zarr_err(dir, node_path, e))?;
    }
    Ok(())
}

pub fn write_array_u64(
    store: &ReadableWritableListableStorage,
    dir: &Path,
    node_path: &str,
    data: &[u64],
    shape: &[usize],
    dims: &[&str],
    attrs: Map<String, Value>,
) -> DspResult<()> {
    build_and_store(store, dir, node_path, shape, data_type::uint64(), zarrs::array::FillValue::from(0u64), "uint64", dims, attrs, data)
}

pub fn write_array_i64(
    store: &ReadableWritableListableStorage,
    dir: &Path,
    node_path: &str,
    data: &[i64],
    shape: &[usize],
    dims: &[&str],
    attrs: Map<String, Value>,
) -> DspResult<()> {
    build_and_store(store, dir, node_path, shape, data_type::int64(), zarrs::array::FillValue::from(0i64), "int64", dims, attrs, data)
}

pub fn write_array_i32(
    store: &ReadableWritableListableStorage,
    dir: &Path,
    node_path: &str,
    data: &[i32],
    shape: &[usize],
    dims: &[&str],
    attrs: Map<String, Value>,
) -> DspResult<()> {
    build_and_store(store, dir, node_path, shape, data_type::int32(), zarrs::array::FillValue::from(0i32), "int32", dims, attrs, data)
}

pub fn write_array_f32(
    store: &ReadableWritableListableStorage,
    dir: &Path,
    node_path: &str,
    data: &[f32],
    shape: &[usize],
    dims: &[&str],
    attrs: Map<String, Value>,
) -> DspResult<()> {
    build_and_store(store, dir, node_path, shape, data_type::float32(), zarrs::array::FillValue::from(0.0f32), "float32", dims, attrs, data)
}

pub fn write_array_f64(
    store: &ReadableWritableListableStorage,
    dir: &Path,
    node_path: &str,
    data: &[f64],
    shape: &[usize],
    dims: &[&str],
    attrs: Map<String, Value>,
) -> DspResult<()> {
    build_and_store(store, dir, node_path, shape, data_type::float64(), zarrs::array::FillValue::from(0.0f64), "float64", dims, attrs, data)
}

/// Reads a Zarr v3 array at `node_path` (or legacy `<node_path>.npy`), converting elements to `T`.
pub fn read_array<T: NpyElement>(dir: &Path, node_path: &str) -> DspResult<NpyArray<T>> {
    let rel = node_path.trim_start_matches('/');
    let zarr_meta_path = dir.join(rel).join("zarr.json");
    if zarr_meta_path.is_file() {
        let meta = read_node_json(dir, node_path).ok_or_else(|| {
            DspError::UnsupportedFormat(format!("{}:{node_path}: invalid zarr.json", dir.display()))
        })?;
        let dtype_str = meta
            .get("data_type")
            .and_then(Value::as_str)
            .unwrap_or("float32");
        let store: Arc<dyn ReadableStorageTraits> =
            Arc::new(FilesystemStore::new(dir).map_err(|e| zarr_err(dir, node_path, e))?);
        let z_path = if node_path.starts_with('/') {
            node_path.to_string()
        } else {
            format!("/{node_path}")
        };
        let array = Array::open(store, &z_path).map_err(|e| zarr_err(dir, &z_path, e))?;
        let shape: Vec<usize> = array.shape().iter().map(|&d| d as usize).collect();
        let total: usize = shape.iter().product();
        if total == 0 {
            return Ok(NpyArray { data: Vec::new(), shape });
        }
        let sub = array.subset_all();
        let rd = |e: zarrs::array::ArrayError| zarr_err(dir, &z_path, e);
        let data: Vec<T> = match dtype_str {
            "uint8" | "bool" => array.retrieve_array_subset::<Vec<u8>>(&sub).map_err(rd)?.into_iter().map(|v| T::from_u64(v as u64)).collect(),
            "uint16" => array.retrieve_array_subset::<Vec<u16>>(&sub).map_err(rd)?.into_iter().map(|v| T::from_u64(v as u64)).collect(),
            "uint32" => array.retrieve_array_subset::<Vec<u32>>(&sub).map_err(rd)?.into_iter().map(|v| T::from_u64(v as u64)).collect(),
            "uint64" => array.retrieve_array_subset::<Vec<u64>>(&sub).map_err(rd)?.into_iter().map(T::from_u64).collect(),
            "int8" => array.retrieve_array_subset::<Vec<i8>>(&sub).map_err(rd)?.into_iter().map(|v| T::from_i64(v as i64)).collect(),
            "int16" => array.retrieve_array_subset::<Vec<i16>>(&sub).map_err(rd)?.into_iter().map(|v| T::from_i64(v as i64)).collect(),
            "int32" => array.retrieve_array_subset::<Vec<i32>>(&sub).map_err(rd)?.into_iter().map(|v| T::from_i64(v as i64)).collect(),
            "int64" => array.retrieve_array_subset::<Vec<i64>>(&sub).map_err(rd)?.into_iter().map(T::from_i64).collect(),
            "float32" => array.retrieve_array_subset::<Vec<f32>>(&sub).map_err(rd)?.into_iter().map(|v| T::from_f64(v as f64)).collect(),
            "float64" => array.retrieve_array_subset::<Vec<f64>>(&sub).map_err(rd)?.into_iter().map(T::from_f64).collect(),
            other => {
                return Err(DspError::UnsupportedFormat(format!(
                    "{}:{z_path}: unsupported Zarr data_type '{other}'",
                    dir.display()
                )));
            }
        };
        return Ok(NpyArray { data, shape });
    }

    let npy_path = dir.join(format!("{rel}.npy"));
    if npy_path.is_file() {
        return read_npy::<T>(&npy_path);
    }

    Err(DspError::UnsupportedFormat(format!(
        "{}: missing Zarr array or .npy file for '{node_path}'",
        dir.display()
    )))
}

/// Reads an optional array from `dir` at `node_path`.
pub fn read_optional_array<T: NpyElement>(dir: &Path, node_path: &str) -> Option<NpyArray<T>> {
    has_array(dir, node_path).then(|| read_array::<T>(dir, node_path).ok()).flatten()
}
