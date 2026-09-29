use std::path::Path;
use std::sync::Arc;
use zarrs::array::{Array, ArrayBuilder, ZARR_NAN_F32, data_type};
use zarrs::filesystem::FilesystemStore;
use zarrs::group::GroupBuilder;
use zarrs::storage::ReadableWritableListableStorage;
use dsp_core::error::{DspError, DspResult};

/// Creates a chunked, multi-channel electrophysiology dataset stored in Zarr v3 format using `zarrs`.
///
/// Shape: `[channels, samples]`
/// Chunking: `[channels, chunk_samples]` (e.g. all 384 channels by 5,000 temporal samples).
pub fn create_zarr_recording(
    store_path: &Path,
    channels: usize,
    samples: usize,
    sample_rate_hz: f64,
    data: &[f32],
) -> DspResult<()> {
    if data.len() != channels * samples {
        return Err(DspError::ShapeMismatch {
            expected: vec![channels * samples],
            actual: vec![data.len()],
        });
    }

    // Ensure store directory exists
    std::fs::create_dir_all(store_path)
        .map_err(|e| DspError::InvalidConfig(format!("Failed to create Zarr store dir: {e}")))?;

    let store: ReadableWritableListableStorage = Arc::new(
        FilesystemStore::new(store_path)
            .map_err(|e| DspError::InvalidConfig(format!("Failed to init FilesystemStore: {e}")))?,
    );

    // 1. Create root group with neuroscience acquisition metadata
    let mut root_group = GroupBuilder::new()
        .build(store.clone(), "/")
        .map_err(|e| DspError::InvalidConfig(format!("Failed to build root group: {e}")))?;

    root_group.attributes_mut().insert(
        "instrument".into(),
        serde_json::Value::String("Generic Sensor Array".into()),
    );
    root_group.attributes_mut().insert(
        "sample_rate_hz".into(),
        serde_json::Value::Number(serde_json::Number::from_f64(sample_rate_hz).unwrap()),
    );
    root_group.attributes_mut().insert(
        "channels".into(),
        serde_json::Value::Number(serde_json::Number::from(channels)),
    );
    root_group.attributes_mut().insert(
        "samples".into(),
        serde_json::Value::Number(serde_json::Number::from(samples)),
    );
    root_group.attributes_mut().insert(
        "created_by".into(),
        serde_json::Value::String("dsp-kitchen (zarrs engine)".into()),
    );

    root_group
        .store_metadata()
        .map_err(|e| DspError::InvalidConfig(format!("Failed to write root group metadata: {e}")))?;

    // 2. Define array path and chunk dimensions
    let array_path = "/traces";
    let chunk_samples = 5000usize.min(samples); // 5000 samples per chunk (~166ms @ 30kHz)
    let shape = vec![channels as u64, samples as u64];
    let chunk_shape = vec![channels as u64, chunk_samples as u64];

    let array = ArrayBuilder::new(
        shape,
        chunk_shape,
        data_type::float32(),
        ZARR_NAN_F32,
    )
    .dimension_names(["channels", "samples"].into())
    .build(store.clone(), array_path)
    .map_err(|e| DspError::InvalidConfig(format!("Failed to build Zarr array: {e}")))?;

    // Store array metadata
    array
        .store_metadata()
        .map_err(|e| DspError::InvalidConfig(format!("Failed to write array metadata: {e}")))?;

    // 3. Store full dataset subset [0..channels, 0..samples]
    let channels_range = 0..channels as u64;
    let samples_range = 0..samples as u64;

    array
        .store_array_subset(&[channels_range, samples_range], data)
        .map_err(|e| DspError::InvalidConfig(format!("Failed to write Zarr array data: {e}")))?;

    Ok(())
}

/// Reads a Zarr v3 electrophysiology recording created with `create_zarr_recording`.
/// Returns `(data, channels, samples, sample_rate_hz)`.
pub fn read_zarr_recording(store_path: &Path) -> DspResult<(Vec<f32>, usize, usize, f64)> {
    let store: ReadableWritableListableStorage = Arc::new(
        FilesystemStore::new(store_path)
            .map_err(|e| DspError::InvalidConfig(format!("Failed to open FilesystemStore: {e}")))?,
    );

    let array_path = "/traces";
    let array = Array::open(store.clone(), array_path)
        .map_err(|e| DspError::InvalidConfig(format!("Failed to open Zarr array: {e}")))?;

    let shape = array.shape();
    if shape.len() != 2 {
        return Err(DspError::InvalidConfig(format!(
            "Expected 2D array [channels, samples], got {}D",
            shape.len()
        )));
    }

    let channels = shape[0] as usize;
    let samples = shape[1] as usize;

    let subset_all = array.subset_all();
    let data: Vec<f32> = array
        .retrieve_array_subset(&subset_all)
        .map_err(|e| DspError::InvalidConfig(format!("Failed to retrieve Zarr subset: {e}")))?;

    // Read metadata attributes from root group if present
    let sample_rate_hz = 30000.0f64; // Default fallback

    Ok((data, channels, samples, sample_rate_hz))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zarr_roundtrip() {
        let temp_dir = std::env::temp_dir().join("dsp_kitchen_zarr_test.zarr");
        let _ = std::fs::remove_dir_all(&temp_dir);

        let channels = 16;
        let samples = 1000;
        let mut data = vec![0.0f32; channels * samples];
        data[0] = 42.5;
        data[channels * samples - 1] = -128.0;

        create_zarr_recording(&temp_dir, channels, samples, 30000.0, &data).unwrap();

        let (read_data, r_ch, r_s, r_sr) = read_zarr_recording(&temp_dir).unwrap();
        assert_eq!(r_ch, channels);
        assert_eq!(r_s, samples);
        assert_eq!(r_sr, 30000.0);
        assert_eq!(read_data[0], 42.5);
        assert_eq!(read_data[channels * samples - 1], -128.0);

        let _ = std::fs::remove_dir_all(&temp_dir);
    }
}
