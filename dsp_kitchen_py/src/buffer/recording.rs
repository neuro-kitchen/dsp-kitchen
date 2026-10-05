use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use pyo3::exceptions::{PyIOError, PyIndexError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PySlice, PyTuple};

use crate::array::to_numpy;

use dsp_core::{RecordingSource, SlicedRecording};
use dsp_io::nwb::{list_series, NwbZarrRecording};

/// Reads `[start_sample, end_sample)` (default: 1 s at the recording rate) of `channels` (default:
/// all) as a µV `float32` `[channels, samples]` NumPy array, with the GIL released during the read.
pub(crate) fn read_to_numpy<'py>(
    py: Python<'py>,
    source: &dyn RecordingSource,
    start_sample: u64,
    end_sample: Option<u64>,
    channels: Option<Vec<usize>>,
) -> PyResult<Bound<'py, PyAny>> {
    let info = source.info();
    let (total_samples, total_channels) = (info.samples, info.channel_count());
    let one_second = info.sample_rate_hz().round().max(1.0) as u64;
    let end = end_sample.unwrap_or_else(|| start_sample.saturating_add(one_second).min(total_samples));
    if start_sample > end || end > total_samples {
        return Err(PyValueError::new_err(format!(
            "sample range {start_sample}..{end} is outside 0..{total_samples}"
        )));
    }
    let ch_indices: Vec<usize> = channels.unwrap_or_else(|| (0..total_channels).collect());
    if let Some(&bad) = ch_indices.iter().find(|&&c| c >= total_channels) {
        return Err(PyIndexError::new_err(format!(
            "channel {bad} out of range for a recording with {total_channels} channels"
        )));
    }
    let n_ch = ch_indices.len();
    let n_samp = (end - start_sample) as usize;
    let buf = py.detach(|| {
        let mut buf = vec![0.0f32; n_ch * n_samp];
        source.read(&ch_indices, start_sample..end, &mut buf).map(|_| buf)
    });
    let buf = buf.map_err(|e| PyIOError::new_err(format!("read error: {e}")))?;
    to_numpy(py, buf, &[n_ch, n_samp])
}

/// Lists continuous series (`ElectricalSeries` and `TimeSeries` with regular rate)
/// inside an NWB Zarr v3 store (`/acquisition/*`).
#[pyfunction]
pub fn list_nwb_series<'py>(
    py: Python<'py>,
    path: &str,
) -> PyResult<Vec<Bound<'py, pyo3::types::PyDict>>> {
    let entries = list_series(Path::new(path));
    let mut out = Vec::with_capacity(entries.len());
    for entry in entries {
        let d = pyo3::types::PyDict::new(py);
        d.set_item("path", entry.path)?;
        d.set_item("neurodata_type", entry.neurodata_type)?;
        d.set_item("channels", entry.channels)?;
        d.set_item("samples", entry.samples)?;
        out.push(d);
    }
    Ok(out)
}

/// Chunked reader for NWB Zarr v3 (`.nwb.zarr`) and general `dsp-io` recordings.
#[pyclass(name = "NwbZarrRecording", skip_from_py_object)]
pub struct PyNwbZarrRecording {
    path: PathBuf,
    pub(crate) inner: Arc<dyn RecordingSource>,
}

#[pymethods]
impl PyNwbZarrRecording {
    #[new]
    #[pyo3(signature = (path, series=None))]
    pub fn new(path: &str, series: Option<&str>) -> PyResult<Self> {
        let p = PathBuf::from(path);
        let inner: Arc<dyn RecordingSource> = if let Some(s) = series {
            let norm = if s.starts_with('/') {
                s.to_string()
            } else {
                format!("/acquisition/{s}")
            };
            Arc::new(NwbZarrRecording::open_series(&p, &norm).map_err(|e| {
                pyo3::exceptions::PyIOError::new_err(format!(
                    "Failed to open NWB series '{}' in '{}': {}",
                    norm, path, e
                ))
            })?)
        } else {
            Arc::from(dsp_io::open(&p).map_err(|e| {
                pyo3::exceptions::PyIOError::new_err(format!(
                    "Failed to open recording '{}': {}",
                    path, e
                ))
            })?)
        };
        Ok(Self { path: p, inner })
    }

    #[getter]
    pub fn path(&self) -> String {
        self.path.to_string_lossy().into_owned()
    }

    #[getter]
    pub fn name(&self) -> &str {
        &self.inner.info().name
    }

    #[getter]
    pub fn channels(&self) -> usize {
        self.inner.info().channel_count()
    }

    #[getter]
    pub fn samples(&self) -> u64 {
        self.inner.info().samples
    }

    #[getter]
    pub fn sample_rate(&self) -> f64 {
        self.inner.info().sample_rate_hz()
    }

    #[getter]
    pub fn duration_sec(&self) -> f64 {
        self.inner.info().duration_sec()
    }

    #[getter]
    pub fn start_time_sec(&self) -> f64 {
        self.inner.info().start_time_sec
    }

    #[getter]
    pub fn shape(&self) -> (usize, u64) {
        (self.inner.info().channel_count(), self.inner.info().samples)
    }

    #[getter]
    pub fn channel_names(&self) -> Vec<String> {
        self.inner
            .info()
            .channels
            .iter()
            .map(|c| c.name.clone())
            .collect()
    }

    #[getter]
    pub fn unit(&self) -> String {
        self.inner
            .info()
            .metadata
            .get("unit")
            .cloned()
            .unwrap_or_else(|| "uV".to_string())
    }

    #[getter]
    pub fn series(&self) -> Option<String> {
        self.inner.info().metadata.get("nwb_series").cloned()
    }

    #[getter]
    pub fn metadata(&self) -> HashMap<String, String> {
        self.inner
            .info()
            .metadata
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect()
    }

    /// Creates a lazy zero-load `NwbZarrRecording` view over `[start_sample..end_sample)` and `channels`.
    #[pyo3(signature = (start_sample=0, end_sample=None, channels=None))]
    pub fn slice_samples(
        &self,
        start_sample: u64,
        end_sample: Option<u64>,
        channels: Option<Vec<usize>>,
    ) -> PyResult<Self> {
        let end = end_sample.unwrap_or(self.samples());
        let sliced = SlicedRecording::new(self.inner.clone(), start_sample..end, channels)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Slice error: {e}")))?;
        Ok(Self {
            path: self.path.clone(),
            inner: Arc::new(sliced),
        })
    }

    /// Creates a lazy zero-load `NwbZarrRecording` view over a time window (`start_sec` to `end_sec`
    /// or `start_sec + duration_sec`) and optional `channels`.
    #[pyo3(signature = (start_sec=0.0, end_sec=None, duration_sec=None, channels=None))]
    pub fn slice_time(
        &self,
        start_sec: f64,
        end_sec: Option<f64>,
        duration_sec: Option<f64>,
        channels: Option<Vec<usize>>,
    ) -> PyResult<Self> {
        let fs = self.sample_rate();
        let total = self.samples();
        let s0 = (start_sec.max(0.0) * fs).round() as u64;
        let s1 = if let Some(dur) = duration_sec {
            s0.saturating_add((dur.max(0.0) * fs).round() as u64).min(total)
        } else if let Some(end_s) = end_sec {
            ((end_s.max(0.0) * fs).round() as u64).min(total)
        } else {
            total
        };
        self.slice_samples(s0.min(total), Some(s1), channels)
    }

    /// Supports lazy zero-load Python indexing:
    /// - `rec[start_sample:end_sample]` -> slices samples across all channels
    /// - `rec[ch_sel, sample_slice]` -> slices channels and samples without reading from disk
    pub fn __getitem__(&self, key: &Bound<'_, PyAny>) -> PyResult<Self> {
        let total_samples = self.samples() as isize;
        let total_channels = self.channels() as isize;

        if let Ok(s) = key.cast::<PySlice>() {
            let indices = s.indices(total_samples)?;
            if indices.step != 1 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "Sample slice step must be 1 for lazy recording slicing",
                ));
            }
            let start = indices.start.max(0) as u64;
            let stop = indices.stop.max(indices.start).max(0) as u64;
            return self.slice_samples(start, Some(stop), None);
        }

        if let Ok(tup) = key.cast::<PyTuple>() {
            if tup.len() == 2 {
                let ch_item = tup.get_item(0)?;
                let samp_item = tup.get_item(1)?;

                let channels: Option<Vec<usize>> = if let Ok(ch_slice) = ch_item.cast::<PySlice>() {
                    let ci = ch_slice.indices(total_channels)?;
                    let mut v = Vec::new();
                    let mut i = ci.start;
                    if ci.step > 0 {
                        while i < ci.stop {
                            v.push(i as usize);
                            i += ci.step;
                        }
                    }
                    Some(v)
                } else if let Ok(ch_list) = ch_item.extract::<Vec<usize>>() {
                    Some(ch_list)
                } else if let Ok(single_ch) = ch_item.extract::<usize>() {
                    Some(vec![single_ch])
                } else {
                    return Err(pyo3::exceptions::PyTypeError::new_err(
                        "First index must be channel slice, list of channel indices, or int",
                    ));
                };

                if let Ok(samp_slice) = samp_item.cast::<PySlice>() {
                    let si = samp_slice.indices(total_samples)?;
                    if si.step != 1 {
                        return Err(pyo3::exceptions::PyValueError::new_err(
                            "Sample slice step must be 1 for lazy recording slicing",
                        ));
                    }
                    let start = si.start.max(0) as u64;
                    let stop = si.stop.max(si.start).max(0) as u64;
                    return self.slice_samples(start, Some(stop), channels);
                }
            }
        }

        Err(pyo3::exceptions::PyTypeError::new_err(
            "Use rec[start:stop] or rec[ch_slice, start:stop] for lazy zero-load recording slicing",
        ))
    }

    /// Reads µV `float32` `[channels, samples]` for `[start_sample, end_sample)` across `channels`
    /// (default: all channels, 1 s from `start_sample`).
    #[pyo3(signature = (start_sample=0, end_sample=None, channels=None))]
    pub fn read<'py>(
        &self,
        py: Python<'py>,
        start_sample: u64,
        end_sample: Option<u64>,
        channels: Option<Vec<usize>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        read_to_numpy(py, self.inner.as_ref(), start_sample, end_sample, channels)
    }

    /// Reads a time window `[start_sec, start_sec + duration_sec]` and returns `(time_sec_array, data_2d_array)`.
    #[pyo3(signature = (start_sec=0.0, duration_sec=0.1, channels=None))]
    pub fn read_window<'py>(
        &self,
        py: Python<'py>,
        start_sec: f64,
        duration_sec: f64,
        channels: Option<Vec<usize>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let fs = self.sample_rate();
        let s0 = ((start_sec - self.start_time_sec()).max(0.0) * fs).round() as u64;
        let n = (duration_sec.max(0.0) * fs).round() as u64;
        let s1 = (s0 + n).min(self.samples());
        self.read(py, s0, Some(s1), channels)
    }

    fn __repr__(&self) -> String {
        format!(
            "NwbZarrRecording(name='{}', shape=({}, {}), sample_rate={:.2}Hz, duration={:.2}s, start_time={:.2}s, unit='{}')",
            self.name(),
            self.channels(),
            self.samples(),
            self.sample_rate(),
            self.duration_sec(),
            self.start_time_sec(),
            self.unit()
        )
    }
}
