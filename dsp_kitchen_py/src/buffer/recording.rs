use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use pyo3::exceptions::{PyIOError, PyIndexError, PyValueError};
use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pyfunction, gen_stub_pymethods};
use pyo3::types::{PySlice, PyTuple};

use crate::array::{to_numpy, F32Array};

use dsp_core::{MemoryRecording, RecordingSource, SlicedRecording};

/// Seconds `read` returns when no end is given.
const DEFAULT_READ_SEC: f64 = 1.0;

/// Reads `[start_sample, end_sample)` (default: [`DEFAULT_READ_SEC`] at the recording rate) of
/// `channels` (default: all) as a `float32` `[channels, samples]` NumPy array in each channel's
/// unit, with the GIL released during the read.
pub(crate) fn read_to_numpy<'py>(
    py: Python<'py>,
    source: &dyn RecordingSource,
    start_sample: u64,
    end_sample: Option<u64>,
    channels: Option<Vec<usize>>,
) -> PyResult<Bound<'py, PyAny>> {
    let info = source.info();
    let (total_samples, total_channels) = (info.samples, info.channel_count());
    let default_len = (info.sample_rate_hz() * DEFAULT_READ_SEC).round().max(1.0) as u64;
    let end = end_sample.unwrap_or_else(|| start_sample.saturating_add(default_len).min(total_samples));
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

/// The signals a recording file holds (a file can hold several, e.g. an NWB file with an HD-EMG
/// series and an accelerometer).
///
/// Parameters
/// ----------
/// path : str
///
/// Returns
/// -------
/// list of dict
///     `id` (pass it as `Recording(path, source=id)`), `name`, `kind` (`"electrical"` / `"other"`),
///     `channels`, `samples`, `sample_rate`, `unit`.
#[gen_stub_pyfunction]
#[pyfunction]
pub fn list_sources<'py>(py: Python<'py>, path: &str) -> PyResult<Vec<Bound<'py, pyo3::types::PyDict>>> {
    let entries = dsp_io::sources(Path::new(path)).map_err(|e| PyIOError::new_err(format!("{path}: {e}")))?;
    entries
        .into_iter()
        .map(|e| {
            let d = pyo3::types::PyDict::new(py);
            d.set_item("id", e.id)?;
            d.set_item("name", e.name)?;
            d.set_item("kind", if e.kind == dsp_io::SourceKind::Electrical { "electrical" } else { "other" })?;
            d.set_item("channels", e.channels)?;
            d.set_item("samples", e.samples)?;
            d.set_item("sample_rate", e.sample_rate.rate_hz())?;
            d.set_item("unit", e.unit.symbol())?;
            Ok(d)
        })
        .collect()
}

/// A recording in any format dsp-io reads (SpikeGLX, NWB, Zarr, raw binary, mtscomp, …).
///
/// Nothing is loaded when it opens: `read` and `read_window` read the samples asked for, and
/// slicing (`slice_samples`, `slice_time`, `rec[...]`) makes lazy views. Values are `float32` in each
/// channel's unit (see `units`).
///
/// Parameters
/// ----------
/// path : str
///     File or folder of the recording.
/// source : str, optional
///     Which signal of a multi-signal file (an `id` from `list_sources`); default: the main one.
///
/// Examples
/// --------
/// >>> from dsp_kitchen.io import Recording
/// >>> rec = Recording("session.nwb.zarr")
/// >>> rec.channels, rec.sample_rate, rec.duration_sec
/// >>> x = rec.read_window(10.0, 2.0)                # [channels, samples] float32, 2 s from t = 10 s
/// >>> first_minute = rec.slice_time(0.0, 60.0)      # lazy view, nothing read
#[gen_stub_pyclass]
#[pyclass(name = "Recording", skip_from_py_object)]
pub struct PyRecording {
    path: PathBuf,
    pub(crate) inner: Arc<dyn RecordingSource>,
}

impl PyRecording {
    pub(crate) fn from_source(path: PathBuf, inner: Arc<dyn RecordingSource>) -> Self {
        Self { path, inner }
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyRecording {
    /// Opens `path` (see the class docs).
    #[new]
    #[pyo3(signature = (path, source=None))]
    pub fn new(path: &str, source: Option<&str>) -> PyResult<Self> {
        let p = PathBuf::from(path);
        let opened = match source {
            Some(id) => dsp_io::open_source(&p, id),
            None => dsp_io::open(&p),
        };
        let inner: Arc<dyn RecordingSource> = Arc::from(opened.map_err(|e| PyIOError::new_err(format!("{path}: {e}")))?);
        Ok(Self { path: p, inner })
    }

    /// An in-memory recording of an array (copied; values unitless). For tests and derived signals;
    /// files open with `Recording(path)`.
    ///
    /// Parameters
    /// ----------
    /// data : numpy.ndarray
    ///     `[channels, samples]`, converted to float32.
    /// fs : float
    ///     Sampling rate, Hz.
    /// name : str, default "array"
    ///     Name shown in `repr` and kept in sorting outputs.
    #[staticmethod]
    #[pyo3(signature = (data, fs, *, name="array"))]
    pub fn from_array(data: Bound<'_, PyAny>, fs: f64, name: &str) -> PyResult<Self> {
        let input = F32Array::new(&data)?;
        let (channels, _) = input.channels_samples(None)?;
        let rec = MemoryRecording::new(name, input.slice().to_vec(), channels, fs).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { path: PathBuf::new(), inner: Arc::new(rec) })
    }

    /// File or folder the recording was opened from.
    #[getter]
    pub fn path(&self) -> String {
        self.path.to_string_lossy().into_owned()
    }

    /// Name of the recording (file name, and source when not the main one).
    #[getter]
    pub fn name(&self) -> &str {
        &self.inner.info().name
    }

    /// Number of channels.
    #[getter]
    pub fn channels(&self) -> usize {
        self.inner.info().channel_count()
    }

    /// Number of samples per channel.
    #[getter]
    pub fn samples(&self) -> u64 {
        self.inner.info().samples
    }

    /// Sampling rate, Hz.
    #[getter]
    pub fn sample_rate(&self) -> f64 {
        self.inner.info().sample_rate_hz()
    }

    /// Duration, s (`samples / sample_rate`).
    #[getter]
    pub fn duration_sec(&self) -> f64 {
        self.inner.info().duration_sec()
    }

    /// Time of sample 0 from the acquisition start (s).
    #[getter]
    pub fn start_time_sec(&self) -> f64 {
        self.inner.info().start_time.as_seconds_f64()
    }

    /// `(channels, samples)`.
    #[getter]
    pub fn shape(&self) -> (usize, u64) {
        (self.inner.info().channel_count(), self.inner.info().samples)
    }

    /// Each channel's name, as stored in the file.
    #[getter]
    pub fn channel_names(&self) -> Vec<String> {
        self.inner
            .info()
            .channels
            .iter()
            .map(|c| c.name.clone())
            .collect()
    }

    /// Each channel's unit symbol (`"µV"`, `"mV"`, …; empty for dimensionless values).
    #[getter]
    pub fn units(&self) -> Vec<String> {
        self.inner.info().channels.iter().map(|c| c.unit.symbol().to_string()).collect()
    }

    /// Format-specific metadata (key → value, as stored).
    #[getter]
    pub fn metadata(&self) -> HashMap<String, String> {
        self.inner
            .info()
            .metadata
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect()
    }

    /// A lazy view (nothing read) over samples `[start_sample, end_sample)` and some channels.
    ///
    /// Parameters
    /// ----------
    /// start_sample : int, default 0
    /// end_sample : int, optional
    ///     Exclusive; default: the end of the recording.
    /// channels : list of int, optional
    ///     Channel indices, in the order wanted; default: all.
    ///
    /// Returns
    /// -------
    /// Recording
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

    /// A lazy view (nothing read) over a time window and some channels.
    ///
    /// Parameters
    /// ----------
    /// start_sec : float, default 0.0
    ///     Start, s from the first sample.
    /// end_sec : float, optional
    ///     End, s; or give `duration_sec`. Default: the end of the recording.
    /// duration_sec : float, optional
    ///     Length of the window, s.
    /// channels : list of int, optional
    ///     Channel indices; default: all.
    ///
    /// Returns
    /// -------
    /// Recording
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

    /// A lazy view: `rec[start:end]` selects samples, `rec[channels, start:end]` channels and samples.
    ///
    /// Nothing is read from disk until the view is read.
    ///
    /// Parameters
    /// ----------
    /// key : slice or tuple
    ///     A sample slice, or `(channels, sample slice)` with `channels` an int, slice or list.
    ///
    /// Returns
    /// -------
    /// Recording
    ///     The view.
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

        if let Ok(tup) = key.cast::<PyTuple>()
            && tup.len() == 2
        {
            {
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

    /// Reads samples `[start_sample, end_sample)` of some channels.
    ///
    /// Parameters
    /// ----------
    /// start_sample : int, default 0
    /// end_sample : int, optional
    ///     Exclusive; default: one second after `start_sample`.
    /// channels : list of int, optional
    ///     Channel indices; default: all.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray
    ///     `[channels, samples]` float32, in each channel's unit.
    #[pyo3(signature = (start_sample=0, end_sample=None, channels=None))]
    #[gen_stub(override_return_type(type_repr = "numpy.typing.NDArray[numpy.float32]", imports = ("numpy", "numpy.typing")))]
    pub fn read<'py>(
        &self,
        py: Python<'py>,
        start_sample: u64,
        end_sample: Option<u64>,
        channels: Option<Vec<usize>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        read_to_numpy(py, self.inner.as_ref(), start_sample, end_sample, channels)
    }

    /// Reads a time window.
    ///
    /// Parameters
    /// ----------
    /// start_sec : float
    ///     Start, s of acquisition time (`start_time_sec` is the time of sample 0).
    /// duration_sec : float
    ///     Length, s.
    /// channels : list of int, optional
    ///     Channel indices; default: all.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray
    ///     `[channels, samples]` float32, in each channel's unit.
    #[pyo3(signature = (start_sec, duration_sec, channels=None))]
    #[gen_stub(override_return_type(type_repr = "numpy.typing.NDArray[numpy.float32]", imports = ("numpy", "numpy.typing")))]
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
            "Recording(name='{}', shape=({}, {}), sample_rate={:.2} Hz, duration={:.2} s, start_time={:.2} s, units={:?})",
            self.name(),
            self.channels(),
            self.samples(),
            self.sample_rate(),
            self.duration_sec(),
            self.start_time_sec(),
            self.units().first()
        )
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyRecording>()?;
    m.add_function(wrap_pyfunction!(list_sources, m)?)?;
    Ok(())
}
