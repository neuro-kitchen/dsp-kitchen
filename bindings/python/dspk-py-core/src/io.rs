//! `dspk.io`: open recordings and read them into NumPy; write NumPy arrays as raw binary or Zarr.
//!
//! These are the generic formats of `dspk-io` (raw binary with a JSON sidecar, Zarr `/traces`);
//! neural formats (NWB, SpikeGLX, …) come with the synapse bindings.

use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};

use dspk_core::{DspResult, MemoryOrder, RecordingInfo, RecordingSource, SampleFormat, SampleRate, SignalUnit};
use dspk_io::{Format, SourceEntry};
use dspk_core::recording::{check_read, check_read_stored};
use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pyfunction, gen_stub_pymethods};

use crate::error::to_py;

/// How `open`, `sources` and `formats` find a format. The generic ones of `dspk-io` by default; the
/// Synapse bindings replace them with the set that adds the neural formats.
#[derive(Clone, Copy)]
pub struct FormatSet {
    pub open: fn(&Path) -> DspResult<Box<dyn RecordingSource>>,
    pub open_source: fn(&Path, &str) -> DspResult<Box<dyn RecordingSource>>,
    pub sources: fn(&Path) -> DspResult<Vec<SourceEntry>>,
    pub all: fn() -> &'static [&'static dyn Format],
}

static FORMAT_SET: OnceLock<FormatSet> = OnceLock::new();

/// Makes `set` the formats of `dspk.io` (the first call wins).
pub fn use_formats(set: FormatSet) {
    let _ = FORMAT_SET.set(set);
}

fn format_set() -> &'static FormatSet {
    FORMAT_SET.get_or_init(|| FormatSet { open: dspk_io::open, open_source: dspk_io::open_source, sources: dspk_io::sources, all: dspk_io::formats })
}

/// Samples per channel each writer step reads from the source (bounds memory while writing).
pub const WRITE_CHUNK_SAMPLES: usize = 65_536;

/// Samples per Zarr chunk of [`write_zarr`] (dspk-io's default).
pub const ZARR_CHUNK_SAMPLES: usize = dspk_io::generic::zarr_traces::DEFAULT_CHUNK_SAMPLES;

/// A recording opened from a file, read in pieces.
///
/// Opening reads only the metadata; `read` returns the samples you ask for, scaled to each
/// channel's unit, as float32. Make one with `open`.
///
/// Examples
/// --------
/// >>> import dspk
/// >>> rec = dspk.io.open("session.zarr")  # doctest: +SKIP
/// >>> rec.channel_count, rec.sample_rate  # doctest: +SKIP
/// (32, 30000.0)
/// >>> first_second = rec.read(start=0, stop=30_000)  # doctest: +SKIP
#[gen_stub_pyclass]
#[pyclass(module = "dspk._native.io", frozen)]
pub struct Recording {
    inner: Arc<dyn RecordingSource>,
}

impl Recording {
    /// The recording, for other binding crates (synapse's detectors read it in place).
    pub fn source(&self) -> Arc<dyn RecordingSource> {
        self.inner.clone()
    }

    /// A recording over any source (a recording on a server: `dspk.remote`).
    pub fn from_source(inner: Arc<dyn RecordingSource>) -> Self {
        Self { inner }
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl Recording {
    /// Name of the recording, from the file or its source.
    #[getter]
    fn name(&self) -> &str {
        &self.inner.info().name
    }

    /// Number of channels.
    #[getter]
    fn channel_count(&self) -> usize {
        self.inner.info().channel_count()
    }

    /// Samples per channel.
    #[getter]
    fn samples(&self) -> u64 {
        self.inner.info().samples
    }

    /// Sampling rate in Hz.
    #[getter]
    fn sample_rate(&self) -> f64 {
        self.inner.info().sample_rate_hz()
    }

    /// Duration in seconds.
    #[getter]
    fn duration(&self) -> f64 {
        self.inner.info().duration_sec()
    }

    /// Stored sample type (`"int16"`, `"float32"`, …).
    #[getter]
    fn format(&self) -> &'static str {
        self.inner.info().format.name()
    }

    /// Name of each channel.
    #[getter]
    fn channel_names(&self) -> Vec<String> {
        self.inner.info().channels.iter().map(|c| c.name.clone()).collect()
    }

    /// Unit symbol of each channel's values (`"µV"`, `"mV"`, …; `""` when dimensionless).
    #[getter]
    fn units(&self) -> Vec<String> {
        self.inner.info().channels.iter().map(|c| c.unit.symbol().to_string()).collect()
    }

    /// Reads samples into memory.
    ///
    /// Parameters
    /// ----------
    /// channels : sequence of int, optional
    ///     Channel indices, in the order wanted; by default all.
    /// start : int
    ///     First sample.
    /// stop : int, optional
    ///     The sample after the last; by default the end.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray
    ///     `(len(channels), stop - start)`, float32, in each channel's unit.
    ///
    /// Raises
    /// ------
    /// IndexError
    ///     When a channel or the sample range is outside the recording.
    /// OSError
    ///     When the file cannot be read.
    #[pyo3(signature = (channels=None, start=0, stop=None))]
    fn read<'py>(&self, py: Python<'py>, channels: Option<Vec<usize>>, start: u64, stop: Option<u64>) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let info = self.inner.info();
        let channels = channels.unwrap_or_else(|| (0..info.channel_count()).collect());
        let samples = start..stop.unwrap_or(info.samples);
        let n = samples.end.saturating_sub(samples.start) as usize;
        let source = Arc::clone(&self.inner);
        let values = py
            .detach(move || {
                let mut out = vec![0.0f32; channels.len() * n];
                source.read(&channels, samples, &mut out).map(|()| (channels.len(), out))
            })
            .map_err(to_py)?;
        let (rows, out) = values;
        Ok(Array2::from_shape_vec((rows, n), out).expect("read fills rows × samples").into_pyarray(py))
    }

    fn __repr__(&self) -> String {
        let info = self.inner.info();
        format!(
            "Recording(name={:?}, channels={}, samples={}, sample_rate={} Hz, format={})",
            info.name,
            info.channel_count(),
            info.samples,
            info.sample_rate_hz(),
            info.format.name()
        )
    }
}

/// One source of a file: what `sources` lists and `open(path, source=...)` opens.
///
/// Most files have one; containers such as NWB have several (one per electrical series).
#[gen_stub_pyclass]
#[pyclass(module = "dspk._native.io", frozen, get_all)]
pub struct Source {
    /// Id to pass to `open(path, source=...)`.
    id: String,
    /// Name of the source.
    name: String,
    /// Number of channels.
    channel_count: usize,
    /// Samples per channel.
    samples: u64,
    /// Sampling rate in Hz.
    sample_rate: f64,
    /// Stored sample type (`"int16"`, `"float32"`, …).
    format: &'static str,
    /// Unit symbol of the values (`""` when dimensionless).
    unit: String,
}

#[gen_stub_pymethods]
#[pymethods]
impl Source {
    fn __repr__(&self) -> String {
        format!("Source(id={:?}, channels={}, samples={}, sample_rate={} Hz)", self.id, self.channel_count, self.samples, self.sample_rate)
    }
}

/// Opens a recording file.
///
/// The format is detected from the file (see `formats`). Only metadata is read here; samples are
/// read with `Recording.read`.
///
/// Parameters
/// ----------
/// path : str or os.PathLike
///     The recording: a file, or a folder for stores such as Zarr.
/// source : str, optional
///     Which source to open, an id from `sources(path)`; by default the file's default source.
///
/// Returns
/// -------
/// Recording
///
/// Raises
/// ------
/// OSError
///     When the file cannot be read.
/// ValueError
///     When the format is not recognized or `source` is not in the file.
///
/// Examples
/// --------
/// >>> import dspk
/// >>> rec = dspk.io.open("session.zarr")  # doctest: +SKIP
#[gen_stub_pyfunction(module = "dspk._native.io")]
#[pyfunction]
#[pyo3(signature = (path, source=None))]
pub fn open(path: PathBuf, source: Option<&str>) -> PyResult<Recording> {
    let formats = format_set();
    let inner = match source {
        Some(id) => (formats.open_source)(&path, id),
        None => (formats.open)(&path),
    }
    .map_err(to_py)?;
    Ok(Recording { inner: Arc::from(inner) })
}

/// Lists the sources of a file, reading only its metadata.
///
/// Parameters
/// ----------
/// path : str or os.PathLike
///     The recording file or folder.
///
/// Returns
/// -------
/// list of Source
///
/// Raises
/// ------
/// OSError
///     When the file cannot be read.
/// ValueError
///     When the format is not recognized.
///
/// Examples
/// --------
/// >>> import dspk
/// >>> for s in dspk.io.sources("session.nwb"):  # doctest: +SKIP
/// ...     print(s.id, s.channel_count, s.sample_rate)
#[gen_stub_pyfunction(module = "dspk._native.io")]
#[pyfunction]
pub fn sources(path: PathBuf) -> PyResult<Vec<Source>> {
    Ok((format_set().sources)(&path)
        .map_err(to_py)?
        .into_iter()
        .map(|e| Source {
            id: e.id,
            name: e.name,
            channel_count: e.channels,
            samples: e.samples,
            sample_rate: e.sample_rate.rate_hz(),
            format: e.format.name(),
            unit: e.unit.symbol().to_string(),
        })
        .collect())
}

/// Names of the formats `open` recognizes, in the order they are tried.
///
/// Returns
/// -------
/// list of str
#[gen_stub_pyfunction(module = "dspk._native.io")]
#[pyfunction]
pub fn formats() -> Vec<&'static str> {
    (format_set().all)().iter().map(|f| f.name()).collect()
}

/// A NumPy array `(channels, samples)` seen as a recording, so the Rust writers can read it.
struct ArrayRecording {
    info: RecordingInfo,
    /// Channel-major little-endian values of `info.format`.
    stored: Vec<u8>,
}

/// A NumPy `(channels, samples)` array in a dtype dspk stores, as its format and little-endian
/// bytes (no conversion).
fn stored_array(data: &Bound<'_, PyAny>) -> PyResult<(SampleFormat, usize, usize, Vec<u8>)> {
    macro_rules! try_dtype {
        ($t:ty, $format:expr) => {
            if let Ok(a) = data.extract::<PyReadonlyArray2<'_, $t>>() {
                let view = a.as_array();
                let (c, n) = view.dim();
                return Ok(($format, c, n, view.iter().flat_map(|v| v.to_le_bytes()).collect()));
            }
        };
    }
    try_dtype!(i8, SampleFormat::I8);
    try_dtype!(i16, SampleFormat::I16);
    try_dtype!(u16, SampleFormat::U16);
    try_dtype!(i32, SampleFormat::I32);
    try_dtype!(f32, SampleFormat::F32);
    try_dtype!(f64, SampleFormat::F64);
    Err(PyValueError::new_err("data must be a 2-D int8, int16, uint16, int32, float32 or float64 array (channels, samples)"))
}

impl ArrayRecording {
    /// `data` as stored values: integer arrays are the stored integers (`value = stored · gain`),
    /// float arrays the values themselves (`gain` 1).
    fn new(data: &Bound<'_, PyAny>, sample_rate: f64, gain: f32, unit: &str) -> PyResult<Self> {
        let (format, channels, samples, stored) = stored_array(data)?;
        if channels == 0 {
            return Err(PyValueError::new_err("data has no channels (expected shape (channels, samples))"));
        }
        let gain = if matches!(format, SampleFormat::F32 | SampleFormat::F64) { 1.0 } else { gain };
        let info = RecordingInfo::new("array", channels, samples as u64, SampleRate::new(sample_rate).map_err(to_py)?, format, MemoryOrder::ChannelMajor)
            .with_gain(gain, parse_unit(unit));
        Ok(Self { info, stored })
    }
}

impl RecordingSource for ArrayRecording {
    fn info(&self) -> &RecordingInfo {
        &self.info
    }

    fn read(&self, channels: &[usize], samples: Range<u64>, out: &mut [f32]) -> dspk_core::DspResult<()> {
        let n = check_read(&self.info, channels, &samples, out.len())?;
        let (total, width) = (self.info.samples as usize, self.info.format.bytes());
        for (dst, &ch) in out.chunks_exact_mut(n.max(1)).zip(channels) {
            let at = (ch * total + samples.start as usize) * width;
            let c = &self.info.channels[ch];
            self.info.format.decode(&self.stored[at..at + n * width], dst, c.gain, c.offset);
        }
        Ok(())
    }

    fn read_stored(&self, channels: &[usize], samples: Range<u64>, out: &mut [u8]) -> dspk_core::DspResult<()> {
        let n = check_read_stored(&self.info, channels, &samples, out.len())?;
        let (total, width) = (self.info.samples as usize, self.info.format.bytes());
        for (dst, &ch) in out.chunks_exact_mut((n * width).max(1)).zip(channels) {
            let at = (ch * total + samples.start as usize) * width;
            dst.copy_from_slice(&self.stored[at..at + n * width]);
        }
        Ok(())
    }
}

/// `""` (dimensionless), `"V"`, `"mV"`, `"µV"` / `"uV"`, `"A"`, `"mA"`, `"µA"` / `"uA"`, or any other
/// symbol, kept as given.
fn parse_unit(symbol: &str) -> SignalUnit {
    match symbol.trim() {
        "" => SignalUnit::Dimensionless,
        "V" => SignalUnit::Volt,
        "mV" => SignalUnit::Millivolt,
        "µV" | "uV" => SignalUnit::Microvolt,
        "A" => SignalUnit::Ampere,
        "mA" => SignalUnit::Milliampere,
        "µA" | "uA" => SignalUnit::Microampere,
        other => SignalUnit::Other(other.to_string()),
    }
}

fn parse_format(name: &str) -> PyResult<SampleFormat> {
    SampleFormat::parse(name).ok_or_else(|| PyValueError::new_err(format!("unknown sample format {name:?} (int8, int16, uint16, int32, float32, float64)")))
}

fn parse_order(name: &str) -> PyResult<MemoryOrder> {
    match name {
        "channel_major" => Ok(MemoryOrder::ChannelMajor),
        "time_major" => Ok(MemoryOrder::TimeMajor),
        other => Err(PyValueError::new_err(format!("unknown order {other:?} (channel_major or time_major)"))),
    }
}

/// Writes `data` `(channels, samples)` as raw binary at `path` with its JSON sidecar.
///
/// Integer `data` is the stored integers (`value = stored · gain`); float `data` the values.
/// `format`: stored type (default: `data`'s dtype; then integers are written as they are). Float
/// values stored in an integer format become `round(value / gain)`.
/// `order`: `"channel_major"` or `"time_major"`. `unit`: symbol of the values (`""`: dimensionless).
#[gen_stub_pyfunction(module = "dspk._native.io")]
#[pyfunction]
#[pyo3(signature = (data, path, sample_rate, *, format=None, order="channel_major", gain=1.0, unit="", chunk_samples=WRITE_CHUNK_SAMPLES))]
#[allow(clippy::too_many_arguments)]
pub fn write_raw(
    py: Python<'_>,
    data: &Bound<'_, PyAny>,
    path: PathBuf,
    sample_rate: f64,
    format: Option<&str>,
    order: &str,
    gain: f32,
    unit: &str,
    chunk_samples: usize,
) -> PyResult<()> {
    let source = ArrayRecording::new(data, sample_rate, gain, unit)?;
    let format = match format {
        Some(name) => parse_format(name)?,
        None => source.info.format,
    };
    let order = parse_order(order)?;
    py.detach(|| dspk_io::write_raw(&source, &path, format, order, gain, chunk_samples, |_, _| {})).map_err(to_py)?;
    Ok(())
}

/// Writes `data` `(channels, samples)` as a Zarr store at `path` (`/traces`, `float32`), chunks of
/// `chunk_samples` compressed with `compression` (`"zstd"`, `"blosc_lz4"` or `"none"`; `level`:
/// zstd's, default 3, or blosc's 0–9, default 5), compressed and stored on several threads unless
/// `parallel` is false.
#[gen_stub_pyfunction(module = "dspk._native.io")]
#[pyfunction]
#[pyo3(signature = (data, path, sample_rate, *, unit="", chunk_samples=ZARR_CHUNK_SAMPLES, compression="zstd", level=None, parallel=true))]
#[allow(clippy::too_many_arguments)]
pub fn write_zarr(
    py: Python<'_>,
    data: &Bound<'_, PyAny>,
    path: PathBuf,
    sample_rate: f64,
    unit: &str,
    chunk_samples: usize,
    compression: &str,
    level: Option<i32>,
    parallel: bool,
) -> PyResult<()> {
    let options = zarr_options(chunk_samples, compression, level, parallel)?;
    let source = ArrayRecording::new(data, sample_rate, 1.0, unit)?;
    py.detach(|| dspk_io::write_zarr(&source, &path, &options, |_, _| {})).map_err(to_py)
}

/// The Zarr options of `write_zarr` and `ZarrWriter`.
fn zarr_options(chunk_samples: usize, compression: &str, level: Option<i32>, parallel: bool) -> PyResult<dspk_io::ZarrOptions> {
    let compression = match (compression, level) {
        ("zstd", level) => dspk_io::ZarrCompression::Zstd { level: level.unwrap_or(DEFAULT_ZSTD_LEVEL) },
        ("blosc_lz4", level) => {
            let level = level.unwrap_or(DEFAULT_BLOSC_LEVEL);
            let level = u8::try_from(level).map_err(|_| PyValueError::new_err(format!("blosc level {level} is not in 0..=9")))?;
            dspk_io::ZarrCompression::BloscLz4 { level }
        }
        ("none", None) => dspk_io::ZarrCompression::None,
        ("none", Some(_)) => return Err(PyValueError::new_err("compression \"none\" takes no level")),
        (other, _) => return Err(PyValueError::new_err(format!("unknown compression {other:?} (zstd, blosc_lz4, none)"))),
    };
    if chunk_samples == 0 {
        return Err(PyValueError::new_err("chunk_samples must be at least 1"));
    }
    Ok(dspk_io::ZarrOptions { chunk_samples, compression, parallel })
}

/// A Zarr store (`/traces`, float32) written block by block.
///
/// For data that does not fit in memory at once, such as a recording processed window by window:
/// create the store for its full size, `push` blocks in order until every sample is written, then
/// `finish`.
///
/// Examples
/// --------
/// >>> import numpy as np
/// >>> import dspk
/// >>> w = dspk.io.ZarrWriter("out.zarr", 4, 60_000, 30_000.0, unit="µV")  # doctest: +SKIP
/// >>> for _ in range(2):  # doctest: +SKIP
/// ...     w.push(np.zeros((4, 30_000), dtype=np.float32))
/// >>> w.finish()  # doctest: +SKIP
#[gen_stub_pyclass]
#[pyclass(module = "dspk._native.io")]
pub struct ZarrWriter {
    inner: Option<dspk_io::ZarrWriter>,
    channels: usize,
}

#[gen_stub_pymethods]
#[pymethods]
impl ZarrWriter {
    /// Creates the store.
    ///
    /// Parameters
    /// ----------
    /// path : str or os.PathLike
    ///     The store (a folder) to create.
    /// channels : int
    ///     Number of channels.
    /// samples : int
    ///     Samples per channel, in total.
    /// sample_rate : float
    ///     Sample rate in Hz.
    /// unit : str
    ///     Unit symbol of the values (`""` for dimensionless).
    /// chunk_samples : int
    ///     Samples per Zarr chunk.
    /// compression : {"zstd", "blosc_lz4", "none"}
    ///     How chunks are compressed.
    /// level : int, optional
    ///     Compression level: zstd's (default 3) or blosc's 0–9 (default 5); `"none"` takes none.
    /// parallel : bool
    ///     Compress and store on several threads; `False` uses one.
    /// channel_names : list of str, optional
    ///     A name per channel.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     For zero channels or samples, a non-positive sample rate, or an invalid option.
    /// OSError
    ///     When the store cannot be created.
    #[new]
    #[pyo3(signature = (path, channels, samples, sample_rate, *, unit="", chunk_samples=ZARR_CHUNK_SAMPLES, compression="zstd", level=None, parallel=true, channel_names=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        path: PathBuf,
        channels: usize,
        samples: u64,
        sample_rate: f64,
        unit: &str,
        chunk_samples: usize,
        compression: &str,
        level: Option<i32>,
        parallel: bool,
        channel_names: Option<Vec<String>>,
    ) -> PyResult<Self> {
        if channels == 0 || samples == 0 || !(sample_rate.is_finite() && sample_rate > 0.0) {
            return Err(PyValueError::new_err(format!("channels and samples at least 1, sample_rate positive; got {channels}, {samples}, {sample_rate}")));
        }
        let options = zarr_options(chunk_samples, compression, level, parallel)?;
        let mut w = dspk_io::ZarrWriter::create(&path, channels, samples, sample_rate, &parse_unit(unit), &options).map_err(to_py)?;
        if let Some(names) = channel_names {
            w.set_channel_names(&names).map_err(to_py)?;
        }
        Ok(Self { inner: Some(w), channels })
    }

    /// Writes the next block of samples.
    ///
    /// Parameters
    /// ----------
    /// block : numpy.ndarray
    ///     The next `n` samples, `(channels, n)` float32.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     When `block` has another channel count, goes past the store's samples, or the writer is
    ///     finished.
    /// OSError
    ///     When the store cannot be written.
    fn push(&mut self, py: Python<'_>, block: PyReadonlyArray2<'_, f32>) -> PyResult<()> {
        let (rows, n) = block.as_array().dim();
        if rows != self.channels {
            return Err(PyValueError::new_err(format!("a block of {rows} channels; the store has {}", self.channels)));
        }
        let data: std::borrow::Cow<'_, [f32]> = match block.as_slice() {
            Ok(s) => std::borrow::Cow::Borrowed(s),
            Err(_) => std::borrow::Cow::Owned(block.as_array().iter().copied().collect()),
        };
        let w = self.inner.as_mut().ok_or_else(|| PyValueError::new_err("the writer is finished"))?;
        py.detach(|| w.push(&data, n, 0..n)).map_err(to_py)
    }

    /// Closes the store, after checking that every sample was written.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     When samples are missing or the writer is already finished.
    /// OSError
    ///     When the store cannot be closed.
    fn finish(&mut self) -> PyResult<()> {
        let w = self.inner.take().ok_or_else(|| PyValueError::new_err("the writer is finished"))?;
        w.finish().map_err(to_py)
    }
}

/// Default zstd level of [`write_zarr`].
const DEFAULT_ZSTD_LEVEL: i32 = 3;
/// Default blosc LZ4 level of [`write_zarr`].
const DEFAULT_BLOSC_LEVEL: i32 = 5;

pyo3_stub_gen::module_variable!("dspk._native.io", "WRITE_CHUNK_SAMPLES", usize, WRITE_CHUNK_SAMPLES);
pyo3_stub_gen::module_variable!("dspk._native.io", "ZARR_CHUNK_SAMPLES", usize, ZARR_CHUNK_SAMPLES);

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Recording>()?;
    m.add_class::<Source>()?;
    m.add_function(wrap_pyfunction!(open, m)?)?;
    m.add_function(wrap_pyfunction!(sources, m)?)?;
    m.add_function(wrap_pyfunction!(formats, m)?)?;
    m.add_function(wrap_pyfunction!(write_raw, m)?)?;
    m.add_function(wrap_pyfunction!(write_zarr, m)?)?;
    m.add_class::<ZarrWriter>()?;
    m.add("WRITE_CHUNK_SAMPLES", WRITE_CHUNK_SAMPLES)?;
    m.add("ZARR_CHUNK_SAMPLES", ZARR_CHUNK_SAMPLES)?;
    Ok(())
}
