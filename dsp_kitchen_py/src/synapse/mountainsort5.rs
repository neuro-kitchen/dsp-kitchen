//! `dsp_kitchen.synapse.ml.mountainsort5`: MountainSort 5 (`dsp_synapse_ml::sorters::mountainsort5`),
//! ported from its Apache-2.0 source, on the device. Settings default to MountainSort 5's (and
//! SpikeInterface's wrapper's where the package leaves them to the caller).

use cubecl::prelude::Client;
use dsp_core::compute::ComputeTask;
use dsp_synapse_ml::sorters::mountainsort5::{
    mountainsort5_provenance as record, run, Mountainsort5Config, Mountainsort5Result, Scheme, TrainingSampling,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pyfunction, gen_stub_pymethods};

use super::ml::{PyProgress, PyProvenance};
use super::probe::PyProbeLayout;
use super::storage::PySortingOutput;
use crate::array::{runtime_error, to_numpy, value_error};
use crate::pipeline::engine::PyPipeline;
use crate::runtime::target;

/// Lists every setting once (name, Python type, doc) and hands the list to `$callback`.
macro_rules! with_settings {
    ($callback:ident) => {
        $callback! {
            /// `2`: train on a stretch, then classify every spike (the default); `1`: one pass.
            scheme: u8,
            /// Common average reference before filtering (not in SpikeInterface's wrapper).
            do_car: bool,
            /// Band-pass the recording (Butterworth order 5, forward-backward).
            do_bandpass: bool,
            /// Lower band edge, Hz.
            bandpass_low_hz: f64,
            /// Upper band edge, Hz; `None`: a high-pass at `bandpass_low_hz`.
            bandpass_high_hz: Option<f64>,
            /// Line-noise notch after the band-pass.
            do_notch: bool,
            /// Notch frequency, Hz.
            notch_hz: f64,
            /// Notch quality factor.
            notch_q: f64,
            /// Global ZCA whitening (MountainSort 5 expects whitened data).
            do_whiten: bool,
            /// Chunks the whitening is fitted on (evenly spaced).
            whitening_chunks: usize,
            /// Length of each whitening chunk, ms.
            whitening_chunk_ms: f64,
            /// Regularisation added to the covariance eigenvalues.
            whitening_epsilon: f32,
            /// Detection threshold (whitened units); scheme 2's classification phase.
            detect_threshold: f32,
            /// `-1` negative peaks, `1` positive, `0` both.
            detect_sign: i8,
            /// Events closer than this compete (ms).
            detect_time_radius_ms: f64,
            /// Scheme 1's detection neighbourhood (µm; `None`: every channel).
            scheme1_detect_channel_radius_um: Option<f32>,
            /// Scheme 2's training-phase detection neighbourhood (µm).
            phase1_detect_channel_radius_um: Option<f32>,
            /// Scheme 2's training-phase threshold.
            phase1_detect_threshold: f32,
            /// Scheme 2's training-phase time radius (ms).
            phase1_detect_time_radius_ms: f64,
            /// Scheme 2's classification-phase detection neighbourhood (µm).
            detect_channel_radius_um: Option<f32>,
            /// Snippet samples before the event.
            snippet_t1: usize,
            /// Snippet samples after the event.
            snippet_t2: usize,
            /// Snippet channels: within this distance of the event's channel (µm; `None`: all).
            snippet_mask_radius_um: Option<f32>,
            /// PCA components per channel of the first clustering.
            npca_per_channel: usize,
            /// PCA components of each isosplit6 subdivision.
            npca_per_subdivision: usize,
            /// Skip the template alignment step.
            skip_alignment: bool,
            /// isosplit6: dip scores below this merge two clusters.
            isocut_threshold: f64,
            /// isosplit6: smaller clusters always merge.
            min_cluster_size: usize,
            /// isosplit6: initial parcels.
            k_init: usize,
            /// isosplit6: iterations per pass.
            max_iterations_per_pass: usize,
            /// Scheme 2's training stretch (s; `None`: the whole recording).
            training_duration_sec: Option<f64>,
            /// `"uniform"`: 10 s chunks spread over the recording; `"initial"`: the start.
            training_sampling: String,
            /// Noise snippets, and snippets per unit and channel, of the classifiers.
            max_num_snippets_per_training_batch: usize,
            /// Classifier PCA components (`None`: `max(12, 3 · mask channels)`).
            classifier_npca: Option<usize>,
            /// Seconds per window (`None`: 10⁸ values / channels, upstream's chunk).
            classification_chunk_sec: Option<f64>,
            /// Above this many features PCA is randomized.
            pca_exact_cap: usize,
            /// Seed of the randomized PCA.
            seed: u64,
        }
    };
}

macro_rules! config_class {
    ($( $(#[$m:meta])* $f:ident : $t:ty ),* $(,)?) => {
        /// MountainSort 5 settings, under the package's names (`Scheme1SortingParameters`,
        /// `Scheme2SortingParameters`) and SpikeInterface's wrapper's.
        ///
        /// Every argument defaults to MountainSort 5's default; thresholds are in whitened units,
        /// snippet lengths in **samples**, distances in µm. Change any setting by keyword, or later
        /// as an attribute.
        ///
        /// Examples
        /// --------
        /// >>> from dsp_kitchen.synapse.ml import mountainsort5
        /// >>> config = mountainsort5.Config(detect_threshold=6.0)
        /// >>> config.scheme = 1
        #[gen_stub_pyclass]
        #[pyclass(name = "Mountainsort5Config", get_all, set_all, skip_from_py_object)]
        #[derive(Clone)]
        pub struct PyMountainsort5Config {
            $( $(#[$m])* $f: $t, )*
        }

        #[gen_stub_pymethods]
        #[pymethods]
        impl PyMountainsort5Config {
            /// MountainSort 5's defaults, changed by keyword (see the class and attribute docs).
            #[new]
            #[pyo3(signature = (*, $( $f = defaults().$f ),*))]
            #[allow(clippy::too_many_arguments)]
            fn new($( $f: $t ),*) -> Self {
                Self { $( $f ),* }
            }

            fn __repr__(&self) -> String {
                let radius = self.snippet_mask_radius_um.map_or("None".to_string(), |r| r.to_string());
                format!(
                    "Mountainsort5Config(scheme={}, detect_threshold={}, snippet_t1={}, snippet_t2={}, snippet_mask_radius_um={radius}, ...)",
                    self.scheme, self.detect_threshold, self.snippet_t1, self.snippet_t2
                )
            }
        }

        impl PyMountainsort5Config {
            fn from_rust(c: &Mountainsort5Config) -> Self {
                Self {
                    $( $f: Convert::to_py(&c.$f), )*
                }
            }

            fn to_rust(&self) -> PyResult<Mountainsort5Config> {
                Ok(Mountainsort5Config { $( $f: Convert::from_py(&self.$f)?, )* })
            }
        }
    };
}

/// Between a Rust setting and its Python form (identity except the two enums).
trait Convert<P>: Sized {
    fn to_py(&self) -> P;
    fn from_py(p: &P) -> PyResult<Self>;
}

impl<T: Clone> Convert<T> for T {
    fn to_py(&self) -> T {
        self.clone()
    }
    fn from_py(p: &T) -> PyResult<Self> {
        Ok(p.clone())
    }
}

impl Convert<u8> for Scheme {
    fn to_py(&self) -> u8 {
        match self {
            Scheme::One => 1,
            Scheme::Two => 2,
        }
    }
    fn from_py(p: &u8) -> PyResult<Self> {
        match p {
            1 => Ok(Scheme::One),
            2 => Ok(Scheme::Two),
            _ => Err(PyValueError::new_err(format!("scheme must be 1 or 2, not {p}"))),
        }
    }
}

impl Convert<String> for TrainingSampling {
    fn to_py(&self) -> String {
        match self {
            TrainingSampling::Initial => "initial".into(),
            TrainingSampling::Uniform => "uniform".into(),
        }
    }
    fn from_py(p: &String) -> PyResult<Self> {
        match p.as_str() {
            "initial" => Ok(TrainingSampling::Initial),
            "uniform" => Ok(TrainingSampling::Uniform),
            _ => Err(PyValueError::new_err(format!("training_sampling must be \"uniform\" or \"initial\", not {p:?}"))),
        }
    }
}

with_settings!(config_class);

/// MountainSort 5's defaults in their Python form: the source of every default of
/// `Mountainsort5Config(...)`.
fn defaults() -> PyMountainsort5Config {
    PyMountainsort5Config::from_rust(&Mountainsort5Config::default())
}

/// Result of a MountainSort 5 run (`mountainsort5.run`).
///
/// Spike times are recording samples (moved to their unit's template peak), amplitudes the whitened
/// trace at the detection, channels the detection channel. Templates are the units' median snippets
/// (whitened).
///
/// Examples
/// --------
/// >>> result = mountainsort5.run(recording, probe, mountainsort5.Config())
/// >>> result.n_units
/// >>> spikes = result.spikes()
/// >>> sorting = result.to_sorting_output(probe)
#[gen_stub_pyclass]
#[pyclass(name = "Mountainsort5Result", skip_from_py_object)]
pub struct PyMountainsort5Result {
    inner: Mountainsort5Result,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyMountainsort5Result {
    /// `"mountainsort5"`.
    #[getter]
    fn sorter(&self) -> &'static str {
        dsp_synapse_ml::sorters::mountainsort5::MOUNTAINSORT5_SORTER
    }
    /// Units found.
    #[getter]
    fn n_units(&self) -> usize {
        self.inner.n_units
    }
    /// Sampling rate of the recording, Hz.
    #[getter]
    fn sample_rate_hz(&self) -> f64 {
        self.inner.sample_rate_hz
    }
    /// Samples in the recording.
    #[getter]
    fn total_samples(&self) -> u64 {
        self.inner.total_samples
    }
    /// Spikes of the first phase (scheme 2: the training stretch; scheme 1: all).
    #[getter]
    fn phase1_spikes(&self) -> usize {
        self.inner.phase1_spikes
    }
    /// The preprocessing the sorter saw (filters, whitening) as a reusable `Pipeline`.
    #[getter]
    fn preprocessing(&self) -> PyPipeline {
        PyPipeline { stages: self.inner.pipeline.stages().to_vec() }
    }
    /// Unit templates, `[n_units, snippet_t1 + snippet_t2, channels]` float32 (whitened; zeros off
    /// the channels a template was computed on).
    #[getter]
    #[gen_stub(override_return_type(type_repr = "numpy.typing.NDArray[numpy.float32]", imports = ("numpy", "numpy.typing")))]
    fn templates<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        to_numpy(py, self.inner.templates.clone(), &[self.inner.n_units, self.inner.width, self.inner.channels])
    }
    /// Channel of each unit's template minimum.
    #[getter]
    fn peak_channels(&self) -> Vec<usize> {
        self.inner.peak_channels.clone()
    }
    /// The spikes, one entry per spike in every array.
    ///
    /// Returns
    /// -------
    /// dict
    ///     `sample` (recording sample), `unit`, `amplitude` (whitened), `channel` (detection
    ///     channel).
    fn spikes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let r = &self.inner;
        let d = PyDict::new(py);
        d.set_item("sample", r.spike_samples.clone())?;
        d.set_item("unit", r.spike_units.clone())?;
        d.set_item("amplitude", to_numpy(py, r.spike_amplitudes.clone(), &[r.spike_amplitudes.len()])?)?;
        d.set_item("channel", r.spike_channels.clone())?;
        Ok(d)
    }
    /// The units in `SortingOutput` form, for export (`save_sorting`: Phy, zarr) and inspection.
    ///
    /// Parameters
    /// ----------
    /// probe : ProbeLayout, optional
    ///     Geometry stored with the units (channel positions in exports).
    #[pyo3(signature = (probe=None))]
    fn to_sorting_output(&self, probe: Option<PyRef<'_, PyProbeLayout>>) -> PySortingOutput {
        PySortingOutput::new(self.inner.to_sorting_output(probe.map(|p| p.inner.clone())))
    }
    fn __repr__(&self) -> String {
        format!("Mountainsort5Result(units={}, spikes={}, phase1_spikes={})", self.inner.n_units, self.inner.spike_samples.len(), self.inner.phase1_spikes)
    }
}

/// MountainSort 5 over a whole recording, without a progress bar. Prefer
/// `dsp_kitchen.synapse.ml.mountainsort5.run`, which documents every argument and shows progress.
#[gen_stub_pyfunction]
#[pyfunction(name = "run_mountainsort5")]
#[pyo3(signature = (recording, probe, config, *, progress=None, runtime=None))]
fn run_mountainsort5_py(
    py: Python<'_>,
    recording: PyRef<'_, crate::buffer::PyRecording>,
    probe: PyRef<'_, PyProbeLayout>,
    config: PyRef<'_, PyMountainsort5Config>,
    progress: Option<Py<PyAny>>,
    runtime: Option<&str>,
) -> PyResult<PyMountainsort5Result> {
    struct Task<'a>(&'a dyn dsp_core::RecordingSource, &'a dsp_io::neuro::probe::SensorLayout, &'a Mountainsort5Config, &'a dyn dsp_core::ProgressSink);
    impl ComputeTask for Task<'_> {
        type Output = dsp_core::DspResult<Mountainsort5Result>;
        fn run(self, client: Client) -> Self::Output {
            run(&client, self.0, self.1, self.2, self.3)
        }
    }
    let config = config.to_rust()?;
    let sink: Box<dyn dsp_core::ProgressSink> = match progress {
        Some(callable) => Box::new(PyProgress(callable)),
        None => Box::new(dsp_core::NoProgress),
    };
    let source = recording.inner.clone();
    let (layout, target) = (probe.inner.clone(), target(runtime)?);
    let inner = py.detach(|| target.run(Task(source.as_ref(), &layout, &config, sink.as_ref()))).map_err(runtime_error)?.map_err(value_error)?;
    Ok(PyMountainsort5Result { inner })
}

/// Provenance of the MountainSort 5 port.
#[gen_stub_pyfunction]
#[pyfunction]
fn mountainsort5_provenance() -> PyProvenance {
    PyProvenance { inner: record() }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyMountainsort5Config>()?;
    m.add_class::<PyMountainsort5Result>()?;
    m.add_function(wrap_pyfunction!(run_mountainsort5_py, m)?)?;
    m.add_function(wrap_pyfunction!(mountainsort5_provenance, m)?)
}
