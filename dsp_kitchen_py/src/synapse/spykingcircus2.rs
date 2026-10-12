//! `dsp_kitchen.synapse.ml.spykingcircus2`: SpyKING CIRCUS 2 (`dsp_synapse_ml::sorters::spykingcircus2`),
//! ported from SpikeInterface (MIT), on the device. Settings default to SpikeInterface's.

use cubecl::prelude::Client;
use dsp_core::compute::ComputeTask;
use dsp_synapse_ml::sorters::spykingcircus2::{run, spykingcircus2_provenance as record, Spykingcircus2Config, Spykingcircus2Result};
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
            /// Waveform window before the peak, ms.
            ms_before: f64,
            /// Waveform window after the peak, ms.
            ms_after: f64,
            /// Feature neighbourhood, µm (detection uses half of it).
            radius_um: f32,
            /// Bessel band-pass (forward-backward).
            do_bandpass: bool,
            /// Lower band edge, Hz.
            bandpass_low_hz: f64,
            /// Upper band edge, Hz; `None`: a high-pass at `bandpass_low_hz`.
            bandpass_high_hz: Option<f64>,
            /// Bessel order (per edge).
            filter_order: usize,
            /// Common median reference (on recordings of at least `common_reference_min_channels`).
            do_common_reference: bool,
            /// Fewest channels the common median reference is applied to.
            common_reference_min_channels: usize,
            /// Whitening.
            do_whiten: bool,
            /// Local whitening radius, µm; `None`: global.
            whitening_radius_um: Option<f32>,
            /// Chunks the whitening is fitted on (evenly spaced).
            whitening_chunks: usize,
            /// Length of each whitening chunk, ms.
            whitening_chunk_ms: f64,
            /// Regularisation added to the covariance eigenvalues.
            whitening_epsilon: f32,
            /// Chunks the noise levels are measured on.
            noise_chunks: usize,
            /// Length of each noise chunk, ms.
            noise_chunk_ms: f64,
            /// Detection threshold (× noise).
            detect_threshold: f64,
            /// Peaks the detection prototype is the median of.
            prototype_peaks: usize,
            /// Chunks the matched filter's thresholds are fitted on.
            matched_filter_chunks: usize,
            /// Peaks clustered per channel (`max(min_n_peaks, n_peaks_per_channel · channels)`).
            n_peaks_per_channel: usize,
            /// Fewest peaks clustered.
            min_n_peaks: usize,
            /// SVD components per channel.
            svd_components: usize,
            /// Peaks the SVD is fitted on.
            svd_peaks_fit: usize,
            /// Split neighbourhood, µm.
            split_radius_um: f32,
            /// Split recursion depth.
            split_depth: usize,
            /// HDBSCAN's smallest cluster.
            min_cluster_size: usize,
            /// Dimensions each split clusters in.
            split_pca_features: usize,
            /// Template channels: peak-to-peak / noise at least this.
            sparsify_threshold: f64,
            /// Templates without a channel of this SNR are dropped.
            min_snr: f64,
            /// Templates whose trough is farther than this from the peak (ms) are dropped.
            max_jitter_ms: f64,
            /// Templates whose mean max-std / noise exceeds this are dropped.
            mean_sd_ratio_threshold: f64,
            /// Templates more similar than this merge.
            merge_similarity: f64,
            /// Lags (samples) the template similarity tries each side.
            merge_num_shifts: usize,
            /// Units firing less than this (Hz) are dropped.
            min_firing_rate: f64,
            /// Smallest matching amplitude kept.
            omp_min_amplitude: f32,
            /// Matching rounds without a new spike before it stops.
            omp_max_failures: usize,
            /// Rank of the templates in the matching.
            omp_rank: usize,
            /// Matching neighbourhood, template widths.
            omp_vicinity: usize,
            /// Final cleaning (`auto_merge_units`, cross-contamination presets).
            final_merges: bool,
            /// Furthest apart (µm) two units' locations may be to merge.
            final_merge_max_distance_um: f64,
            /// Merged trains drop spikes closer than this, ms.
            final_merge_censor_ms: f64,
            /// Units merge only when their channels overlap (intersection / union) at least this much.
            final_merge_sparsity_overlap: f64,
            /// Template similarity lag each side, ms.
            final_merge_max_lag_ms: f64,
            /// Window length, s.
            chunk_sec: f64,
            /// Seed of the shuffles and selections.
            seed: u64,
        }
    };
}

macro_rules! config_class {
    ($( $(#[$m:meta])* $f:ident : $t:ty ),* $(,)?) => {
        /// SpyKING CIRCUS 2 settings: SpikeInterface's `_default_params` and the defaults of the
        /// components it calls.
        ///
        /// Every argument defaults to SpyKING CIRCUS 2's default; lengths of time in ms, distances
        /// in µm. Change any setting by keyword, or later as an attribute.
        ///
        /// Examples
        /// --------
        /// >>> from dsp_kitchen.synapse.ml import spykingcircus2
        /// >>> config = spykingcircus2.Config(detect_threshold=6.0)
        #[gen_stub_pyclass]
        #[pyclass(name = "Spykingcircus2Config", get_all, set_all, skip_from_py_object)]
        #[derive(Clone)]
        pub struct PySpykingcircus2Config {
            $( $(#[$m])* $f: $t, )*
        }

        #[gen_stub_pymethods]
        #[pymethods]
        impl PySpykingcircus2Config {
            /// SpyKING CIRCUS 2's defaults, changed by keyword (see the class and attribute docs).
            #[new]
            #[pyo3(signature = (*, $( $f = defaults().$f ),*))]
            #[allow(clippy::too_many_arguments)]
            fn new($( $f: $t ),*) -> Self {
                Self { $( $f ),* }
            }

            fn __repr__(&self) -> String {
                format!(
                    "Spykingcircus2Config(detect_threshold={}, radius_um={}, min_cluster_size={}, ...)",
                    self.detect_threshold, self.radius_um, self.min_cluster_size
                )
            }
        }

        impl PySpykingcircus2Config {
            fn from_rust(c: &Spykingcircus2Config) -> Self {
                Self {
                    $( $f: c.$f.clone(), )*
                }
            }

            fn to_rust(&self) -> PyResult<Spykingcircus2Config> {
                Ok(Spykingcircus2Config { $( $f: self.$f.clone(), )* })
            }
        }
    };
}

with_settings!(config_class);

/// SpyKING CIRCUS 2's defaults in their Python form: the source of every default of
/// `Spykingcircus2Config(...)`.
fn defaults() -> PySpykingcircus2Config {
    PySpykingcircus2Config::from_rust(&Spykingcircus2Config::default())
}

/// Result of a SpyKING CIRCUS 2 run (`spykingcircus2.run`).
///
/// Spikes are template matches: recording sample of the template's peak, unit, scaling (the
/// template's amplitude multiplier). Templates are whitened.
///
/// Examples
/// --------
/// >>> result = spykingcircus2.run(recording, probe, spykingcircus2.Config())
/// >>> result.n_units
/// >>> sorting = result.to_sorting_output(probe)
#[gen_stub_pyclass]
#[pyclass(name = "Spykingcircus2Result", skip_from_py_object)]
pub struct PySpykingcircus2Result {
    inner: Spykingcircus2Result,
}

#[gen_stub_pymethods]
#[pymethods]
impl PySpykingcircus2Result {
    /// `"spykingcircus2"`.
    #[getter]
    fn sorter(&self) -> &'static str {
        dsp_synapse_ml::sorters::spykingcircus2::SPYKINGCIRCUS2_SORTER
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
    /// Units merged by the final cleaning.
    #[getter]
    fn final_merges(&self) -> usize {
        self.inner.final_merges
    }
    /// Peaks detected, and how many of them were clustered.
    #[getter]
    fn peaks(&self) -> (usize, usize) {
        (self.inner.detected, self.inner.selected)
    }
    /// Noise level of every channel (whitened units).
    #[getter]
    fn noise_levels(&self) -> Vec<f64> {
        self.inner.noise_levels.clone()
    }
    /// The detection prototype waveform.
    #[getter]
    fn prototype(&self) -> Vec<f32> {
        self.inner.prototype.clone()
    }
    /// The preprocessing the sorter saw (filters, reference, whitening) as a reusable `Pipeline`.
    #[getter]
    fn preprocessing(&self) -> PyPipeline {
        PyPipeline { stages: self.inner.pipeline.stages().to_vec() }
    }
    /// Unit templates, `[n_units, samples, channels]` float32 (whitened; zeros off each unit's channels).
    #[getter]
    #[gen_stub(override_return_type(type_repr = "numpy.typing.NDArray[numpy.float32]", imports = ("numpy", "numpy.typing")))]
    fn templates<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let t = &self.inner.templates;
        to_numpy(py, t.data.clone(), &[t.len(), t.width, t.channels])
    }
    /// The spikes, one entry per spike in every array.
    ///
    /// Returns
    /// -------
    /// dict
    ///     `sample` (recording sample), `unit`, `scaling` (matching amplitude).
    fn spikes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let r = &self.inner;
        let d = PyDict::new(py);
        d.set_item("sample", r.spike_samples.clone())?;
        d.set_item("unit", r.spike_units.clone())?;
        d.set_item("scaling", to_numpy(py, r.spike_scalings.clone(), &[r.spike_scalings.len()])?)?;
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
        format!("Spykingcircus2Result(units={}, spikes={}, peaks={})", self.inner.n_units, self.inner.spike_samples.len(), self.inner.detected)
    }
}

/// SpyKING CIRCUS 2 over a whole recording, without a progress bar. Prefer
/// `dsp_kitchen.synapse.ml.spykingcircus2.run`, which documents every argument and shows progress.
#[gen_stub_pyfunction]
#[pyfunction(name = "run_spykingcircus2")]
#[pyo3(signature = (recording, probe, config, *, progress=None, runtime=None))]
fn run_spykingcircus2_py(
    py: Python<'_>,
    recording: PyRef<'_, crate::buffer::PyRecording>,
    probe: PyRef<'_, PyProbeLayout>,
    config: PyRef<'_, PySpykingcircus2Config>,
    progress: Option<Py<PyAny>>,
    runtime: Option<&str>,
) -> PyResult<PySpykingcircus2Result> {
    struct Task<'a>(&'a dyn dsp_core::RecordingSource, &'a dsp_io::neuro::probe::SensorLayout, &'a Spykingcircus2Config, &'a dyn dsp_core::ProgressSink);
    impl ComputeTask for Task<'_> {
        type Output = dsp_core::DspResult<Spykingcircus2Result>;
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
    Ok(PySpykingcircus2Result { inner })
}

/// Provenance of the SpyKING CIRCUS 2 port.
#[gen_stub_pyfunction]
#[pyfunction]
fn spykingcircus2_provenance() -> PyProvenance {
    PyProvenance { inner: record() }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySpykingcircus2Config>()?;
    m.add_class::<PySpykingcircus2Result>()?;
    m.add_function(wrap_pyfunction!(run_spykingcircus2_py, m)?)?;
    m.add_function(wrap_pyfunction!(spykingcircus2_provenance, m)?)
}
