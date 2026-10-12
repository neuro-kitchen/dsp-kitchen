//! `dsp_kitchen.synapse.ml.tridesclous2`: Tridesclous 2 (`dsp_synapse_ml::sorters::tridesclous2`),
//! ported from SpikeInterface (MIT), on the device. Settings default to SpikeInterface's.

use cubecl::prelude::Client;
use dsp_core::compute::ComputeTask;
use dsp_synapse_ml::sorters::tridesclous2::{run, tridesclous2_provenance as record, Tridesclous2Config, Tridesclous2Result};
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
            /// Detection threshold (× noise), also the peeler's.
            detect_threshold: f64,
            /// Detection neighbourhood, µm.
            detection_radius_um: f32,
            /// Detection exclusion window, ms.
            detection_exclude_sweep_ms: f64,
            /// Peaks clustered per channel (`max(min_n_peaks, n_peaks_per_channel · channels)`).
            n_peaks_per_channel: usize,
            /// Fewest peaks clustered.
            min_n_peaks: usize,
            /// Clustering waveform window before the peak, ms.
            clustering_ms_before: f64,
            /// Clustering waveform window after the peak, ms.
            clustering_ms_after: f64,
            /// SVD feature neighbourhood, µm.
            features_radius_um: f32,
            /// SVD components per channel.
            n_svd_components_per_channel: usize,
            /// Peaks the SVD is fitted on.
            svd_peaks_fit: usize,
            /// Split neighbourhood, µm.
            split_radius_um: f32,
            /// Split recursion depth.
            clustering_recursive_depth: usize,
            /// Smallest cluster that is split.
            min_size_split: usize,
            /// Dimensions each split clusters in.
            n_pca_features: usize,
            /// isosplit's initial k-means clusters (lowered for small sets).
            isosplit_n_init: usize,
            /// isosplit's smallest cluster.
            isosplit_min_cluster_size: usize,
            /// isosplit iterations per pass.
            isosplit_max_iterations_per_pass: usize,
            /// isosplit: dip scores below this merge.
            isocut_threshold: f64,
            /// Clustering templates' channels: peak-to-peak / noise at least this.
            clustering_sparsify_threshold: f64,
            /// Clustering templates without a channel of this SNR are dropped.
            clustering_min_snr: f64,
            /// Templates more similar than this merge.
            merge_similarity: f64,
            /// Template similarity lag each side, ms.
            merge_similarity_lag_ms: f64,
            /// Units firing less than this (Hz) are dropped.
            min_firing_rate: f64,
            /// Matching templates' window before the peak, ms.
            ms_before: f64,
            /// Matching templates' window after the peak, ms.
            ms_after: f64,
            /// Matching templates' channels: within this of the unit's peaks' barycentre, µm.
            template_radius_um: f32,
            /// Matching templates' channels: peak-to-peak / noise at least this.
            template_sparsify_threshold: f64,
            /// Matching templates without a channel of this SNR are dropped.
            template_min_snr_ptp: f64,
            /// Templates whose trough is farther than this from the peak (ms) are dropped.
            template_max_jitter_ms: f64,
            /// Peeler detection exclusion window, ms.
            peeler_exclude_sweep_ms: f64,
            /// Peeler detection neighbourhood, µm.
            peeler_detection_radius_um: f32,
            /// Candidate units: main channel within this of the peak, µm.
            peeler_cluster_radius_um: f32,
            /// Neighbouring spikes fitted together within this, µm.
            peeler_amplitude_fitting_radius_um: f32,
            /// Shifts tried each side, samples.
            peeler_sample_shift: usize,
            /// Short template window before the peak, ms.
            peeler_ms_before: f64,
            /// Short template window after the peak, ms.
            peeler_ms_after: f64,
            /// Peeling levels with the fast detector.
            peeler_max_loop: usize,
            /// Smallest amplitude kept.
            peeler_amplitude_min: f64,
            /// Largest amplitude subtracted.
            peeler_amplitude_max: f64,
            /// A last level with the matched-filter detector.
            peeler_fine_detector: bool,
            /// Chunks the fine detector's thresholds are fitted on.
            fine_detector_chunks: usize,
            /// Final cleaning (`auto_merge_units`, cross-contamination presets).
            final_merges: bool,
            /// Furthest apart (µm) two units' locations may be to merge.
            final_merge_max_distance_um: f64,
            /// Merged trains drop spikes closer than this, ms.
            final_merge_censor_ms: f64,
            /// Units merge only when their channels overlap (intersection / union) at least this much.
            final_merge_sparsity_overlap: f64,
            /// Window length, s.
            chunk_sec: f64,
            /// Seed of the selections and k-means.
            seed: u64,
        }
    };
}

macro_rules! config_class {
    ($( $(#[$m:meta])* $f:ident : $t:ty ),* $(,)?) => {
        /// Tridesclous 2 settings: SpikeInterface's `_default_params` and the defaults of the
        /// components it calls.
        ///
        /// Every argument defaults to Tridesclous 2's default; lengths of time in ms, distances
        /// in µm. Change any setting by keyword, or later as an attribute.
        ///
        /// Examples
        /// --------
        /// >>> from dsp_kitchen.synapse.ml import tridesclous2
        /// >>> config = tridesclous2.Config(detect_threshold=6.0)
        #[gen_stub_pyclass]
        #[pyclass(name = "Tridesclous2Config", get_all, set_all, skip_from_py_object)]
        #[derive(Clone)]
        pub struct PyTridesclous2Config {
            $( $(#[$m])* $f: $t, )*
        }

        #[gen_stub_pymethods]
        #[pymethods]
        impl PyTridesclous2Config {
            /// Tridesclous 2's defaults, changed by keyword (see the class and attribute docs).
            #[new]
            #[pyo3(signature = (*, $( $f = defaults().$f ),*))]
            #[allow(clippy::too_many_arguments)]
            fn new($( $f: $t ),*) -> Self {
                Self { $( $f ),* }
            }

            fn __repr__(&self) -> String {
                format!(
                    "Tridesclous2Config(detect_threshold={}, detection_radius_um={}, n_pca_features={}, ...)",
                    self.detect_threshold, self.detection_radius_um, self.n_pca_features
                )
            }
        }

        impl PyTridesclous2Config {
            fn from_rust(c: &Tridesclous2Config) -> Self {
                Self {
                    $( $f: c.$f.clone(), )*
                }
            }

            fn to_rust(&self) -> PyResult<Tridesclous2Config> {
                Ok(Tridesclous2Config { $( $f: self.$f.clone(), )* })
            }
        }
    };
}

with_settings!(config_class);

/// Tridesclous 2's defaults in their Python form: the source of every default of
/// `Tridesclous2Config(...)`.
fn defaults() -> PyTridesclous2Config {
    PyTridesclous2Config::from_rust(&Tridesclous2Config::default())
}

/// Result of a Tridesclous 2 run (`tridesclous2.run`).
///
/// Spikes are template matches: recording sample of the template's peak, unit, scaling (the
/// template's amplitude multiplier). Templates are whitened.
///
/// Examples
/// --------
/// >>> result = tridesclous2.run(recording, probe, tridesclous2.Config())
/// >>> result.n_units
/// >>> sorting = result.to_sorting_output(probe)
#[gen_stub_pyclass]
#[pyclass(name = "Tridesclous2Result", skip_from_py_object)]
pub struct PyTridesclous2Result {
    inner: Tridesclous2Result,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyTridesclous2Result {
    /// `"tridesclous2"`.
    #[getter]
    fn sorter(&self) -> &'static str {
        dsp_synapse_ml::sorters::tridesclous2::TRIDESCLOUS2_SORTER
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
        format!("Tridesclous2Result(units={}, spikes={}, peaks={})", self.inner.n_units, self.inner.spike_samples.len(), self.inner.detected)
    }
}

/// Tridesclous 2 over a whole recording, without a progress bar. Prefer
/// `dsp_kitchen.synapse.ml.tridesclous2.run`, which documents every argument and shows progress.
#[gen_stub_pyfunction]
#[pyfunction(name = "run_tridesclous2")]
#[pyo3(signature = (recording, probe, config, *, progress=None, runtime=None))]
fn run_tridesclous2_py(
    py: Python<'_>,
    recording: PyRef<'_, crate::buffer::PyRecording>,
    probe: PyRef<'_, PyProbeLayout>,
    config: PyRef<'_, PyTridesclous2Config>,
    progress: Option<Py<PyAny>>,
    runtime: Option<&str>,
) -> PyResult<PyTridesclous2Result> {
    struct Task<'a>(&'a dyn dsp_core::RecordingSource, &'a dsp_io::neuro::probe::SensorLayout, &'a Tridesclous2Config, &'a dyn dsp_core::ProgressSink);
    impl ComputeTask for Task<'_> {
        type Output = dsp_core::DspResult<Tridesclous2Result>;
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
    Ok(PyTridesclous2Result { inner })
}

/// Provenance of the Tridesclous 2 port.
#[gen_stub_pyfunction]
#[pyfunction]
fn tridesclous2_provenance() -> PyProvenance {
    PyProvenance { inner: record() }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTridesclous2Config>()?;
    m.add_class::<PyTridesclous2Result>()?;
    m.add_function(wrap_pyfunction!(run_tridesclous2_py, m)?)?;
    m.add_function(wrap_pyfunction!(tridesclous2_provenance, m)?)
}
