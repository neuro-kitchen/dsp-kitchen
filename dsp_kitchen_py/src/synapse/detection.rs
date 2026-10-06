//! Spike detection, spatial deduplication and noise estimation. Detection settings default to
//! `StreamingDetectionConfig::default()` (threshold factor, polarity, distance rule); the
//! refractory period and the deduplication radius / window depend on the recording and have no
//! default.

use dsp_base::peaks::DistanceRule;
use dsp_synapse::detection::{
    deduplicate_spikes_spatial, detect_spikes_multichannel, detect_spikes_with_sigma, estimate_noise_rms, estimate_noise_std, estimate_noise_trimmed, DeduplicatedSpike,
    SpikeEvent, SpikePolarity, SpikeSpacing,
};
use dsp_synapse::StreamingDetectionConfig;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use super::probe::PyProbeLayout;
use crate::array::{to_numpy, F32Array};

/// A threshold crossing: `amplitude` is the extremum, in the recording's unit.
#[pyclass(name = "SpikeEvent", skip_from_py_object)]
#[derive(Clone)]
pub struct PySpikeEvent {
    #[pyo3(get)]
    pub channel: usize,
    #[pyo3(get)]
    pub sample: u64,
    #[pyo3(get)]
    pub amplitude: f32,
}

#[pymethods]
impl PySpikeEvent {
    #[new]
    fn new(channel: usize, sample: u64, amplitude: f32) -> Self {
        Self { channel, sample, amplitude }
    }
    fn __repr__(&self) -> String {
        format!("SpikeEvent(channel={}, sample={}, amplitude={})", self.channel, self.sample, self.amplitude)
    }
}

impl From<SpikeEvent> for PySpikeEvent {
    fn from(s: SpikeEvent) -> Self {
        Self { channel: s.channel_id, sample: s.sample_index, amplitude: s.peak_amplitude_uv }
    }
}

impl From<&PySpikeEvent> for SpikeEvent {
    fn from(s: &PySpikeEvent) -> Self {
        Self { channel_id: s.channel, sample_index: s.sample, peak_amplitude_uv: s.amplitude }
    }
}

/// One spike after deduplication: its strongest channel and the channels it reached.
#[pyclass(name = "DeduplicatedSpike", skip_from_py_object)]
#[derive(Clone)]
pub struct PyDeduplicatedSpike {
    #[pyo3(get)]
    pub primary_channel: usize,
    #[pyo3(get)]
    pub sample: u64,
    #[pyo3(get)]
    pub amplitude: f32,
    #[pyo3(get)]
    pub participating_channels: Vec<usize>,
}

#[pymethods]
impl PyDeduplicatedSpike {
    fn __repr__(&self) -> String {
        format!("DeduplicatedSpike(primary_channel={}, sample={}, amplitude={}, channels={:?})", self.primary_channel, self.sample, self.amplitude, self.participating_channels)
    }
}

impl From<DeduplicatedSpike> for PyDeduplicatedSpike {
    fn from(d: DeduplicatedSpike) -> Self {
        Self { primary_channel: d.primary_channel, sample: d.sample_index, amplitude: d.peak_amplitude_uv, participating_channels: d.participating_channels }
    }
}

impl From<&PyDeduplicatedSpike> for DeduplicatedSpike {
    fn from(d: &PyDeduplicatedSpike) -> Self {
        Self { primary_channel: d.primary_channel, sample_index: d.sample, peak_amplitude_uv: d.amplitude, participating_channels: d.participating_channels.clone() }
    }
}

/// A noise σ estimator of one channel.
type NoiseEstimator = Box<dyn Fn(&[f32]) -> f32 + Send + Sync>;

pub(crate) fn parse_polarity(polarity: &str) -> PyResult<SpikePolarity> {
    match polarity {
        "negative" => Ok(SpikePolarity::Negative),
        "positive" => Ok(SpikePolarity::Positive),
        "both" => Ok(SpikePolarity::Both),
        other => Err(PyValueError::new_err(format!("polarity must be 'negative', 'positive' or 'both', got '{other}'"))),
    }
}

/// `distance_rule=`: `"locally-exclusive"` (a spike stays unless a larger one is nearer; exact in
/// chunks) or `"scipy"` (`scipy.signal.find_peaks` `distance`).
pub(crate) fn parse_distance_rule(rule: &str) -> PyResult<DistanceRule> {
    match rule {
        "locally-exclusive" => Ok(DistanceRule::LocallyExclusive),
        "scipy" => Ok(DistanceRule::Scipy),
        other => Err(PyValueError::new_err(format!("distance_rule must be 'locally-exclusive' or 'scipy', got '{other}'"))),
    }
}

/// Threshold crossings of `data` (`[channels, samples]`): local extrema beyond
/// `threshold_factor · σ` per channel (σ = MAD / 0.6745 of each channel, or `sigmas`), at least
/// `refractory_samples` apart on a channel.
#[pyfunction]
#[pyo3(signature = (data, *, refractory_samples, threshold_factor=None, polarity=None, distance_rule=None, sigmas=None))]
pub fn detect_spikes(
    py: Python<'_>,
    data: Bound<'_, PyAny>,
    refractory_samples: usize,
    threshold_factor: Option<f32>,
    polarity: Option<&str>,
    distance_rule: Option<&str>,
    sigmas: Option<Vec<f32>>,
) -> PyResult<Vec<PySpikeEvent>> {
    let defaults = StreamingDetectionConfig::default();
    let threshold = threshold_factor.unwrap_or(defaults.threshold_factor);
    let polarity = polarity.map(parse_polarity).transpose()?.unwrap_or(defaults.polarity);
    let rule = distance_rule.map(parse_distance_rule).transpose()?.unwrap_or(defaults.distance_rule);
    let spacing = SpikeSpacing { refractory_samples, rule };
    let input = F32Array::new(&data)?;
    let (channels, samples) = input.channels_samples(None)?;
    if let Some(s) = &sigmas
        && s.len() != channels
    {
        return Err(PyValueError::new_err(format!("{} sigmas for {channels} channels", s.len())));
    }
    let x = input.slice();
    let spikes = py.detach(|| match &sigmas {
        Some(s) => detect_spikes_with_sigma(x, channels, samples, s, threshold, polarity, spacing),
        None => detect_spikes_multichannel(x, channels, samples, threshold, polarity, spacing),
    });
    Ok(spikes.into_iter().map(PySpikeEvent::from).collect())
}

/// One spike per action potential: a crossing survives unless a stronger one is within
/// `radius_um` (on `probe`) and `window_samples`.
#[pyfunction]
#[pyo3(signature = (spikes, probe, *, radius_um, window_samples))]
pub fn deduplicate_spikes(py: Python<'_>, spikes: Vec<PyRef<'_, PySpikeEvent>>, probe: PyRef<'_, PyProbeLayout>, radius_um: f32, window_samples: u64) -> Vec<PyDeduplicatedSpike> {
    let events: Vec<SpikeEvent> = spikes.iter().map(|s| SpikeEvent::from(&**s)).collect();
    let layout = &probe.inner;
    let deduplicated = py.detach(|| deduplicate_spikes_spatial(&events, layout, radius_um, window_samples));
    deduplicated.into_iter().map(PyDeduplicatedSpike::from).collect()
}

/// Noise σ of every channel of `data` (`[channels, samples]`): `"mad"` (median |x| / 0.6745,
/// robust to spikes), `"rms"`, or `"trimmed"` (σ re-estimated without samples beyond
/// `clip_sigma · σ`, `iterations` times; both required).
#[pyfunction]
#[pyo3(signature = (data, method="mad", *, clip_sigma=None, iterations=None))]
pub fn estimate_noise<'py>(py: Python<'py>, data: Bound<'py, PyAny>, method: &str, clip_sigma: Option<f32>, iterations: Option<usize>) -> PyResult<Bound<'py, PyAny>> {
    let estimator: NoiseEstimator = match method {
        "mad" => Box::new(estimate_noise_std),
        "rms" => Box::new(estimate_noise_rms),
        "trimmed" => {
            let (Some(clip), Some(iters)) = (clip_sigma, iterations) else {
                return Err(PyValueError::new_err("method 'trimmed' needs clip_sigma and iterations"));
            };
            Box::new(move |x| estimate_noise_trimmed(x, clip, iters))
        }
        other => return Err(PyValueError::new_err(format!("method must be 'mad', 'rms' or 'trimmed', got '{other}'"))),
    };
    let input = F32Array::new(&data)?;
    let (channels, samples) = input.channels_samples(None)?;
    let x = input.slice();
    let sigmas: Vec<f32> = py.detach(|| x.chunks_exact(samples.max(1)).map(&*estimator).collect());
    to_numpy(py, sigmas, &[channels])
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySpikeEvent>()?;
    m.add_class::<PyDeduplicatedSpike>()?;
    m.add_function(wrap_pyfunction!(detect_spikes, m)?)?;
    m.add_function(wrap_pyfunction!(deduplicate_spikes, m)?)?;
    m.add_function(wrap_pyfunction!(estimate_noise, m)?)?;
    Ok(())
}
