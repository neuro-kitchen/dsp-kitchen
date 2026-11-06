//! `dspk.synapse.detection`: spike detectors on in-memory data (threshold, NEO, matched filter),
//! spatial de-duplication, and the streaming detector over a whole recording.

use dspk_ops::peaks::DistanceRule;
use dspk_ops::{Pipeline, PipelineStage};
use dspk_py_core::error::to_py;
use dspk_py_core::io::Recording;
use dspk_synapse::detection::{SpikePolarity, SpikeSpacing};
use dspk_synapse::{self as syn, SpikeEvent, StreamingDetectionConfig, StreamingDetector};
use numpy::{IntoPyArray, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3_stub_gen::derive::gen_stub_pyfunction;

use crate::io::{Probe, Sorting};

type Dict<'py> = Bound<'py, PyDict>;

fn invalid(e: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(e.to_string())
}

pub(crate) fn polarity(p: &str) -> PyResult<SpikePolarity> {
    match p {
        "negative" => Ok(SpikePolarity::Negative),
        "positive" => Ok(SpikePolarity::Positive),
        "both" => Ok(SpikePolarity::Both),
        other => Err(invalid(format!("polarity is negative, positive or both, got {other:?}"))),
    }
}

fn rule(r: &str) -> PyResult<DistanceRule> {
    match r {
        "locally_exclusive" => Ok(DistanceRule::LocallyExclusive),
        "scipy" => Ok(DistanceRule::Scipy),
        other => Err(invalid(format!("spacing is locally_exclusive or scipy, got {other:?}"))),
    }
}

fn matrix(data: &PyReadonlyArray2<'_, f32>) -> (Vec<f32>, usize, usize) {
    let a = data.as_array();
    (a.iter().copied().collect(), a.nrows(), a.ncols())
}

fn events<'py>(py: Python<'py>, mut ev: Vec<SpikeEvent>) -> PyResult<Dict<'py>> {
    ev.sort_by_key(|e| (e.sample_index, e.channel_id));
    let out = PyDict::new(py);
    out.set_item("samples", ev.iter().map(|e| e.sample_index).collect::<Vec<_>>().into_pyarray(py))?;
    out.set_item("channels", ev.iter().map(|e| e.channel_id as u32).collect::<Vec<_>>().into_pyarray(py))?;
    out.set_item("amplitudes", ev.iter().map(|e| e.peak_amplitude_uv).collect::<Vec<_>>().into_pyarray(py))?;
    Ok(out)
}

/// Threshold crossings of every channel of `data` `(channels, samples)`: local extrema of
/// `polarity` beyond `threshold · σ` (σ per channel: `sigmas`, else the median-based noise
/// estimate), at least `refractory_samples + 1` apart per channel. `samples`, `channels`,
/// `amplitudes` arrays, by time.
#[gen_stub_pyfunction(module = "dspk._native.synapse.detection")]
#[pyfunction]
#[pyo3(signature = (data, threshold, polarity, refractory_samples, spacing, sigmas=None))]
pub fn threshold<'py>(py: Python<'py>, data: PyReadonlyArray2<'_, f32>, threshold: f32, polarity: &str, refractory_samples: usize, spacing: &str, sigmas: Option<PyReadonlyArray1<'_, f32>>) -> PyResult<Dict<'py>> {
    let (x, c, n) = matrix(&data);
    let (pol, sp) = (self::polarity(polarity)?, SpikeSpacing { refractory_samples, rule: rule(spacing)? });
    let ev = match sigmas {
        Some(s) => {
            let s = s.as_slice()?.to_vec();
            if s.len() != c {
                return Err(invalid(format!("{} sigmas for {c} channels", s.len())));
            }
            py.detach(|| syn::detect_spikes_with_sigma(&x, c, n, &s, threshold, pol, sp))
        }
        None => py.detach(|| syn::detect_spikes_multichannel(&x, c, n, threshold, pol, sp)),
    };
    events(py, ev)
}

/// Teager–Kaiser energy (NEO) detection: energy peaks beyond `threshold · σ(energy)` per channel.
#[gen_stub_pyfunction(module = "dspk._native.synapse.detection")]
#[pyfunction]
pub fn neo<'py>(py: Python<'py>, data: PyReadonlyArray2<'_, f32>, threshold: f32, refractory_samples: usize, spacing: &str) -> PyResult<Dict<'py>> {
    let (x, c, n) = matrix(&data);
    let sp = SpikeSpacing { refractory_samples, rule: rule(spacing)? };
    let ev = py.detach(|| syn::detect_spikes_neo(&x, c, n, threshold, sp));
    events(py, ev)
}

/// Matched-filter detection: correlation with `prototype` peaking beyond `threshold · σ(score)`
/// where the signal is negative.
#[gen_stub_pyfunction(module = "dspk._native.synapse.detection")]
#[pyfunction]
pub fn matched_filter<'py>(py: Python<'py>, data: PyReadonlyArray2<'_, f32>, prototype: PyReadonlyArray1<'_, f32>, threshold: f32, refractory_samples: usize, spacing: &str) -> PyResult<Dict<'py>> {
    let (x, c, n) = matrix(&data);
    let proto = prototype.as_slice()?.to_vec();
    let sp = SpikeSpacing { refractory_samples, rule: rule(spacing)? };
    let ev = py.detach(|| syn::detect_spikes_matched_filter(&x, c, n, &proto, threshold, sp));
    events(py, ev)
}

/// A zero-mean, unit-norm biphasic spike prototype of `samples` samples (trough first).
#[gen_stub_pyfunction(module = "dspk._native.synapse.detection")]
#[pyfunction]
pub fn prototype<'py>(py: Python<'py>, samples: usize) -> Bound<'py, numpy::PyArray1<f32>> {
    syn::canonical_biphasic_prototype(samples).into_pyarray(py)
}

/// Detection height per channel: `threshold · σ`, `inf` where σ is not positive.
#[gen_stub_pyfunction(module = "dspk._native.synapse.detection")]
#[pyfunction]
pub fn heights<'py>(py: Python<'py>, sigmas: PyReadonlyArray1<'_, f32>, threshold: f32) -> PyResult<Bound<'py, numpy::PyArray1<f32>>> {
    Ok(syn::detection_heights(sigmas.as_slice()?, threshold).into_pyarray(py))
}

/// One spike per event seen on several neighbouring channels ("locally exclusive": the largest
/// within `radius_um` and `window_samples` stays). `samples`, `channels` (primary),
/// `amplitudes`, and `participating` (channel lists).
#[gen_stub_pyfunction(module = "dspk._native.synapse.detection")]
#[pyfunction]
pub fn deduplicate<'py>(py: Python<'py>, samples: PyReadonlyArray1<'_, u64>, channels: PyReadonlyArray1<'_, u32>, amplitudes: PyReadonlyArray1<'_, f32>, probe: &Probe, radius_um: f32, window_samples: u64) -> PyResult<Dict<'py>> {
    let (s, c, a) = (samples.as_slice()?, channels.as_slice()?, amplitudes.as_slice()?);
    if s.len() != c.len() || s.len() != a.len() {
        return Err(invalid("samples, channels and amplitudes differ in length"));
    }
    let ev: Vec<SpikeEvent> = (0..s.len()).map(|i| SpikeEvent { channel_id: c[i] as usize, sample_index: s[i], peak_amplitude_uv: a[i] }).collect();
    let d = py.detach(|| syn::deduplicate_spikes_spatial(&ev, &probe.inner, radius_um, window_samples));
    let out = PyDict::new(py);
    out.set_item("samples", d.iter().map(|e| e.sample_index).collect::<Vec<_>>().into_pyarray(py))?;
    out.set_item("channels", d.iter().map(|e| e.primary_channel as u32).collect::<Vec<_>>().into_pyarray(py))?;
    out.set_item("amplitudes", d.iter().map(|e| e.peak_amplitude_uv).collect::<Vec<_>>().into_pyarray(py))?;
    out.set_item("participating", d.iter().map(|e| e.participating_channels.clone()).collect::<Vec<_>>())?;
    Ok(out)
}

/// The streaming detector over a whole recording, read in place in batches: optional CAR, a
/// zero-phase band-pass, noise calibrated on spread-out chunks, threshold crossings, spatial
/// de-duplication, snippets and per-channel templates — as `dspk detect`. Returns the spikes as a
/// [`Sorting`] (one unit per channel) and a summary (`sigmas_uv`, `raw_crossings`, `spikes`,
/// `channel_counts`).
#[gen_stub_pyfunction(module = "dspk._native.synapse.detection")]
#[pyfunction]
#[pyo3(signature = (recording, probe, *, band, car, threshold, refractory_ms, polarity, spacing, radius_um, k_neighbors, pre_ms, post_ms, sinc_shift, batch_s, calibration_s, calibration_chunks, runtime))]
#[allow(clippy::too_many_arguments)]
pub fn detect_recording<'py>(
    py: Python<'py>,
    recording: &Recording,
    probe: &Probe,
    band: (f64, f64),
    car: bool,
    threshold: f32,
    refractory_ms: f64,
    polarity: &str,
    spacing: &str,
    radius_um: f32,
    k_neighbors: usize,
    pre_ms: f64,
    post_ms: f64,
    sinc_shift: bool,
    batch_s: f64,
    calibration_s: f64,
    calibration_chunks: usize,
    runtime: Option<&str>,
) -> PyResult<(Sorting, Dict<'py>)> {
    let config = StreamingDetectionConfig {
        batch_duration_sec: batch_s,
        calibration_duration_sec: calibration_s,
        calibration_chunks,
        threshold_factor: threshold,
        refractory_ms,
        polarity: self::polarity(polarity)?,
        distance_rule: rule(spacing)?,
        spatial_radius_um: radius_um,
        k_neighbors,
        pre_ms,
        post_ms,
        apply_sinc_shift: sinc_shift,
    };
    let mut pipeline = Pipeline::new();
    if car {
        pipeline.add(PipelineStage::CommonAverageReference);
    }
    pipeline.add(PipelineStage::bandpass(band.0, band.1));
    let target = dspk_py_core::runtime::target(runtime)?;
    let source = recording.source();
    let layout = probe.inner.clone();
    let result = py.detach(|| StreamingDetector::new(config).run_with(target, source.as_ref(), &pipeline, &layout)).map_err(to_py)?;
    let summary = PyDict::new(py);
    summary.set_item("sigmas_uv", result.channel_sigmas_uv.clone().into_pyarray(py))?;
    summary.set_item("raw_crossings", result.total_raw_crossings)?;
    summary.set_item("spikes", result.total_dedup_spikes)?;
    summary.set_item("channel_counts", result.channel_spike_counts.clone().into_pyarray(py))?;
    let sorting = Sorting { inner: result.to_sorting_output("dspk-detect", Some(probe.inner.clone())) };
    Ok((sorting, summary))
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let d = StreamingDetectionConfig::default();
    m.add("DEFAULT_THRESHOLD", d.threshold_factor)?;
    m.add("DEFAULT_REFRACTORY_MS", d.refractory_ms)?;
    m.add("DEFAULT_RADIUS_UM", d.spatial_radius_um)?;
    m.add("DEFAULT_K_NEIGHBORS", d.k_neighbors)?;
    m.add("DEFAULT_PRE_MS", d.pre_ms)?;
    m.add("DEFAULT_POST_MS", d.post_ms)?;
    m.add("DEFAULT_BATCH_S", d.batch_duration_sec)?;
    m.add("DEFAULT_CALIBRATION_S", d.calibration_duration_sec)?;
    m.add("DEFAULT_CALIBRATION_CHUNKS", d.calibration_chunks)?;
    m.add("DEFAULT_NEO_THRESHOLD", syn::NeoSpikeDetector::default().threshold_factor)?;
    for f in [
        wrap_pyfunction!(threshold, m)?,
        wrap_pyfunction!(neo, m)?,
        wrap_pyfunction!(matched_filter, m)?,
        wrap_pyfunction!(prototype, m)?,
        wrap_pyfunction!(heights, m)?,
        wrap_pyfunction!(deduplicate, m)?,
        wrap_pyfunction!(detect_recording, m)?,
    ] {
        m.add_function(f)?;
    }
    Ok(())
}

pyo3_stub_gen::module_variable!("dspk._native.synapse.detection", "DEFAULT_THRESHOLD", f32);
pyo3_stub_gen::module_variable!("dspk._native.synapse.detection", "DEFAULT_REFRACTORY_MS", f64);
pyo3_stub_gen::module_variable!("dspk._native.synapse.detection", "DEFAULT_RADIUS_UM", f32);
pyo3_stub_gen::module_variable!("dspk._native.synapse.detection", "DEFAULT_K_NEIGHBORS", usize);
pyo3_stub_gen::module_variable!("dspk._native.synapse.detection", "DEFAULT_PRE_MS", f64);
pyo3_stub_gen::module_variable!("dspk._native.synapse.detection", "DEFAULT_POST_MS", f64);
pyo3_stub_gen::module_variable!("dspk._native.synapse.detection", "DEFAULT_BATCH_S", f64);
pyo3_stub_gen::module_variable!("dspk._native.synapse.detection", "DEFAULT_CALIBRATION_S", f64);
pyo3_stub_gen::module_variable!("dspk._native.synapse.detection", "DEFAULT_CALIBRATION_CHUNKS", usize);
pyo3_stub_gen::module_variable!("dspk._native.synapse.detection", "DEFAULT_NEO_THRESHOLD", f32);
