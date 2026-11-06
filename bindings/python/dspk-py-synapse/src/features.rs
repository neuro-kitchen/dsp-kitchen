//! `dspk.synapse.features` and `dspk.synapse.extraction`: snippets around spikes, waveform
//! morphology, wavelet coefficients, HD-EMG conduction velocity.

use dspk_synapse::{self as syn, DeduplicatedSpike, WaveformSnippet};
use numpy::ndarray::{Array2, Array3};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3_stub_gen::derive::gen_stub_pyfunction;

use crate::io::Probe;

type Dict<'py> = Bound<'py, PyDict>;

fn invalid(e: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// Snippets `[center − pre, center + post)` around each spike on its `k` nearest channels (by
/// `probe`), from `data` `(channels, samples)`; `sinc_shift` realigns each to its sub-sample trough.
/// Spikes too near an edge are skipped. `waveforms` `(spikes, k, pre + post)`, `channels`
/// `(spikes, k)`, `centers`, `offsets` (sub-sample shift), `kept` (indices of the input spikes).
#[gen_stub_pyfunction(module = "dspk._native.synapse.extraction")]
#[pyfunction]
pub fn snippets<'py>(py: Python<'py>, data: PyReadonlyArray2<'_, f32>, samples: PyReadonlyArray1<'_, u64>, channels: PyReadonlyArray1<'_, u32>, probe: &Probe, k: usize, pre: usize, post: usize, sinc_shift: bool) -> PyResult<Dict<'py>> {
    let a = data.as_array();
    let (c, n) = (a.nrows(), a.ncols());
    let x: Vec<f32> = a.iter().copied().collect();
    let (s, ch) = (samples.as_slice()?, channels.as_slice()?);
    if s.len() != ch.len() {
        return Err(invalid("samples and channels differ in length"));
    }
    let spikes: Vec<DeduplicatedSpike> = (0..s.len())
        .map(|i| DeduplicatedSpike { primary_channel: ch[i] as usize, sample_index: s[i], peak_amplitude_uv: 0.0, participating_channels: vec![ch[i] as usize] })
        .collect();
    let options = syn::extraction::SnippetOptions { k_neighbors: k, pre_samples: pre, post_samples: post, sinc_shift, subtract_mean: false };
    let kept: Vec<usize> = (0..s.len()).filter(|&i| syn::extraction::snippet_fits(s[i] as usize, n, &options)).collect();
    let chosen: Vec<DeduplicatedSpike> = kept.iter().map(|&i| spikes[i].clone()).collect();
    let snips = py.detach(|| syn::extract_snippets(&x, c, n, &chosen, &probe.inner, &options));
    let width = pre + post;
    let kk = snips.first().map_or(k, |sn| sn.num_channels());
    let mut wav = Vec::with_capacity(snips.len() * kk * width);
    let mut chans = Vec::with_capacity(snips.len() * kk);
    for sn in &snips {
        wav.extend_from_slice(&sn.waveform);
        chans.extend(sn.channel_ids.iter().map(|&c| c as u32));
    }
    let out = PyDict::new(py);
    out.set_item("waveforms", Array3::from_shape_vec((snips.len(), kk, width), wav).map_err(invalid)?.into_pyarray(py))?;
    out.set_item("channels", Array2::from_shape_vec((snips.len(), kk), chans).map_err(invalid)?.into_pyarray(py))?;
    out.set_item("centers", snips.iter().map(|sn| sn.center_sample).collect::<Vec<_>>().into_pyarray(py))?;
    out.set_item("offsets", snips.iter().map(|sn| sn.subsample_offset).collect::<Vec<_>>().into_pyarray(py))?;
    out.set_item("kept", kept.iter().map(|&i| i as u64).collect::<Vec<_>>().into_pyarray(py))?;
    Ok(out)
}

/// Waveform shape of each row of `waveforms` `(spikes, samples)` (one channel):
/// `trough_uv`, `trough_to_peak_samples`, `half_width_samples`, `repolarization_slope` (NaN where
/// a waveform has no trough).
#[gen_stub_pyfunction(module = "dspk._native.synapse.features")]
#[pyfunction]
pub fn morphology<'py>(py: Python<'py>, waveforms: PyReadonlyArray2<'_, f32>) -> PyResult<Dict<'py>> {
    let a = waveforms.as_array();
    let rows: Vec<Option<syn::SpikeMorphology>> = a
        .rows()
        .into_iter()
        .map(|r| {
            let snippet = WaveformSnippet { primary_channel: 0, center_sample: 0, subsample_offset: 0.0, channel_ids: vec![0], num_samples: r.len(), waveform: r.to_vec() };
            syn::compute_morphology(&snippet)
        })
        .collect();
    let col = |f: fn(&syn::SpikeMorphology) -> f32| rows.iter().map(|m| m.as_ref().map_or(f32::NAN, f)).collect::<Vec<_>>();
    let out = PyDict::new(py);
    out.set_item("trough_uv", col(|m| m.trough_amplitude_uv).into_pyarray(py))?;
    out.set_item("trough_to_peak_samples", col(|m| m.trough_to_peak_samples).into_pyarray(py))?;
    out.set_item("half_width_samples", col(|m| m.half_width_samples).into_pyarray(py))?;
    out.set_item("repolarization_slope", col(|m| m.repolarization_slope).into_pyarray(py))?;
    Ok(out)
}

/// Multi-level Haar wavelet transform of `signal` (`levels` levels; coefficients concatenated,
/// approximation first).
#[gen_stub_pyfunction(module = "dspk._native.synapse.features")]
#[pyfunction]
pub fn haar_wavelet<'py>(py: Python<'py>, signal: PyReadonlyArray1<'_, f32>, levels: usize) -> PyResult<Bound<'py, PyArray1<f32>>> {
    Ok(syn::compute_haar_dwt(signal.as_slice()?, levels).into_pyarray(py))
}

/// Muscle-fibre conduction velocity from an HD-EMG template `(rows · cols, samples)` laid out
/// row-major on a grid of `pitch_um`: `velocity_m_per_s`, `mean_delay_ms`, `correlation`.
#[gen_stub_pyfunction(module = "dspk._native.synapse.features")]
#[pyfunction]
pub fn conduction_velocity<'py>(py: Python<'py>, template: PyReadonlyArray2<'_, f32>, rows: usize, cols: usize, pitch_um: f32, sample_rate: f64) -> PyResult<Dict<'py>> {
    let a = template.as_array();
    if a.nrows() != rows * cols {
        return Err(invalid(format!("{} channels for a {rows} × {cols} grid", a.nrows())));
    }
    let x: Vec<f32> = a.iter().copied().collect();
    let e = syn::features::estimate_hdemg_conduction_velocity(&x, rows, cols, a.ncols(), sample_rate, pitch_um);
    let out = PyDict::new(py);
    out.set_item("velocity_m_per_s", e.velocity_m_per_s)?;
    out.set_item("mean_delay_ms", e.mean_delay_ms)?;
    out.set_item("correlation", e.correlation_r)?;
    Ok(out)
}

pub(crate) fn register_extraction(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(snippets, m)?)?;
    Ok(())
}

pub(crate) fn register_features(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(morphology, m)?)?;
    m.add_function(wrap_pyfunction!(haar_wavelet, m)?)?;
    m.add_function(wrap_pyfunction!(conduction_velocity, m)?)?;
    Ok(())
}
