//! `dspk.ops.math`: per-channel statistics on the device and window designs (`dspk_ops::math`).
//!
//! Only device implementations are bound; the host-only statistics of dspk-ops (percentile, IQR,
//! peak-to-peak, trimmed noise, histogram, cross-correlation) follow once they run on the device.

use crate::ops::device::DeviceInput;
use dspk_ops::core::buffer;
use dspk_ops::math::stats::MAD_TO_SIGMA;
use dspk_ops::math::{execute_channel_mean_std, execute_channel_noise_std, gaussian_window, hamming_window, hann_window, kaiser_window};
use numpy::{IntoPyArray, PyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_stub_gen::derive::gen_stub_pyfunction;

use super::device;

/// Per-channel mean and population standard deviation of `data` `(channels, samples)`.
/// Two per-channel vectors (means and standard deviations).
type PerChannel<'py> = (Bound<'py, PyArray1<f32>>, Bound<'py, PyArray1<f32>>);

#[gen_stub_pyfunction(module = "dspk._native.ops.math")]
#[pyfunction]
#[pyo3(signature = (data, *, runtime))]
pub fn channel_mean_std<'py>(py: Python<'py>, data: DeviceInput, runtime: Option<&str>) -> PyResult<PerChannel<'py>> {
    let (mean, std) = device::with_input(py, data, runtime, |client, input, channels, samples| {
        let (mean, std) = (buffer::empty::<f32>(client, channels), buffer::empty::<f32>(client, channels));
        execute_channel_mean_std::<f32>(client, input, &mean, &std, channels, samples);
        Ok((buffer::download_prefix::<f32>(client, mean, channels), buffer::download_prefix::<f32>(client, std, channels)))
    })?;
    Ok((mean.into_pyarray(py), std.into_pyarray(py)))
}

/// Robust noise level per channel: `median(|x|) / MAD_TO_SIGMA` (the median is the element at
/// index `samples // 2` of the sorted magnitudes).
#[gen_stub_pyfunction(module = "dspk._native.ops.math")]
#[pyfunction]
#[pyo3(signature = (data, *, runtime))]
pub fn channel_noise_std<'py>(py: Python<'py>, data: DeviceInput, runtime: Option<&str>) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let noise = device::with_input(py, data, runtime, |client, input, channels, samples| {
        Ok(execute_channel_noise_std::<f32>(client, input, channels, samples, 0..samples))
    })?;
    Ok(noise.into_pyarray(py))
}

/// Window of `length` values: `"hann"`, `"hamming"` (symmetric), `"gaussian"` (needs `sigma`),
/// `"kaiser"` (needs `beta`).
#[gen_stub_pyfunction(module = "dspk._native.ops.math")]
#[pyfunction]
#[pyo3(signature = (name, length, *, sigma, beta))]
pub fn window<'py>(py: Python<'py>, name: &str, length: usize, sigma: Option<f32>, beta: Option<f64>) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let values: Vec<f64> = match (name, sigma, beta) {
        ("hann", None, None) => hann_window(length).into_iter().map(f64::from).collect(),
        ("hamming", None, None) => hamming_window(length).into_iter().map(f64::from).collect(),
        ("gaussian", Some(sigma), None) => gaussian_window(length, sigma).into_iter().map(f64::from).collect(),
        ("kaiser", None, Some(beta)) => kaiser_window(length, beta),
        ("gaussian", None, _) => return Err(PyValueError::new_err("window \"gaussian\" needs sigma")),
        ("kaiser", _, None) => return Err(PyValueError::new_err("window \"kaiser\" needs beta")),
        ("hann" | "hamming" | "gaussian" | "kaiser", _, _) => return Err(PyValueError::new_err(format!("sigma applies to gaussian and beta to kaiser only, not {name}"))),
        (other, _, _) => return Err(PyValueError::new_err(format!("unknown window {other:?} (hann, hamming, gaussian, kaiser)"))),
    };
    Ok(values.into_pyarray(py))
}

pyo3_stub_gen::module_variable!("dspk._native.ops.math", "MAD_TO_SIGMA", f32);

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(channel_mean_std, m)?)?;
    m.add_function(wrap_pyfunction!(channel_noise_std, m)?)?;
    m.add_function(wrap_pyfunction!(window, m)?)?;
    m.add("MAD_TO_SIGMA", MAD_TO_SIGMA)?;
    Ok(())
}
