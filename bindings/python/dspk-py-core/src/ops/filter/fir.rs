//! FIR filters and Gaussian smoothing on the device (`dspk_ops::filter::fir`), and `firwin` design.

use dspk_ops::core::buffer;
use dspk_ops::filter::fir::gaussian::GAUSSIAN_TRUNCATE;
use dspk_ops::filter::{execute_fir, execute_fir_centered, execute_gaussian_smooth, FIR_DEFAULT_EDGE, GAUSSIAN_DEFAULT_EDGE};
use dspk_ops::resampler::{firwin as design_firwin, FirWindow};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_stub_gen::derive::gen_stub_pyfunction;

use super::{edge, edge_name};
use crate::ops::device;

/// Low-pass FIR taps (`scipy.signal.firwin` with one cutoff), unit DC gain.
///
/// `cutoff` is in Hz when `sample_rate` is given, else a fraction of Nyquist in `(0, 1]`.
/// `window`: `"hamming"` (firwin's default) or `"kaiser"` (needs `beta`).
#[gen_stub_pyfunction(module = "dspk._native.ops.filter")]
#[pyfunction]
#[pyo3(signature = (num_taps, cutoff, *, sample_rate, window, beta))]
pub fn firwin<'py>(py: Python<'py>, num_taps: usize, cutoff: f64, sample_rate: Option<f64>, window: &str, beta: Option<f64>) -> PyResult<Bound<'py, PyArray1<f64>>> {
    if num_taps == 0 {
        return Err(PyValueError::new_err("num_taps must be at least 1"));
    }
    let fraction = match sample_rate {
        Some(fs) if fs > 0.0 => cutoff / (fs / 2.0),
        Some(fs) => return Err(PyValueError::new_err(format!("sample_rate must be positive, got {fs}"))),
        None => cutoff,
    };
    if !(fraction > 0.0 && fraction <= 1.0) {
        return Err(PyValueError::new_err(format!("cutoff must lie in (0, Nyquist], got {fraction} of Nyquist")));
    }
    let window = match (window, beta) {
        ("hamming", None) => FirWindow::Hamming,
        ("kaiser", Some(beta)) => FirWindow::Kaiser { beta },
        ("kaiser", None) => return Err(PyValueError::new_err("window=\"kaiser\" needs beta")),
        ("hamming", Some(_)) => return Err(PyValueError::new_err("beta applies to window=\"kaiser\" only")),
        (other, _) => return Err(PyValueError::new_err(format!("unknown window {other:?} (hamming, kaiser)"))),
    };
    Ok(design_firwin(num_taps, fraction, window).into_pyarray(py))
}

/// FIR filtering of `data` `(channels, samples)` with `taps` on the device.
///
/// `centered=False`: causal, `y[t] = Σ taps[k] · x[t − k]` (`scipy.signal.lfilter(taps, 1, x)` with
/// `edge="zeros"`). `centered=True`: odd number of taps centred on each sample (zero phase for
/// symmetric taps). Samples past the ends come from `edge`.
#[gen_stub_pyfunction(module = "dspk._native.ops.filter")]
#[pyfunction]
#[pyo3(signature = (data, taps, *, centered, edge, runtime))]
pub fn fir<'py>(py: Python<'py>, data: PyReadonlyArray2<'_, f32>, taps: PyReadonlyArray1<'_, f32>, centered: bool, edge: &str, runtime: Option<&str>) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let taps: Vec<f32> = taps.as_array().iter().copied().collect();
    if taps.is_empty() {
        return Err(PyValueError::new_err("taps is empty"));
    }
    if centered && taps.len() % 2 == 0 {
        return Err(PyValueError::new_err(format!("centered FIR needs an odd number of taps, got {}", taps.len())));
    }
    let edge = self::edge(edge)?;
    device::run(py, data, runtime, move |client, input, output, channels, samples| {
        let taps_h = buffer::upload(client, &taps);
        if centered {
            execute_fir_centered::<f32>(client, input, output, &taps_h, channels, samples, taps.len() / 2, edge);
        } else {
            execute_fir::<f32>(client, input, output, &taps_h, channels, samples, taps.len(), edge);
        }
        Ok(())
    })
}

/// Zero-phase Gaussian smoothing of standard deviation `sigma_samples`, cut at `GAUSSIAN_TRUNCATE`
/// σ (`scipy.ndimage.gaussian_filter1d` with the same `truncate` and `mode` = `edge`).
#[gen_stub_pyfunction(module = "dspk._native.ops.filter")]
#[pyfunction]
#[pyo3(signature = (data, sigma_samples, *, edge, runtime))]
pub fn gaussian_smooth<'py>(py: Python<'py>, data: PyReadonlyArray2<'_, f32>, sigma_samples: f32, edge: &str, runtime: Option<&str>) -> PyResult<Bound<'py, PyArray2<f32>>> {
    if !(sigma_samples > 0.0) {
        return Err(PyValueError::new_err(format!("sigma_samples must be positive, got {sigma_samples}")));
    }
    let edge = self::edge(edge)?;
    device::run(py, data, runtime, move |client, input, output, channels, samples| {
        execute_gaussian_smooth::<f32>(client, input, output, channels, samples, sigma_samples, edge);
        Ok(())
    })
}

pyo3_stub_gen::module_variable!("dspk._native.ops.filter", "FIR_DEFAULT_EDGE", String);
pyo3_stub_gen::module_variable!("dspk._native.ops.filter", "GAUSSIAN_DEFAULT_EDGE", String);
pyo3_stub_gen::module_variable!("dspk._native.ops.filter", "GAUSSIAN_TRUNCATE", f32);

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(firwin, m)?)?;
    m.add_function(wrap_pyfunction!(fir, m)?)?;
    m.add_function(wrap_pyfunction!(gaussian_smooth, m)?)?;
    m.add("FIR_DEFAULT_EDGE", edge_name(FIR_DEFAULT_EDGE))?;
    m.add("GAUSSIAN_DEFAULT_EDGE", edge_name(GAUSSIAN_DEFAULT_EDGE))?;
    m.add("GAUSSIAN_TRUNCATE", GAUSSIAN_TRUNCATE)?;
    Ok(())
}
