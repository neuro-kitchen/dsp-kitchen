//! `dspk.ops.resample`: sample-rate changes on the device (`dspk_ops::resampler`), with
//! `scipy.signal.decimate` / `resample_poly` semantics.

use dspk_ops::filter::FilterError;
use dspk_ops::resampler::decimate::{DECIMATE_DEFAULT, DECIMATE_FIR_TAPS_PER_FACTOR, DECIMATE_IIR_CUTOFF};
use dspk_ops::resampler::poly::{RESAMPLE_POLY_DEFAULT_EDGE, RESAMPLE_POLY_HALF_LEN_PER_RATE, RESAMPLE_POLY_WINDOW};
use dspk_ops::resampler::{decimate as dev_decimate, decimate_len, resample_poly as dev_resample_poly, resample_poly_len, DecimateFilter, FirWindow, ResampleFilter};
use numpy::{PyArray2, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_stub_gen::derive::gen_stub_pyfunction;

use super::device;
use super::filter::{edge, edge_name};

/// Down-samples `data` `(channels, samples)` by `q` behind an anti-alias filter, as
/// `scipy.signal.decimate(x, q, ftype=..., zero_phase=True)`.
///
/// `ftype="iir"`: Chebyshev type I of `order` with `ripple_db` at `DECIMATE_IIR_CUTOFF` of the
/// new Nyquist, zero phase (samples near the ends can differ slightly from scipy, whose padding
/// length differs). `ftype="fir"`: Hamming `firwin` of `taps_per_factor · q + 1` taps through
/// `resample_poly`.
#[gen_stub_pyfunction(module = "dspk._native.ops.resample")]
#[pyfunction]
#[pyo3(signature = (data, q, *, ftype, order, ripple_db, taps_per_factor, runtime))]
#[allow(clippy::too_many_arguments)]
pub fn decimate<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'_, f32>,
    q: usize,
    ftype: &str,
    order: usize,
    ripple_db: f64,
    taps_per_factor: usize,
    runtime: Option<&str>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    if q == 0 {
        return Err(PyValueError::new_err("q must be at least 1"));
    }
    let filter = match ftype {
        "iir" => DecimateFilter::Iir { order, ripple_db },
        "fir" => DecimateFilter::Fir { taps_per_factor },
        other => return Err(PyValueError::new_err(format!("unknown ftype {other:?} (iir, fir)"))),
    };
    let out_samples = decimate_len(data.as_array().dim().1, q);
    device::run_resized(py, data, runtime, out_samples, move |client, input, output, channels, samples| {
        dev_decimate::<f32>(client, input, output, channels, samples, q, filter).map(|_| ()).map_err(|e: FilterError| e.to_string())
    })
}

/// Resamples `data` `(channels, samples)` by `up / down` with a polyphase FIR, as
/// `scipy.signal.resample_poly` (output sample 0 on input sample 0; Kaiser `beta` window,
/// `RESAMPLE_POLY_HALF_LEN_PER_RATE · max(up, down)` half-length; samples past the ends from `edge`).
#[gen_stub_pyfunction(module = "dspk._native.ops.resample")]
#[pyfunction]
#[pyo3(signature = (data, up, down, *, beta, edge, runtime))]
#[allow(clippy::too_many_arguments)]
pub fn resample_poly<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'_, f32>,
    up: usize,
    down: usize,
    beta: f64,
    edge: &str,
    runtime: Option<&str>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    if up == 0 || down == 0 {
        return Err(PyValueError::new_err("up and down must be at least 1"));
    }
    let edge = self::edge(edge)?;
    let filter = ResampleFilter::Window(FirWindow::Kaiser { beta });
    let out_samples = resample_poly_len(data.as_array().dim().1, up, down);
    device::run_resized(py, data, runtime, out_samples, move |client, input, output, channels, samples| {
        dev_resample_poly::<f32>(client, input, output, channels, samples, up, down, &filter, edge);
        Ok(())
    })
}

pyo3_stub_gen::module_variable!("dspk._native.ops.resample", "DECIMATE_IIR_CUTOFF", f64);
pyo3_stub_gen::module_variable!("dspk._native.ops.resample", "DECIMATE_DEFAULT_ORDER", usize);
pyo3_stub_gen::module_variable!("dspk._native.ops.resample", "DECIMATE_DEFAULT_RIPPLE_DB", f64);
pyo3_stub_gen::module_variable!("dspk._native.ops.resample", "RESAMPLE_POLY_BETA", f64);
pyo3_stub_gen::module_variable!("dspk._native.ops.resample", "DECIMATE_FIR_TAPS_PER_FACTOR", usize);
pyo3_stub_gen::module_variable!("dspk._native.ops.resample", "RESAMPLE_POLY_HALF_LEN_PER_RATE", usize);
pyo3_stub_gen::module_variable!("dspk._native.ops.resample", "RESAMPLE_POLY_DEFAULT_EDGE", String);

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(decimate, m)?)?;
    m.add_function(wrap_pyfunction!(resample_poly, m)?)?;
    m.add("DECIMATE_IIR_CUTOFF", DECIMATE_IIR_CUTOFF)?;
    // Taken from the dspk-ops defaults, not retyped
    let DecimateFilter::Iir { order, ripple_db } = DECIMATE_DEFAULT else { unreachable!("decimate's default is the IIR") };
    m.add("DECIMATE_DEFAULT_ORDER", order)?;
    m.add("DECIMATE_DEFAULT_RIPPLE_DB", ripple_db)?;
    let FirWindow::Kaiser { beta } = RESAMPLE_POLY_WINDOW else { unreachable!("resample_poly's default window is Kaiser") };
    m.add("RESAMPLE_POLY_BETA", beta)?;
    m.add("DECIMATE_FIR_TAPS_PER_FACTOR", DECIMATE_FIR_TAPS_PER_FACTOR)?;
    m.add("RESAMPLE_POLY_HALF_LEN_PER_RATE", RESAMPLE_POLY_HALF_LEN_PER_RATE)?;
    m.add("RESAMPLE_POLY_DEFAULT_EDGE", edge_name(RESAMPLE_POLY_DEFAULT_EDGE))?;
    Ok(())
}
