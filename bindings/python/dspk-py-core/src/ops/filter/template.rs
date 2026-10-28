//! Template subtraction on the device (`dspk_ops::filter::TemplateFilter::apply_device`).

use dspk_ops::filter::template::DEFAULT_MAX_LAG;
use dspk_ops::filter::TemplateFilter;
use numpy::{PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_stub_gen::derive::gen_stub_pyfunction;

use crate::ops::device;

/// Subtracts `template` `(channels, template_samples)` from `data` `(channels, samples)` at every
/// event (sample indices). The template's sample `center_offset` sits on the event; each event is
/// aligned by cross-correlation within `±max_lag` samples and, with `dynamic_scaling`, scaled by
/// least squares. Overlapping events are subtracted in their order, as one after another.
#[gen_stub_pyfunction(module = "dspk._native.ops.filter")]
#[pyfunction]
#[pyo3(signature = (data, template, events, center_offset, *, max_lag, dynamic_scaling, runtime))]
#[allow(clippy::too_many_arguments)]
pub fn subtract_template<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'_, f32>,
    template: PyReadonlyArray2<'_, f32>,
    events: PyReadonlyArray1<'_, u64>,
    center_offset: usize,
    max_lag: usize,
    dynamic_scaling: bool,
    runtime: Option<&str>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let (t_channels, t_samples) = template.as_array().dim();
    let channels = data.as_array().dim().0;
    if t_channels != channels {
        return Err(PyValueError::new_err(format!("template has {t_channels} channels, data has {channels}")));
    }
    if center_offset >= t_samples {
        return Err(PyValueError::new_err(format!("center_offset {center_offset} lies outside the template's {t_samples} samples")));
    }
    let filter = TemplateFilter::new_multichannel(template.as_array().iter().copied().collect(), t_channels, t_samples, center_offset)
        .with_max_lag(max_lag)
        .with_dynamic_scaling(dynamic_scaling);
    let events: Vec<u64> = events.as_array().iter().copied().collect();
    device::run_in_place(py, data, runtime, move |client, signal, channels, samples| {
        filter.apply_device::<f32>(client, signal, channels, samples, &events);
        Ok(())
    })
}

pyo3_stub_gen::module_variable!("dspk._native.ops.filter", "TEMPLATE_DEFAULT_MAX_LAG", usize);

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(subtract_template, m)?)?;
    m.add("TEMPLATE_DEFAULT_MAX_LAG", DEFAULT_MAX_LAG)?;
    Ok(())
}
