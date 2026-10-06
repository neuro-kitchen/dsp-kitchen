use pyo3::prelude::*;

use crate::array::{to_numpy, F32Array};
use dsp_base::filter::template::subtraction::DEFAULT_MAX_LAG;
use dsp_base::filter::template::TemplateFilter;

/// Python wrapper for TemplateFilter.
#[pyclass(name = "TemplateFilter", skip_from_py_object)]
#[derive(Clone)]
pub struct PyTemplateFilter {
    pub inner: TemplateFilter,
}

#[pymethods]
impl PyTemplateFilter {
    #[new]
    #[pyo3(signature = (template, center_offset=0, max_lag=DEFAULT_MAX_LAG, dynamic_scaling=true))]
    pub fn new<'py>(
        template: Bound<'py, PyAny>,
        center_offset: usize,
        max_lag: usize,
        dynamic_scaling: bool,
    ) -> PyResult<Self> {
        let input = F32Array::new(&template)?;
        let shape = input.shape().to_vec();
        let float_slice = input.slice();

        let filter = match shape.len() {
            1 => TemplateFilter::new_1d(float_slice.to_vec(), center_offset)
                .with_max_lag(max_lag)
                .with_dynamic_scaling(dynamic_scaling),
            2 => TemplateFilter::new_multichannel(
                float_slice.to_vec(),
                shape[0],
                shape[1],
                center_offset,
            )
            .with_max_lag(max_lag)
            .with_dynamic_scaling(dynamic_scaling),
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "Template must be 1D [samples] or 2D [channels, samples]",
                ));
            }
        };

        Ok(Self { inner: filter })
    }

    /// Applies template subtraction to data [samples] or [channels, samples] at event_indices.
    #[pyo3(signature = (data, event_indices))]
    pub fn apply<'py>(
        &self,
        py: Python<'py>,
        data: Bound<'py, PyAny>,
        event_indices: Vec<u64>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let input = F32Array::new(&data)?;
        let shape = input.shape().to_vec();
        let mut values = input.slice().to_vec();
        let filter = &self.inner;
        match shape[..] {
            [_] => {
                py.detach(|| filter.apply_1d(&mut values, &event_indices));
                to_numpy(py, values, &shape)
            }
            [channels, samples] => {
                py.detach(|| filter.apply_multichannel(&mut values, channels, samples, &event_indices));
                to_numpy(py, values, &shape)
            }
            _ => Err(pyo3::exceptions::PyValueError::new_err(
                "Data must be 1D [samples] or 2D [channels, samples]",
            )),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "TemplateFilter(channels={}, samples={}, center={}, max_lag={}, scaling={})",
            self.inner.template_channels,
            self.inner.template_samples,
            self.inner.center_offset,
            self.inner.max_lag,
            self.inner.dynamic_scaling
        )
    }
}

/// Direct function for template subtraction on 1D or 2D signals.
#[pyfunction]
#[pyo3(signature = (data, template, event_indices, center_offset=0, max_lag=DEFAULT_MAX_LAG, dynamic_scaling=true))]
pub fn subtract_template<'py>(
    py: Python<'py>,
    data: Bound<'py, PyAny>,
    template: Bound<'py, PyAny>,
    event_indices: Vec<u64>,
    center_offset: usize,
    max_lag: usize,
    dynamic_scaling: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let filter = PyTemplateFilter::new(template, center_offset, max_lag, dynamic_scaling)?;
    filter.apply(py, data, event_indices)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTemplateFilter>()?;
    m.add_function(wrap_pyfunction!(subtract_template, m)?)?;
    Ok(())
}
