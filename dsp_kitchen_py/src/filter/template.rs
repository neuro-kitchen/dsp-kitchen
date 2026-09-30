use pyo3::prelude::*;
use pyo3::types::PyBytes;
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
    #[pyo3(signature = (template, center_offset=0, max_lag=8, dynamic_scaling=true))]
    pub fn new<'py>(
        py: Python<'py>,
        template: Bound<'py, PyAny>,
        center_offset: usize,
        max_lag: usize,
        dynamic_scaling: bool,
    ) -> PyResult<Self> {
        let np = py.import("numpy")?;
        let arr = np.call_method1("ascontiguousarray", (template, "float32"))?;
        let shape: Vec<usize> = arr.getattr("shape")?.extract()?;
        let py_bytes = arr.call_method0("tobytes")?;
        let raw_bytes: &[u8] = py_bytes.extract()?;
        let float_slice: &[f32] = unsafe {
            std::slice::from_raw_parts(
                raw_bytes.as_ptr() as *const f32,
                raw_bytes.len() / std::mem::size_of::<f32>(),
            )
        };

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
        let np = py.import("numpy")?;
        let arr = np.call_method1("ascontiguousarray", (data, "float32"))?;
        let shape: Vec<usize> = arr.getattr("shape")?.extract()?;
        let py_bytes = arr.call_method0("tobytes")?;
        let raw_bytes: &[u8] = py_bytes.extract()?;
        let mut float_vec: Vec<f32> = unsafe {
            std::slice::from_raw_parts(
                raw_bytes.as_ptr() as *const f32,
                raw_bytes.len() / std::mem::size_of::<f32>(),
            )
        }
        .to_vec();

        match shape.len() {
            1 => {
                self.inner.apply_1d(&mut float_vec, &event_indices);
                let out_bytes = PyBytes::new(py, unsafe {
                    std::slice::from_raw_parts(
                        float_vec.as_ptr() as *const u8,
                        float_vec.len() * std::mem::size_of::<f32>(),
                    )
                });
                let flat = np.call_method1("frombuffer", (out_bytes, "float32"))?;
                Ok(flat)
            }
            2 => {
                let channels = shape[0];
                let samples = shape[1];
                self.inner.apply_multichannel(&mut float_vec, channels, samples, &event_indices);
                let out_bytes = PyBytes::new(py, unsafe {
                    std::slice::from_raw_parts(
                        float_vec.as_ptr() as *const u8,
                        float_vec.len() * std::mem::size_of::<f32>(),
                    )
                });
                let flat = np.call_method1("frombuffer", (out_bytes, "float32"))?;
                let reshaped = flat.call_method1("reshape", ((channels, samples),))?;
                Ok(reshaped)
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
#[pyo3(signature = (data, template, event_indices, center_offset=0, max_lag=8, dynamic_scaling=true))]
pub fn subtract_template<'py>(
    py: Python<'py>,
    data: Bound<'py, PyAny>,
    template: Bound<'py, PyAny>,
    event_indices: Vec<u64>,
    center_offset: usize,
    max_lag: usize,
    dynamic_scaling: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let filter = PyTemplateFilter::new(py, template, center_offset, max_lag, dynamic_scaling)?;
    filter.apply(py, data, event_indices)
}
