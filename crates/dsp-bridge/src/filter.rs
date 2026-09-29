use pyo3::prelude::*;
use crate::pipeline::PyPipeline;
use dsp_base::pipeline::{Pipeline, PipelineStage};

#[pyclass(name = "NotchFilter", skip_from_py_object)]
#[derive(Clone)]
pub struct PyNotchFilter {
    pub freq_hz: f64,
    pub q: f64,
}

#[pymethods]
impl PyNotchFilter {
    #[new]
    #[pyo3(signature = (freq_hz=None, q=30.0, freq=None))]
    pub fn new(freq_hz: Option<f64>, q: f64, freq: Option<f64>) -> Self {
        let f = freq_hz.or(freq).unwrap_or(60.0);
        Self { freq_hz: f, q }
    }

    fn __repr__(&self) -> String {
        format!("NotchFilter(freq_hz={}Hz, q={})", self.freq_hz, self.q)
    }
}

#[pyclass(name = "BandpassFilter", skip_from_py_object)]
#[derive(Clone)]
pub struct PyBandpassFilter {
    pub low_hz: f64,
    pub high_hz: f64,
}

#[pymethods]
impl PyBandpassFilter {
    #[new]
    #[pyo3(signature = (low_hz=None, high_hz=None, low=None, high=None))]
    pub fn new(
        low_hz: Option<f64>,
        high_hz: Option<f64>,
        low: Option<f64>,
        high: Option<f64>,
    ) -> Self {
        let l = low_hz.or(low).unwrap_or(300.0);
        let h = high_hz.or(high).unwrap_or(6000.0);
        Self {
            low_hz: l,
            high_hz: h,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "BandpassFilter(low_hz={}Hz, high_hz={}Hz)",
            self.low_hz, self.high_hz
        )
    }
}

#[pyclass(name = "MedianFilter", skip_from_py_object)]
#[derive(Clone)]
pub struct PyMedianFilter;

#[pymethods]
impl PyMedianFilter {
    #[new]
    pub fn new() -> Self {
        Self
    }

    fn __repr__(&self) -> String {
        "MedianFilter(points=9)".to_string()
    }
}

#[pyclass(name = "TeagerKaiser", skip_from_py_object)]
#[derive(Clone)]
pub struct PyTeagerKaiser;

#[pymethods]
impl PyTeagerKaiser {
    #[new]
    pub fn new() -> Self {
        Self
    }

    fn __repr__(&self) -> String {
        "TeagerKaiser()".to_string()
    }
}

#[pyfunction]
#[pyo3(signature = (data, freq=60.0, q=30.0, fs=30000.0))]
pub fn notch_filter<'py>(
    py: Python<'py>,
    data: Bound<'py, PyAny>,
    freq: f64,
    q: f64,
    fs: f64,
) -> PyResult<Bound<'py, PyAny>> {
    let mut pipe = Pipeline::new();
    pipe.add(PipelineStage::Notch { freq_hz: freq, q });
    let py_pipe = PyPipeline::from_stages(pipe.stages().to_vec());
    py_pipe.run(py, data, fs, None)
}

#[pyfunction]
#[pyo3(signature = (data, low=300.0, high=6000.0, fs=30000.0))]
pub fn bandpass_filter<'py>(
    py: Python<'py>,
    data: Bound<'py, PyAny>,
    low: f64,
    high: f64,
    fs: f64,
) -> PyResult<Bound<'py, PyAny>> {
    let mut pipe = Pipeline::new();
    pipe.add(PipelineStage::Bandpass {
        low_hz: low,
        high_hz: high,
    });
    let py_pipe = PyPipeline::from_stages(pipe.stages().to_vec());
    py_pipe.run(py, data, fs, None)
}

#[pyfunction]
#[pyo3(signature = (data))]
pub fn median_filter_9p<'py>(
    py: Python<'py>,
    data: Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let mut pipe = Pipeline::new();
    pipe.add(PipelineStage::Median9p);
    let py_pipe = PyPipeline::from_stages(pipe.stages().to_vec());
    py_pipe.run(py, data, 30000.0, None)
}

#[pyfunction]
#[pyo3(signature = (data))]
pub fn teager_kaiser_filter<'py>(
    py: Python<'py>,
    data: Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let mut pipe = Pipeline::new();
    pipe.add(PipelineStage::TeagerKaiser);
    let py_pipe = PyPipeline::from_stages(pipe.stages().to_vec());
    py_pipe.run(py, data, 30000.0, None)
}
