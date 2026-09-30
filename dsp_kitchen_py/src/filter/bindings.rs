use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use crate::pipeline::PyPipeline;
use dsp_base::filter::{FilterBand, FilterDesign, FilterMode, FilterSpec};
use dsp_base::pipeline::PipelineStage;

/// Parses the Python `direction` argument ("forward-backward" | "forward"), SpikeInterface naming.
fn parse_direction(direction: &str) -> PyResult<FilterMode> {
    match direction {
        "forward-backward" | "zero-phase" => Ok(FilterMode::ForwardBackward),
        "forward" | "causal" => Ok(FilterMode::Forward),
        other => Err(PyValueError::new_err(format!(
            "direction must be 'forward-backward' or 'forward', got '{other}'"
        ))),
    }
}

fn direction_name(mode: FilterMode) -> &'static str {
    match mode {
        FilterMode::ForwardBackward => "forward-backward",
        FilterMode::Forward => "forward",
    }
}

/// Extracts the filter spec from any of the filter stage classes.
pub fn extract_filter_spec(item: &Bound<'_, PyAny>) -> Option<FilterSpec> {
    if let Ok(f) = item.extract::<PyRef<PyNotchFilter>>() {
        Some(f.spec.clone())
    } else if let Ok(f) = item.extract::<PyRef<PyBandpassFilter>>() {
        Some(f.spec.clone())
    } else if let Ok(f) = item.extract::<PyRef<PyHighpassFilter>>() {
        Some(f.spec.clone())
    } else if let Ok(f) = item.extract::<PyRef<PyLowpassFilter>>() {
        Some(f.spec.clone())
    } else if let Ok(f) = item.extract::<PyRef<PyBandstopFilter>>() {
        Some(f.spec.clone())
    } else {
        None
    }
}

#[pyclass(name = "NotchFilter", skip_from_py_object)]
#[derive(Clone)]
pub struct PyNotchFilter {
    pub spec: FilterSpec,
}

#[pymethods]
impl PyNotchFilter {
    #[new]
    #[pyo3(signature = (freq_hz=None, q=30.0, direction="forward-backward", freq=None))]
    pub fn new(freq_hz: Option<f64>, q: f64, direction: &str, freq: Option<f64>) -> PyResult<Self> {
        let f = freq_hz.or(freq).unwrap_or(60.0);
        Ok(Self { spec: FilterSpec::notch(f, q).with_mode(parse_direction(direction)?) })
    }

    fn __repr__(&self) -> String {
        match &self.spec.design {
            FilterDesign::Notch { freq_hz, q } => format!(
                "NotchFilter(freq_hz={freq_hz}, q={q}, direction='{}')",
                direction_name(self.spec.mode)
            ),
            _ => unreachable!(),
        }
    }
}

fn butterworth_repr(class: &str, spec: &FilterSpec) -> String {
    match &spec.design {
        FilterDesign::Butterworth { order, band } => {
            let edges = match band {
                FilterBand::Lowpass(f) | FilterBand::Highpass(f) => format!("cutoff_hz={f}"),
                FilterBand::Bandpass(lo, hi) | FilterBand::Bandstop(lo, hi) => {
                    format!("low_hz={lo}, high_hz={hi}")
                }
            };
            format!("{class}({edges}, order={order}, direction='{}')", direction_name(spec.mode))
        }
        _ => unreachable!(),
    }
}

/// Butterworth band-pass (order per edge, as scipy / SpikeInterface).
#[pyclass(name = "BandpassFilter", skip_from_py_object)]
#[derive(Clone)]
pub struct PyBandpassFilter {
    pub spec: FilterSpec,
}

#[pymethods]
impl PyBandpassFilter {
    #[new]
    #[pyo3(signature = (low_hz=None, high_hz=None, order=5, direction="forward-backward", low=None, high=None))]
    pub fn new(
        low_hz: Option<f64>,
        high_hz: Option<f64>,
        order: usize,
        direction: &str,
        low: Option<f64>,
        high: Option<f64>,
    ) -> PyResult<Self> {
        let l = low_hz.or(low).unwrap_or(300.0);
        let h = high_hz.or(high).unwrap_or(6000.0);
        let spec = FilterSpec::butterworth(order, FilterBand::Bandpass(l, h))
            .with_mode(parse_direction(direction)?);
        Ok(Self { spec })
    }

    fn __repr__(&self) -> String {
        butterworth_repr("BandpassFilter", &self.spec)
    }
}

/// Butterworth high-pass (e.g. 300 Hz for the spike band).
#[pyclass(name = "HighpassFilter", skip_from_py_object)]
#[derive(Clone)]
pub struct PyHighpassFilter {
    pub spec: FilterSpec,
}

#[pymethods]
impl PyHighpassFilter {
    #[new]
    #[pyo3(signature = (cutoff_hz=300.0, order=5, direction="forward-backward"))]
    pub fn new(cutoff_hz: f64, order: usize, direction: &str) -> PyResult<Self> {
        let spec = FilterSpec::butterworth(order, FilterBand::Highpass(cutoff_hz))
            .with_mode(parse_direction(direction)?);
        Ok(Self { spec })
    }

    fn __repr__(&self) -> String {
        butterworth_repr("HighpassFilter", &self.spec)
    }
}

/// Butterworth low-pass (e.g. 300 Hz for LFP).
#[pyclass(name = "LowpassFilter", skip_from_py_object)]
#[derive(Clone)]
pub struct PyLowpassFilter {
    pub spec: FilterSpec,
}

#[pymethods]
impl PyLowpassFilter {
    #[new]
    #[pyo3(signature = (cutoff_hz=300.0, order=5, direction="forward-backward"))]
    pub fn new(cutoff_hz: f64, order: usize, direction: &str) -> PyResult<Self> {
        let spec = FilterSpec::butterworth(order, FilterBand::Lowpass(cutoff_hz))
            .with_mode(parse_direction(direction)?);
        Ok(Self { spec })
    }

    fn __repr__(&self) -> String {
        butterworth_repr("LowpassFilter", &self.spec)
    }
}

/// Butterworth band-stop.
#[pyclass(name = "BandstopFilter", skip_from_py_object)]
#[derive(Clone)]
pub struct PyBandstopFilter {
    pub spec: FilterSpec,
}

#[pymethods]
impl PyBandstopFilter {
    #[new]
    #[pyo3(signature = (low_hz, high_hz, order=5, direction="forward-backward"))]
    pub fn new(low_hz: f64, high_hz: f64, order: usize, direction: &str) -> PyResult<Self> {
        let spec = FilterSpec::butterworth(order, FilterBand::Bandstop(low_hz, high_hz))
            .with_mode(parse_direction(direction)?);
        Ok(Self { spec })
    }

    fn __repr__(&self) -> String {
        butterworth_repr("BandstopFilter", &self.spec)
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

fn run_filter<'py>(
    py: Python<'py>,
    data: Bound<'py, PyAny>,
    spec: FilterSpec,
    fs: f64,
) -> PyResult<Bound<'py, PyAny>> {
    PyPipeline::from_stages(vec![PipelineStage::Filter(spec)]).run(py, data, fs, None)
}

#[pyfunction]
#[pyo3(signature = (data, freq=60.0, q=30.0, fs=30000.0, direction="forward-backward"))]
pub fn notch_filter<'py>(
    py: Python<'py>,
    data: Bound<'py, PyAny>,
    freq: f64,
    q: f64,
    fs: f64,
    direction: &str,
) -> PyResult<Bound<'py, PyAny>> {
    run_filter(py, data, FilterSpec::notch(freq, q).with_mode(parse_direction(direction)?), fs)
}

#[pyfunction]
#[pyo3(signature = (data, low=300.0, high=6000.0, fs=30000.0, order=5, direction="forward-backward"))]
pub fn bandpass_filter<'py>(
    py: Python<'py>,
    data: Bound<'py, PyAny>,
    low: f64,
    high: f64,
    fs: f64,
    order: usize,
    direction: &str,
) -> PyResult<Bound<'py, PyAny>> {
    let spec = FilterSpec::butterworth(order, FilterBand::Bandpass(low, high))
        .with_mode(parse_direction(direction)?);
    run_filter(py, data, spec, fs)
}

#[pyfunction]
#[pyo3(signature = (data, cutoff=300.0, fs=30000.0, order=5, direction="forward-backward"))]
pub fn highpass_filter<'py>(
    py: Python<'py>,
    data: Bound<'py, PyAny>,
    cutoff: f64,
    fs: f64,
    order: usize,
    direction: &str,
) -> PyResult<Bound<'py, PyAny>> {
    let spec = FilterSpec::butterworth(order, FilterBand::Highpass(cutoff))
        .with_mode(parse_direction(direction)?);
    run_filter(py, data, spec, fs)
}

#[pyfunction]
#[pyo3(signature = (data, cutoff=300.0, fs=30000.0, order=5, direction="forward-backward"))]
pub fn lowpass_filter<'py>(
    py: Python<'py>,
    data: Bound<'py, PyAny>,
    cutoff: f64,
    fs: f64,
    order: usize,
    direction: &str,
) -> PyResult<Bound<'py, PyAny>> {
    let spec = FilterSpec::butterworth(order, FilterBand::Lowpass(cutoff))
        .with_mode(parse_direction(direction)?);
    run_filter(py, data, spec, fs)
}

#[pyfunction]
#[pyo3(signature = (data))]
pub fn median_filter_9p<'py>(
    py: Python<'py>,
    data: Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let py_pipe = PyPipeline::from_stages(vec![PipelineStage::Median9p]);
    py_pipe.run(py, data, 30000.0, None)
}

#[pyfunction]
#[pyo3(signature = (data))]
pub fn teager_kaiser_filter<'py>(
    py: Python<'py>,
    data: Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let py_pipe = PyPipeline::from_stages(vec![PipelineStage::TeagerKaiser]);
    py_pipe.run(py, data, 30000.0, None)
}
