use pyo3::prelude::*;
use pyo3::types::PyBytes;
use dsp_synapse::detection::DeduplicatedSpike;
use dsp_synapse::extraction::{extract_snippets_multichannel, WaveformSnippet};
use super::detection::PyDeduplicatedSpike;
use super::probe::PyProbeLayout;

/// Extracted multi-channel waveform snippet.
#[pyclass(name = "WaveformSnippet", skip_from_py_object)]
#[derive(Clone)]
pub struct PyWaveformSnippet {
    pub inner: WaveformSnippet,
}

#[pymethods]
impl PyWaveformSnippet {
    #[getter]
    pub fn primary_channel(&self) -> usize {
        self.inner.primary_channel
    }

    #[getter]
    pub fn center_sample(&self) -> u64 {
        self.inner.center_sample
    }

    #[getter]
    pub fn subsample_offset(&self) -> f32 {
        self.inner.subsample_offset
    }

    #[getter]
    pub fn channel_ids(&self) -> Vec<usize> {
        self.inner.channel_ids.clone()
    }

    #[getter]
    pub fn num_samples(&self) -> usize {
        self.inner.num_samples
    }

    #[getter]
    pub fn num_channels(&self) -> usize {
        self.inner.num_channels()
    }

    /// Returns the 2D waveform as a numpy array of shape [num_channels, num_samples].
    pub fn waveform<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let np = py.import("numpy")?;
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                self.inner.waveform.as_ptr() as *const u8,
                self.inner.waveform.len() * std::mem::size_of::<f32>(),
            )
        };
        let py_bytes = PyBytes::new(py, bytes);
        let flat = np.call_method1("frombuffer", (py_bytes, "float32"))?;
        let k = self.inner.num_channels();
        let samples = self.inner.num_samples;
        let arr = flat.call_method1("reshape", ((k, samples),))?;
        Ok(arr)
    }

    fn __repr__(&self) -> String {
        format!(
            "WaveformSnippet(primary_channel={}, center_sample={}, shape=[{}, {}], offset={:.3})",
            self.inner.primary_channel,
            self.inner.center_sample,
            self.inner.num_channels(),
            self.inner.num_samples,
            self.inner.subsample_offset
        )
    }
}

#[pyfunction]
#[pyo3(signature = (data, spikes, probe, channels=None, k_neighbors=7, pre_samples=20, post_samples=40, apply_sinc_shift=true))]
pub fn extract_snippets<'py>(
    py: Python<'py>,
    data: Bound<'py, PyAny>,
    spikes: Vec<PyRef<PyDeduplicatedSpike>>,
    probe: PyRef<PyProbeLayout>,
    channels: Option<usize>,
    k_neighbors: usize,
    pre_samples: usize,
    post_samples: usize,
    apply_sinc_shift: bool,
) -> PyResult<Vec<PyWaveformSnippet>> {
    let np = py.import("numpy")?;
    let arr = np.call_method1("ascontiguousarray", (data, "float32"))?;
    let shape: Vec<usize> = arr.getattr("shape")?.extract()?;
    let (ch, samples) = match shape.len() {
        1 => {
            let c = channels.ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err("channels must be specified for 1D arrays")
            })?;
            (c, shape[0] / c)
        }
        2 => (shape[0], shape[1]),
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Data must be a 1D or 2D float32 array [channels, samples]",
            ));
        }
    };

    let py_bytes = arr.call_method0("tobytes")?;
    let raw_bytes: &[u8] = py_bytes.extract()?;
    let float_slice: &[f32] = unsafe {
        std::slice::from_raw_parts(
            raw_bytes.as_ptr() as *const f32,
            raw_bytes.len() / std::mem::size_of::<f32>(),
        )
    };

    let rust_spikes: Vec<DeduplicatedSpike> = spikes
        .iter()
        .map(|s| DeduplicatedSpike {
            primary_channel: s.primary_channel,
            sample_index: s.sample_index,
            peak_amplitude_uv: s.peak_amplitude_uv,
            participating_channels: s.participating_channels.clone(),
        })
        .collect();

    let snippets = extract_snippets_multichannel(
        float_slice,
        ch,
        samples,
        &rust_spikes,
        &probe.inner,
        k_neighbors,
        pre_samples,
        post_samples,
        apply_sinc_shift,
    );

    Ok(snippets
        .into_iter()
        .map(|s| PyWaveformSnippet { inner: s })
        .collect())
}
