//! Waveform snippets around spikes, realigned to their sub-sample trough (windowed sinc).

use pyo3::prelude::*;

use crate::array::{to_numpy, F32Array};
use dsp_synapse::detection::DeduplicatedSpike;
use dsp_synapse::extraction::{extract_snippets_multichannel, WaveformSnippet};
use dsp_synapse::StreamingDetectionConfig;
use super::detection::PyDeduplicatedSpike;
use super::probe::PyProbeLayout;

/// Extracted multi-channel waveform snippet.
#[pyclass(name = "WaveformSnippet", skip_from_py_object)]
#[derive(Clone)]
pub struct PyWaveformSnippet {
    pub inner: WaveformSnippet,
    /// Sample of the snippet the spike's trough is aligned to (`pre_samples` at extraction).
    pub peak_index: usize,
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

    /// Sample of the snippet the trough is aligned to.
    #[getter]
    pub fn peak_index(&self) -> usize {
        self.peak_index
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
        let shape = [self.inner.num_channels(), self.inner.num_samples];
        to_numpy(py, self.inner.waveform.clone(), &shape)
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

/// Snippets of `data` (`[channels, samples]`) on each spike's `k_neighbors` nearest channels
/// (on `probe`), `pre_samples` before and `post_samples` after it; realigned to the sub-sample
/// trough unless `apply_sinc_shift=False` (default from the streaming detection settings).
#[pyfunction]
#[pyo3(signature = (data, spikes, probe, *, k_neighbors, pre_samples, post_samples, apply_sinc_shift=None))]
#[allow(clippy::too_many_arguments)]
pub fn extract_snippets(
    py: Python<'_>,
    data: Bound<'_, PyAny>,
    spikes: Vec<PyRef<'_, PyDeduplicatedSpike>>,
    probe: PyRef<'_, PyProbeLayout>,
    k_neighbors: usize,
    pre_samples: usize,
    post_samples: usize,
    apply_sinc_shift: Option<bool>,
) -> PyResult<Vec<PyWaveformSnippet>> {
    let input = F32Array::new(&data)?;
    let (channels, samples) = input.channels_samples(None)?;
    let x = input.slice();
    let shift = apply_sinc_shift.unwrap_or(StreamingDetectionConfig::default().apply_sinc_shift);
    let spikes: Vec<DeduplicatedSpike> = spikes.iter().map(|s| DeduplicatedSpike::from(&**s)).collect();
    let layout = &probe.inner;
    let snippets = py.detach(|| extract_snippets_multichannel(x, channels, samples, &spikes, layout, k_neighbors, pre_samples, post_samples, shift));
    Ok(snippets.into_iter().map(|inner| PyWaveformSnippet { inner, peak_index: pre_samples }).collect())
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyWaveformSnippet>()?;
    m.add_function(wrap_pyfunction!(extract_snippets, m)?)?;
    Ok(())
}
