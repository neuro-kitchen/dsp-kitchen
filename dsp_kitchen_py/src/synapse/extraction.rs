//! Waveform snippets around spikes, realigned to their sub-sample trough (windowed sinc).

use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pyfunction, gen_stub_pymethods};

use crate::array::{to_numpy, F32Array};
use dsp_synapse::detection::DeduplicatedSpike;
use dsp_synapse::extraction::{extract_snippets_multichannel, WaveformSnippet};
use dsp_synapse::StreamingDetectionConfig;
use super::detection::PyDeduplicatedSpike;
use super::probe::PyProbeLayout;

/// A spike's multi-channel waveform cut from the recording (see `extract_snippets`).
#[gen_stub_pyclass]
#[pyclass(name = "WaveformSnippet", skip_from_py_object)]
#[derive(Clone)]
pub struct PyWaveformSnippet {
    pub inner: WaveformSnippet,
    /// Sample of the snippet the spike's trough is aligned to (`pre_samples` at extraction).
    pub peak_index: usize,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyWaveformSnippet {
    /// Channel the spike was detected on.
    #[getter]
    pub fn primary_channel(&self) -> usize {
        self.inner.primary_channel
    }

    /// Recording sample of the spike.
    #[getter]
    pub fn center_sample(&self) -> u64 {
        self.inner.center_sample
    }

    /// Sub-sample shift of the trough found by interpolation, samples.
    #[getter]
    pub fn subsample_offset(&self) -> f32 {
        self.inner.subsample_offset
    }

    /// Recording channel of each row of the snippet.
    #[getter]
    pub fn channel_ids(&self) -> Vec<usize> {
        self.inner.channel_ids.clone()
    }

    /// Sample of the snippet the trough is aligned to.
    #[getter]
    pub fn peak_index(&self) -> usize {
        self.peak_index
    }

    /// Samples per channel.
    #[getter]
    pub fn num_samples(&self) -> usize {
        self.inner.num_samples
    }

    /// Channels in the snippet.
    #[getter]
    pub fn num_channels(&self) -> usize {
        self.inner.num_channels()
    }

    /// The snippet, `[channels, samples]` float32 (rows follow `channel_ids`).
    #[gen_stub(override_return_type(type_repr = "numpy.typing.NDArray[numpy.float32]", imports = ("numpy", "numpy.typing")))]
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

/// Cuts each spike's waveform out of a signal on its nearest channels.
///
/// Parameters
/// ----------
/// data : numpy.ndarray
///     `[channels, samples]`, converted to float32 (spike samples index into it).
/// spikes : list of DeduplicatedSpike
/// probe : ProbeLayout
/// k_neighbors : int
///     Channels per snippet: the primary and its nearest (see `ProbeLayout.k_nearest_neighbors`).
/// pre_samples, post_samples : int
///     Samples before and after each spike.
/// apply_sinc_shift : bool, default True
///     Realign each snippet to its sub-sample trough (sinc interpolation).
///
/// Returns
/// -------
/// list of WaveformSnippet
#[gen_stub_pyfunction]
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
