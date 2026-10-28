//! `dspk.ops.peaks`: peak candidates on the device (`dspk_ops::peaks::find_peak_candidates`).
//!
//! The scipy-style filters of `find_peaks` (distance, prominence, width) run on the host in
//! dspk-ops and are bound once they run on the device.

use dspk_ops::core::buffer;
use dspk_ops::peaks::{find_peak_candidates, Polarity};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_stub_gen::derive::gen_stub_pyfunction;

use super::device;

type Table<'py> = (Bound<'py, PyArray1<u32>>, Bound<'py, PyArray1<u32>>, Bound<'py, PyArray1<f32>>);

/// Local extrema of `data` `(channels, samples)` whose value reaches the channel's `heights`
/// entry: `polarity="positive"` local maxima (`value ≥ height`), `"negative"` local minima
/// (`value ≤ -height`), `"both"` either (maxima reaching `height` and minima reaching `-height`). Strict on the left
/// (`x[t-1] < x[t] >= x[t+1]`): a flat peak reports its first sample. Returns `(channel, sample,
/// value)` arrays, ordered by channel then sample. The first and last samples are never peaks.
#[gen_stub_pyfunction(module = "dspk._native.ops.peaks")]
#[pyfunction]
#[pyo3(signature = (data, heights, *, polarity, runtime))]
pub fn candidates<'py>(py: Python<'py>, data: PyReadonlyArray2<'_, f32>, heights: PyReadonlyArray1<'_, f32>, polarity: &str, runtime: Option<&str>) -> PyResult<Table<'py>> {
    let polarity = match polarity {
        "positive" => Polarity::Positive,
        "negative" => Polarity::Negative,
        "both" => Polarity::Both,
        other => return Err(PyValueError::new_err(format!("unknown polarity {other:?} (positive, negative, both)"))),
    };
    let channels = data.as_array().dim().0;
    let heights: Vec<f32> = heights.as_array().iter().copied().collect();
    if heights.len() != channels {
        return Err(PyValueError::new_err(format!("heights has {} values, data has {channels} channels", heights.len())));
    }
    let found = device::with_input(py, data, runtime, move |client, input, channels, samples| {
        let heights = buffer::upload(client, &heights);
        Ok(find_peak_candidates::<f32>(client, input, &heights, channels, samples, 0..samples, polarity))
    })?;
    let rows: Vec<u32> = (0..channels).flat_map(|c| std::iter::repeat_n(c as u32, found.bases[c + 1] - found.bases[c])).collect();
    Ok((rows.into_pyarray(py), found.indices.into_pyarray(py), found.values.into_pyarray(py)))
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(candidates, m)?)?;
    Ok(())
}
