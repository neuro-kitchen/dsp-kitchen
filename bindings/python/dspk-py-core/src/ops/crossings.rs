//! `dspk.ops.crossings`: threshold crossings as events on the device
//! (`dspk_ops::crossings::detect_crossings_on_device`).

use crate::ops::device::DeviceInput;
use dspk_ops::crossings::{detect_crossings_on_device, CrossingOptions, CrossingTracker, Direction};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_stub_gen::derive::gen_stub_pyfunction;

use super::device;

type Events<'py> = (Bound<'py, PyArray1<u32>>, Bound<'py, PyArray1<u64>>, Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<u64>>, Bound<'py, PyArray1<f32>>);

/// Events where each channel of `data` `(channels, samples)` goes past `levels` (one per channel)
/// and back past `releases`: `direction="rising"`, `"falling"` or `"both"` (magnitudes). Events
/// shorter than `min_duration_samples` are dropped; onsets within `refractory_samples` after an
/// event's end are ignored. Returns `(channel, onset, offset, peak_sample, peak_value)` arrays by
/// onset then channel; `offset` is -1 for an event still open at the end.
#[gen_stub_pyfunction(module = "dspk._native.ops.crossings")]
#[pyfunction]
#[pyo3(signature = (data, levels, releases, *, direction, min_duration_samples, refractory_samples, runtime))]
#[allow(clippy::too_many_arguments)]
pub fn detect<'py>(
    py: Python<'py>,
    data: DeviceInput,
    levels: PyReadonlyArray1<'_, f32>,
    releases: PyReadonlyArray1<'_, f32>,
    direction: &str,
    min_duration_samples: u64,
    refractory_samples: u64,
    runtime: Option<&str>,
) -> PyResult<Events<'py>> {
    let direction = match direction {
        "rising" => Direction::Rising,
        "falling" => Direction::Falling,
        "both" => Direction::Both,
        other => return Err(PyValueError::new_err(format!("unknown direction {other:?} (rising, falling, both)"))),
    };
    let options = CrossingOptions {
        levels: levels.as_array().iter().copied().collect(),
        releases: releases.as_array().iter().copied().collect(),
        direction,
        min_duration_samples,
        refractory_samples,
    };
    // Options checked here, so a bad level is a ValueError rather than a runtime failure
    CrossingTracker::new(data.dim().0, &options).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let events = device::with_input(py, data, runtime, move |client, input, channels, samples| {
        detect_crossings_on_device::<f32>(client, input, channels, samples, &options).map_err(|e| e.to_string())
    })?;
    let channel: Vec<u32> = events.iter().map(|e| e.channel as u32).collect();
    let onset: Vec<u64> = events.iter().map(|e| e.onset).collect();
    let offset: Vec<i64> = events.iter().map(|e| e.offset.map_or(-1, |o| o as i64)).collect();
    let peak_sample: Vec<u64> = events.iter().map(|e| e.peak_sample).collect();
    let peak_value: Vec<f32> = events.iter().map(|e| e.peak_value).collect();
    Ok((channel.into_pyarray(py), onset.into_pyarray(py), offset.into_pyarray(py), peak_sample.into_pyarray(py), peak_value.into_pyarray(py)))
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(detect, m)?)?;
    Ok(())
}
