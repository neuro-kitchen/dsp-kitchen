//! `dspk.ops.layout`: channels in a new order on the device (`dspk_ops::core::layout`), and
//! channel maps (`dspk_core::ChannelMap`: pairs, from TOML / JSON files).

use dspk_core::ChannelMap;
use dspk_ops::core::layout::execute_gather_channels;
use numpy::PyArray2;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_stub_gen::derive::gen_stub_pyfunction;

use super::device::{self, DeviceInput};

fn invalid(e: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// `data` `(channels, samples)` with its channels in `order`: row `i` of the result is channel
/// `order[i]` (channels may be left out or repeated).
#[gen_stub_pyfunction(module = "dspk._native.ops.layout")]
#[pyfunction]
#[pyo3(signature = (data, order, *, runtime))]
pub fn gather_channels<'py>(py: Python<'py>, data: DeviceInput, order: Vec<usize>, runtime: Option<&str>) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let (channels, samples) = data.dim();
    if let Some(&c) = order.iter().find(|&&c| c >= channels) {
        return Err(PyValueError::new_err(format!("channel {c} does not exist (the data has {channels})")));
    }
    if order.is_empty() {
        return Err(PyValueError::new_err("order is empty"));
    }
    device::run_shaped(py, data, runtime, (order.len(), samples), move |client, input, output, channels, samples| {
        execute_gather_channels::<f32>(client, input, output, channels, samples, &order);
        Ok(())
    })
}

/// A channel map file's text (TOML with `toml=True`, else JSON; the pairs under `remap`), checked;
/// returns the map as canonical JSON (`{"base": …, "pairs": [{"from", "to", "name"?}, …]}`).
#[gen_stub_pyfunction(module = "dspk._native.ops.layout")]
#[pyfunction]
#[pyo3(signature = (text, toml=false))]
pub fn parse_map(text: &str, toml: bool) -> PyResult<String> {
    let map = dspk_io::parse_channel_map(text, toml).map_err(invalid)?;
    Ok(serde_json::to_string(&map).expect("a channel map serialises"))
}

/// A map (canonical JSON, as `parse_map` returns) as a TOML file's text.
#[gen_stub_pyfunction(module = "dspk._native.ops.layout")]
#[pyfunction]
pub fn map_toml(map: &str) -> PyResult<String> {
    let map: ChannelMap = serde_json::from_str(map).map_err(invalid)?;
    map.order().map_err(invalid)?;
    Ok(dspk_io::channel_map_toml(&map))
}

/// The input channel (0-based) of every output channel and the names the map gives
/// (`None`: unnamed), for data of `channels` channels (`None`: not checked against a count).
#[gen_stub_pyfunction(module = "dspk._native.ops.layout")]
#[pyfunction]
#[pyo3(signature = (map, channels=None))]
pub fn map_order(map: &str, channels: Option<usize>) -> PyResult<(Vec<usize>, Vec<Option<String>>)> {
    let map: ChannelMap = serde_json::from_str(map).map_err(invalid)?;
    let order = match channels {
        Some(n) => map.resolve(n),
        None => map.order(),
    }
    .map_err(invalid)?;
    Ok((order, map.names()))
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(gather_channels, m)?)?;
    m.add_function(wrap_pyfunction!(parse_map, m)?)?;
    m.add_function(wrap_pyfunction!(map_toml, m)?)?;
    m.add_function(wrap_pyfunction!(map_order, m)?)?;
    Ok(())
}
