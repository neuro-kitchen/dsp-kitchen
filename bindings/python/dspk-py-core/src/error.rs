//! dspk errors as Python exceptions: file problems are `OSError`, bad arguments `ValueError`,
//! device failures `RuntimeError`.

use dspk_core::DspError;
use pyo3::exceptions::{PyIndexError, PyOSError, PyRuntimeError, PyValueError};
use pyo3::PyErr;

pub fn to_py(e: DspError) -> PyErr {
    let msg = e.to_string();
    match e {
        DspError::Io(_) => PyOSError::new_err(msg),
        DspError::InvalidChannel { .. } | DspError::SampleRange { .. } => PyIndexError::new_err(msg),
        DspError::ComputeError(_) => PyRuntimeError::new_err(msg),
        _ => PyValueError::new_err(msg),
    }
}
