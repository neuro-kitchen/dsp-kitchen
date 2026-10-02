//! NumPy array exchange: borrow float32 input without copying when it is already contiguous
//! float32, and hand results to NumPy by moving the `Vec` (no byte round-trips).

use numpy::{IntoPyArray, PyArrayDyn, PyArrayMethods, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// A float32 NumPy array borrowed from Python. Converted (one copy) only when the input is not
/// already C-contiguous float32.
pub struct F32Array<'py> {
    arr: PyReadonlyArrayDyn<'py, f32>,
}

impl<'py> F32Array<'py> {
    pub fn new(data: &Bound<'py, PyAny>) -> PyResult<Self> {
        let np = data.py().import("numpy")?;
        let contiguous = np.call_method1("ascontiguousarray", (data, "float32"))?;
        let arr = contiguous.cast_into::<PyArrayDyn<f32>>()?.readonly();
        Ok(Self { arr })
    }

    pub fn shape(&self) -> &[usize] {
        self.arr.shape()
    }

    pub fn ndim(&self) -> usize {
        self.arr.ndim()
    }

    /// The samples, row-major. `Send`, so it can be used inside `py.detach`.
    pub fn slice(&self) -> &[f32] {
        self.arr.as_slice().expect("ascontiguousarray returns a contiguous array")
    }

    /// `(channels, samples)` of a 2-D `[channels, samples]` array, or of a 1-D array holding
    /// `channels` (default 1) channels back to back.
    pub fn channels_samples(&self, channels: Option<usize>) -> PyResult<(usize, usize)> {
        match *self.shape() {
            [n] => {
                let c = channels.unwrap_or(1);
                if c == 0 || n % c != 0 {
                    return Err(PyValueError::new_err(format!(
                        "a 1-D array of {n} samples cannot hold {c} channels"
                    )));
                }
                Ok((c, n / c))
            }
            [c, n] => {
                if channels.is_some_and(|expected| expected != c) {
                    return Err(PyValueError::new_err(format!(
                        "array has {c} channels but channels={} was given",
                        channels.unwrap_or_default()
                    )));
                }
                Ok((c, n))
            }
            _ => Err(PyValueError::new_err(format!(
                "expected a 1-D or 2-D array [channels, samples], got shape {:?}",
                self.shape()
            ))),
        }
    }
}

/// Moves `data` into a NumPy array of `shape` (no copy).
pub fn to_numpy<'py>(py: Python<'py>, data: Vec<f32>, shape: &[usize]) -> PyResult<Bound<'py, PyAny>> {
    debug_assert_eq!(data.len(), shape.iter().product::<usize>());
    Ok(data.into_pyarray(py).reshape(shape)?.into_any())
}

/// Moves `u64` data into a NumPy array of `shape` (no copy).
pub fn to_numpy_u64<'py>(py: Python<'py>, data: Vec<u64>, shape: &[usize]) -> PyResult<Bound<'py, PyAny>> {
    debug_assert_eq!(data.len(), shape.iter().product::<usize>());
    Ok(data.into_pyarray(py).reshape(shape)?.into_any())
}

/// Moves `f64` data into a NumPy array of `shape` (no copy).
pub fn to_numpy_f64<'py>(py: Python<'py>, data: Vec<f64>, shape: &[usize]) -> PyResult<Bound<'py, PyAny>> {
    debug_assert_eq!(data.len(), shape.iter().product::<usize>());
    Ok(data.into_pyarray(py).reshape(shape)?.into_any())
}

pub fn value_error(e: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// A failure while running a model or kernel (not a bad argument).
pub fn runtime_error(e: impl std::fmt::Display) -> PyErr {
    pyo3::exceptions::PyRuntimeError::new_err(e.to_string())
}
