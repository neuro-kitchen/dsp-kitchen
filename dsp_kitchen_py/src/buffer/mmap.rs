use std::fs::File;
use std::sync::Arc;
use memmap2::Mmap;
use pyo3::prelude::*;
use pyo3::types::PyMemoryView;

/// Zero-copy Memory-Mapped Raw Binary Recording.
#[pyclass(name = "MmapRecording", skip_from_py_object)]
pub struct PyMmapRecording {
    path: String,
    mmap: Arc<Mmap>,
    channels: usize,
    samples: usize,
    sample_rate: f64,
}

#[pymethods]
impl PyMmapRecording {
    #[new]
    #[pyo3(signature = (path, channels=384, samples=0, sample_rate=30000.0))]
    pub fn new(path: &str, channels: usize, mut samples: usize, sample_rate: f64) -> PyResult<Self> {
        let file = File::open(path).map_err(|e| {
            pyo3::exceptions::PyFileNotFoundError::new_err(format!(
                "Failed to open recording file '{}': {}",
                path, e
            ))
        })?;

        let mmap = unsafe {
            Mmap::map(&file).map_err(|e| {
                pyo3::exceptions::PyIOError::new_err(format!(
                    "Failed to memory-map file '{}': {}",
                    path, e
                ))
            })?
        };

        let total_bytes = mmap.len();
        let bytes_per_sample = std::mem::size_of::<f32>();

        if samples == 0 {
            if channels == 0 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "channels must be > 0",
                ));
            }
            samples = total_bytes / (channels * bytes_per_sample);
        }

        let expected_bytes = channels * samples * bytes_per_sample;
        if total_bytes < expected_bytes {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "File size ({} bytes) is smaller than required ({} bytes for {} ch x {} samples of float32)",
                total_bytes, expected_bytes, channels, samples
            )));
        }

        Ok(Self {
            path: path.to_string(),
            mmap: Arc::new(mmap),
            channels,
            samples,
            sample_rate,
        })
    }

    #[getter]
    pub fn path(&self) -> &str {
        &self.path
    }

    #[getter]
    pub fn channels(&self) -> usize {
        self.channels
    }

    #[getter]
    pub fn samples(&self) -> usize {
        self.samples
    }

    #[getter]
    pub fn sample_rate(&self) -> f64 {
        self.sample_rate
    }

    #[getter]
    pub fn shape(&self) -> (usize, usize) {
        (self.channels, self.samples)
    }

    #[getter]
    pub fn total_bytes(&self) -> usize {
        self.channels * self.samples * std::mem::size_of::<f32>()
    }

    pub fn memoryview<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyMemoryView>> {
        let expected_bytes = self.channels * self.samples * std::mem::size_of::<f32>();
        let slice = &self.mmap[..expected_bytes];

        unsafe {
            let ptr = pyo3::ffi::PyMemoryView_FromMemory(
                slice.as_ptr() as *mut std::ffi::c_char,
                slice.len() as isize,
                pyo3::ffi::PyBUF_READ,
            );
            if ptr.is_null() {
                return Err(PyErr::fetch(py));
            }
            Ok(Bound::from_owned_ptr(py, ptr).cast_into_unchecked())
        }
    }

    pub fn to_numpy<'py>(self_: Bound<'py, Self>, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let np = py.import("numpy")?;
        let (channels, samples) = {
            let inner = self_.borrow();
            (inner.channels, inner.samples)
        };
        let memview = self_.borrow().memoryview(py)?;
        let flat_arr = np.call_method1("frombuffer", (memview, "float32"))?;
        let reshaped = flat_arr.call_method1("reshape", ((channels, samples),))?;
        Ok(reshaped)
    }

    fn __repr__(&self) -> String {
        format!(
            "MmapRecording(path='{}', shape=({}, {}), sample_rate={:.1}Hz, size={:.2}MB)",
            self.path,
            self.channels,
            self.samples,
            self.sample_rate,
            self.mmap.len() as f64 / 1_048_576.0
        )
    }
}
