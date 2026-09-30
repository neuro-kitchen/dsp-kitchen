use pyo3::prelude::*;
use pyo3::types::PyDict;
use dsp_core::SensorLayout;
use dsp_synapse::probe::{neuropixels_1_0, neuropixels_2_0, tetrode, utah_array, find_k_nearest_neighbors};

/// Neural Probe Layout representation for Python (SpikeInterface compatible).
#[pyclass(name = "ProbeLayout", skip_from_py_object)]
#[derive(Clone)]
pub struct PyProbeLayout {
    pub inner: SensorLayout,
}

#[pymethods]
impl PyProbeLayout {
    #[staticmethod]
    pub fn neuropixels_1_0() -> Self {
        Self {
            inner: neuropixels_1_0(),
        }
    }

    #[staticmethod]
    pub fn neuropixels_2_0() -> Self {
        Self {
            inner: neuropixels_2_0(),
        }
    }

    #[staticmethod]
    pub fn tetrode() -> Self {
        Self {
            inner: tetrode(),
        }
    }

    #[staticmethod]
    pub fn utah_array() -> Self {
        Self {
            inner: utah_array(),
        }
    }

    #[staticmethod]
    #[pyo3(signature = (name, positions, shank_ids=None))]
    pub fn from_positions(name: &str, positions: Vec<(f32, f32)>, shank_ids: Option<Vec<usize>>) -> Self {
        let contacts = positions
            .into_iter()
            .enumerate()
            .map(|(idx, (x, y))| {
                let shank = shank_ids.as_ref().and_then(|s| s.get(idx)).copied().unwrap_or(0);
                dsp_core::SensorSite {
                    channel_id: idx,
                    device_index: idx,
                    group_id: shank,
                    shank_id: shank,
                    position: dsp_core::Position3D::new(x, y, 0.0),
                    enabled: true,
                }
            })
            .collect();
        Self {
            inner: SensorLayout::new(name, contacts),
        }
    }

    #[getter]
    pub fn name(&self) -> String {
        self.inner.name.clone()
    }

    #[getter]
    pub fn total_channels(&self) -> usize {
        self.inner.total_channels()
    }

    #[getter]
    pub fn active_channels(&self) -> usize {
        self.inner.active_channels()
    }

    pub fn contact_positions(&self) -> Vec<[f32; 3]> {
        self.inner
            .contacts
            .iter()
            .map(|c| [c.position.x_um, c.position.y_um, c.position.z_um])
            .collect()
    }

    pub fn channel_ids(&self) -> Vec<usize> {
        self.inner.contacts.iter().map(|c| c.channel_id).collect()
    }

    pub fn shank_ids(&self) -> Vec<usize> {
        self.inner.contacts.iter().map(|c| c.shank_id).collect()
    }

    pub fn k_nearest_neighbors(&self, channel_id: usize, k: usize) -> Vec<usize> {
        find_k_nearest_neighbors(&self.inner, channel_id, k)
    }

    pub fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("name", &self.inner.name)?;
        dict.set_item("ndim", 3)?;
        dict.set_item("total_channels", self.inner.total_channels())?;
        dict.set_item("contact_positions", self.contact_positions())?;
        dict.set_item("channel_ids", self.channel_ids())?;
        dict.set_item("shank_ids", self.shank_ids())?;
        Ok(dict)
    }

    fn __repr__(&self) -> String {
        format!(
            "ProbeLayout(name='{}', channels={}, active={})",
            self.inner.name,
            self.inner.total_channels(),
            self.inner.active_channels()
        )
    }
}
