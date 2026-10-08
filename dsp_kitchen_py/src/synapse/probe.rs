//! Probe layouts (dsp-io `neuro::probe`): presets, positions, or the geometry a recording carries.

use std::path::Path;

use dsp_io::neuro::probe::{
    find_k_nearest_neighbors, hdemg_4x8, hdemg_8x8, hdemg_grid, neuropixels_1_0, neuropixels_2_0, probe_of, tetrode, utah_array, Position3D, SensorLayout,
    SensorSite,
};
use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use pyo3::types::PyDict;

/// Dimensions of contact positions in `to_dict` (x, y, z).
const POSITION_DIMS: usize = 3;

/// Positions of a probe's contacts, in µm, one per recording channel (channel `i` is contact `i`).
///
/// Sorters and spatial operators need it to know which channels are neighbours. Make one from a
/// preset, from a list of positions, or from the geometry a recording file carries.
///
/// Examples
/// --------
/// >>> import dsp_kitchen.synapse as syn
/// >>> probe = syn.ProbeLayout.neuropixels_1_0()
/// >>> probe = syn.ProbeLayout.from_recording("session.ap.bin")        # None when the file has none
/// >>> probe = syn.ProbeLayout.hdemg_grid("forearm", 4, 8, 8000.0)      # 8 mm pitch
/// >>> probe = syn.ProbeLayout.from_positions("custom", [(0, 0), (0, 25), (0, 50)])
#[gen_stub_pyclass]
#[pyclass(name = "ProbeLayout", skip_from_py_object)]
#[derive(Clone)]
pub struct PyProbeLayout {
    pub inner: SensorLayout,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyProbeLayout {
    /// Neuropixels 1.0: 384 channels on one shank, 20 µm rows, staggered columns (x = 43 / 11 µm on even rows, 59 / 27 µm on odd rows; probeinterface `NP1010`).
    #[staticmethod]
    pub fn neuropixels_1_0() -> Self {
        Self {
            inner: neuropixels_1_0(),
        }
    }

    /// Neuropixels 2.0: four shanks of 96 sites (384 channels), two columns 32 µm apart, 15 µm rows, shanks 250 µm apart, numbered shank by shank.
    #[staticmethod]
    pub fn neuropixels_2_0() -> Self {
        Self {
            inner: neuropixels_2_0(),
        }
    }

    /// Tetrode: four wires on a cross, ±12.5 µm along x then y.
    #[staticmethod]
    pub fn tetrode() -> Self {
        Self {
            inner: tetrode(),
        }
    }

    /// Utah array: 10 × 10 at 400 µm without the four corners (96 channels), numbered row-major.
    #[staticmethod]
    pub fn utah_array() -> Self {
        Self {
            inner: utah_array(),
        }
    }

    /// 32-channel 4 × 8 HD-EMG grid.
    ///
    /// Parameters
    /// ----------
    /// pitch_um : float
    ///     Distance between neighbouring electrodes, µm.
    #[staticmethod]
    pub fn hdemg_4x8(pitch_um: f32) -> Self {
        Self {
            inner: hdemg_4x8(pitch_um),
        }
    }

    /// 64-channel 8 × 8 HD-EMG grid.
    ///
    /// Parameters
    /// ----------
    /// pitch_um : float
    ///     Distance between neighbouring electrodes, µm.
    #[staticmethod]
    pub fn hdemg_8x8(pitch_um: f32) -> Self {
        Self {
            inner: hdemg_8x8(pitch_um),
        }
    }

    /// A planar `rows × cols` grid (HD-EMG, ECoG); channel `r · cols + c` sits at
    /// `(c · pitch_um, r · pitch_um)`.
    ///
    /// Parameters
    /// ----------
    /// name : str
    /// rows, cols : int
    /// pitch_um : float
    ///     Distance between neighbouring electrodes, µm.
    #[staticmethod]
    pub fn hdemg_grid(name: &str, rows: usize, cols: usize, pitch_um: f32) -> Self {
        Self {
            inner: hdemg_grid(name, rows, cols, pitch_um),
        }
    }

    /// A probe from contact positions.
    ///
    /// Parameters
    /// ----------
    /// name : str
    /// positions : list of (float, float)
    ///     `(x, y)` of each contact, µm, in channel order.
    /// shank_ids : list of int, optional
    ///     Shank of each contact (used by the sorters, which place template positions per shank);
    ///     default: one shank.
    #[staticmethod]
    #[pyo3(signature = (name, positions, shank_ids=None))]
    pub fn from_positions(name: &str, positions: Vec<(f32, f32)>, shank_ids: Option<Vec<usize>>) -> Self {
        let contacts = positions
            .into_iter()
            .enumerate()
            .map(|(idx, (x, y))| {
                let shank = shank_ids.as_ref().and_then(|s| s.get(idx)).copied().unwrap_or(0);
                SensorSite {
                    channel_id: idx,
                    device_index: idx,
                    group_id: shank,
                    shank_id: shank,
                    position: Position3D::new(x, y, 0.0),
                    enabled: true,
                }
            })
            .collect();
        Self {
            inner: SensorLayout::new(name, contacts),
        }
    }

    /// The geometry a recording file carries (e.g. a SpikeGLX `.meta`).
    ///
    /// Parameters
    /// ----------
    /// path : str
    /// source : str, optional
    ///     Which signal of a multi-signal file (see `list_sources`); default: the main one.
    ///
    /// Returns
    /// -------
    /// ProbeLayout or None
    ///     `None` when the format stores no geometry.
    #[staticmethod]
    #[pyo3(signature = (path, source=None))]
    pub fn from_recording(path: &str, source: Option<&str>) -> PyResult<Option<Self>> {
        let p = Path::new(path);
        let id = match source {
            Some(id) => id.to_string(),
            None => {
                let entries = dsp_io::sources(p).map_err(|e| PyIOError::new_err(format!("{path}: {e}")))?;
                dsp_io::default_source(&entries).map(|e| e.id.clone()).ok_or_else(|| PyValueError::new_err(format!("{path} holds no source")))?
            }
        };
        Ok(probe_of(p, &id).map_err(|e| PyIOError::new_err(format!("{path}: {e}")))?.map(|inner| Self { inner }))
    }

    /// Name of the probe.
    #[getter]
    pub fn name(&self) -> String {
        self.inner.name.clone()
    }

    /// Number of contacts (channels).
    #[getter]
    pub fn total_channels(&self) -> usize {
        self.inner.total_channels()
    }

    /// Number of contacts in use (disabled ones excluded).
    #[getter]
    pub fn active_channels(&self) -> usize {
        self.inner.active_channels()
    }

    /// `(x, y, z)` of every contact, µm, in channel order.
    pub fn contact_positions(&self) -> Vec<[f32; 3]> {
        self.inner
            .contacts
            .iter()
            .map(|c| [c.position.x_um, c.position.y_um, c.position.z_um])
            .collect()
    }

    /// Recording channel of every contact.
    pub fn channel_ids(&self) -> Vec<usize> {
        self.inner.contacts.iter().map(|c| c.channel_id).collect()
    }

    /// Shank of every contact.
    pub fn shank_ids(&self) -> Vec<usize> {
        self.inner.contacts.iter().map(|c| c.shank_id).collect()
    }

    /// The `k` enabled contacts nearest to a channel, nearest first; the channel itself comes first
    /// (distance 0). Shanks are not considered.
    ///
    /// Parameters
    /// ----------
    /// channel_id : int
    /// k : int
    ///
    /// Returns
    /// -------
    /// list of int
    pub fn k_nearest_neighbors(&self, channel_id: usize, k: usize) -> Vec<usize> {
        find_k_nearest_neighbors(&self.inner, channel_id, k)
    }

    /// The layout as a plain dict (`name`, `ndim`, `total_channels`, `contact_positions`, `channel_ids`, `shank_ids`).
    pub fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("name", &self.inner.name)?;
        dict.set_item("ndim", POSITION_DIMS)?;
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

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyProbeLayout>()
}
