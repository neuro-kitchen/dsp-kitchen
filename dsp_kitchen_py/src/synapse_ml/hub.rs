//! PyO3 bindings for the `dsp-synapse-ml` Pretrained Electrophysiology Model Hub (`ModelHub`).
//!
//! Because `EMBEDDED_MODELS_JSON` is compiled directly into `dsp-synapse-ml` via `include_str!`,
//! `ModelHub` works from any working directory without needing filesystem path lookups to find
//! `catalog/models.json`.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use dsp_synapse_ml::hub::{EMBEDDED_MODELS_JSON, ModelHub, ModelHubEntry, ModelStatus};

/// Pretrained Electrophysiology Model Hub exposed to Python (`dsp_kitchen.synapse.ml.ModelHub`).
#[pyclass(name = "ModelHub", skip_from_py_object)]
#[derive(Clone)]
pub struct PyModelHub {
    pub inner: ModelHub,
}

#[pymethods]
impl PyModelHub {
    #[new]
    pub fn new() -> PyResult<Self> {
        let inner = ModelHub::new()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Returns the raw embedded JSON catalog string compiled into `dsp-synapse-ml`.
    #[staticmethod]
    pub fn embedded_catalog_json() -> &'static str {
        EMBEDDED_MODELS_JSON
    }

    /// Root directory of the local Hub weight cache (`~/.cache/dsp-kitchen/hub` or `DSP_KITCHEN_HUB_DIR`).
    #[getter]
    pub fn cache_dir(&self) -> String {
        self.inner.cache().root_dir().to_string_lossy().into_owned()
    }

    /// Lists all models in the catalog (optionally filtered by `family` or `installed=True`).
    #[pyo3(signature = (family=None, installed=false))]
    pub fn list<'py>(
        &self,
        py: Python<'py>,
        family: Option<&str>,
        installed: bool,
    ) -> PyResult<Vec<Bound<'py, PyDict>>> {
        let mut entries = self.inner.list();
        if let Some(fam) = family {
            entries.retain(|e| e.manifest.family.eq_ignore_ascii_case(fam));
        }
        if installed {
            entries.retain(|e| e.status == ModelStatus::Installed);
        }
        entries.iter().map(|e| entry_to_pydict(py, e)).collect()
    }

    /// Returns a dictionary containing the full manifest, resolved download URL, local cache path,
    /// and installation status for `model_id`.
    pub fn info<'py>(&self, py: Python<'py>, model_id: &str) -> PyResult<Bound<'py, PyDict>> {
        let entry = self
            .inner
            .info(model_id)
            .map_err(|e| pyo3::exceptions::PyKeyError::new_err(e.to_string()))?;
        entry_to_pydict(py, &entry)
    }

    /// Downloads and SHA-256 verifies a model's weights into the local Hub cache, returning its info dict.
    #[pyo3(signature = (model_id, force=false))]
    pub fn pull<'py>(
        &self,
        py: Python<'py>,
        model_id: &str,
        force: bool,
    ) -> PyResult<Bound<'py, PyDict>> {
        let hub = self.inner.clone();
        let id = model_id.to_string();
        let entry = py
            .detach(move || hub.pull(&id, force))
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        entry_to_pydict(py, &entry)
    }

    /// Resolves the local cached weights path for `model_id` (pulling it first if `auto_pull=True` and not installed).
    #[pyo3(signature = (model_id, auto_pull=false))]
    pub fn weights_path(&self, py: Python<'_>, model_id: &str, auto_pull: bool) -> PyResult<String> {
        let entry = if auto_pull {
            let hub = self.inner.clone();
            let id = model_id.to_string();
            py.detach(move || hub.pull(&id, false))
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?
        } else {
            self.inner
                .info(model_id)
                .map_err(|e| pyo3::exceptions::PyKeyError::new_err(e.to_string()))?
        };
        Ok(entry.local_weights_path.to_string_lossy().into_owned())
    }
}

fn entry_to_pydict<'py>(py: Python<'py>, entry: &ModelHubEntry) -> PyResult<Bound<'py, PyDict>> {
    let m = &entry.manifest;
    let d = PyDict::new(py);
    d.set_item("id", &m.id)?;
    d.set_item("name", &m.name)?;
    d.set_item("family", &m.family)?;
    d.set_item("task", &m.task)?;
    d.set_item("version", &m.version)?;
    d.set_item("format", m.format.extension())?;
    d.set_item("description", &m.description)?;
    d.set_item("upstream_repo", &m.upstream_repo)?;
    d.set_item("paper_url", &m.paper_url)?;
    d.set_item("license", &m.license)?;
    d.set_item("weights_url", &m.weights_url)?;
    d.set_item("resolved_download_url", &entry.resolved_download_url)?;
    d.set_item("sha256", &m.sha256)?;
    d.set_item("size_bytes", m.size_bytes)?;
    d.set_item("status", entry.status.badge())?;
    d.set_item("installed", entry.status == ModelStatus::Installed)?;
    d.set_item(
        "local_weights_path",
        entry.local_weights_path.to_string_lossy().as_ref(),
    )?;

    let io_dict = PyDict::new(py);
    io_dict.set_item("sample_rate_hz", m.io_spec.sample_rate_hz)?;
    io_dict.set_item("normalization", &m.io_spec.normalization)?;

    let inputs_list = PyList::empty(py);
    for inp in &m.io_spec.inputs {
        let p = PyDict::new(py);
        p.set_item("name", &inp.name)?;
        p.set_item("dtype", &inp.dtype)?;
        p.set_item("shape", &inp.shape)?;
        p.set_item("description", inp.description.as_deref())?;
        inputs_list.append(p)?;
    }
    io_dict.set_item("inputs", inputs_list)?;

    let outputs_list = PyList::empty(py);
    for out in &m.io_spec.outputs {
        let p = PyDict::new(py);
        p.set_item("name", &out.name)?;
        p.set_item("dtype", &out.dtype)?;
        p.set_item("shape", &out.shape)?;
        p.set_item("description", out.description.as_deref())?;
        outputs_list.append(p)?;
    }
    io_dict.set_item("outputs", outputs_list)?;
    d.set_item("io_spec", io_dict)?;

    Ok(d)
}
