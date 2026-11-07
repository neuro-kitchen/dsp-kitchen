//! `dspk.remote`: a client of `dspk serve` — who the server is, its files, recordings read there
//! (a [`Recording`] like a local one, read in small pieces), sorts run there (configured,
//! started, followed, stopped, their logs), results fetched. One connection per `Session`, over
//! the identity file `dspk clients enroll` wrote (mutual TLS: the server knows the client, the
//! client trusts the server through the same file).

use std::collections::HashMap;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, OnceLock};

use dspk_stream::protocol::wire;
use dspk_stream::{client_config_with, ClientIdentity, RemoteSource, ServerTrust};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

use crate::io::Recording;

/// The certificate name of a server when none is given (`dspk serve --name` adds others).
pub const DEFAULT_SERVER_NAME: &str = "localhost";
/// Threads of the network runtime (connections are I/O-bound).
const NETWORK_THREADS: usize = 2;

/// The runtime every session's connection runs on (started on first use).
fn runtime() -> &'static tokio::runtime::Runtime {
    static RUNTIME: OnceLock<tokio::runtime::Runtime> = OnceLock::new();
    RUNTIME.get_or_init(|| tokio::runtime::Builder::new_multi_thread().worker_threads(NETWORK_THREADS).thread_name("dspk-remote").enable_all().build().expect("the network runtime starts"))
}

fn failed(e: impl std::fmt::Display) -> PyErr {
    PyRuntimeError::new_err(e.to_string())
}

fn state(s: i32) -> &'static str {
    match wire::DeviceState::try_from(s) {
        Ok(wire::DeviceState::Stopped) => "stopped",
        Ok(wire::DeviceState::Running) => "running",
        Ok(wire::DeviceState::Error) => "error",
        _ => "unknown",
    }
}

fn status_dict<'py>(py: Python<'py>, s: &wire::Status) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("state", state(s.state))?;
    d.set_item("job", &s.job)?;
    d.set_item("stage", &s.stage)?;
    d.set_item("step", s.step)?;
    d.set_item("steps", s.steps)?;
    d.set_item("done", s.done)?;
    d.set_item("total", s.total)?;
    d.set_item("unit", &s.unit)?;
    d.set_item("message", &s.message)?;
    d.set_item("output", &s.output)?;
    d.set_item("started_by", &s.started_by)?;
    d.set_item("started_unix", s.started_unix)?;
    Ok(d)
}

/// A connection to a `dspk serve` server.
#[gen_stub_pyclass]
#[pyclass(module = "dspk._native.remote", subclass)]
pub struct Session {
    /// `None` once closed. The recording a session opens takes the connection (a server opens
    /// one recording per session), so `open` uses a session of its own.
    inner: Mutex<Option<Arc<dspk_stream::Session>>>,
    address: SocketAddr,
    server_name: String,
    identity: PathBuf,
}

impl Session {
    async fn connect(address: SocketAddr, server_name: &str, identity: &std::path::Path) -> Result<dspk_stream::Session, String> {
        let id = ClientIdentity::from_file(identity).map_err(|e| format!("reading {}: {e}", identity.display()))?;
        let config = client_config_with(&ServerTrust::from_identity(&id), Some(&id)).map_err(|e| e.to_string())?;
        dspk_stream::Session::connect(address, server_name, config).await.map_err(|e| format!("connecting to {address}: {e}"))
    }

    fn session(&self) -> PyResult<Arc<dspk_stream::Session>> {
        self.inner.lock().unwrap_or_else(|e| e.into_inner()).clone().ok_or_else(|| failed("the session is closed"))
    }

    /// Runs the future `f` makes from the session, without the GIL.
    fn call<T, F, Fut>(&self, py: Python<'_>, f: F) -> PyResult<T>
    where
        T: Send,
        F: FnOnce(Arc<dspk_stream::Session>) -> Fut + Send,
        Fut: std::future::Future<Output = dspk_stream::StreamResult<T>>,
    {
        let session = self.session()?;
        py.detach(|| runtime().block_on(f(session))).map_err(failed)
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl Session {
    /// Connects to the server at `address` (`IP:port`) as the client of `identity` (the file
    /// `dspk clients enroll` wrote); `server_name` is a name on the server's certificate.
    #[new]
    #[pyo3(signature = (address, identity, *, server_name=DEFAULT_SERVER_NAME.to_string()))]
    fn new(py: Python<'_>, address: &str, identity: PathBuf, server_name: String) -> PyResult<Self> {
        let address: SocketAddr = address.parse().map_err(|_| PyValueError::new_err(format!("address {address:?}: an IP and port, e.g. 192.168.1.20:50051")))?;
        if !identity.is_file() {
            return Err(PyValueError::new_err(format!("identity file {} not found", identity.display())));
        }
        let session = py.detach(|| runtime().block_on(Self::connect(address, &server_name, &identity))).map_err(failed)?;
        Ok(Self { inner: Mutex::new(Some(Arc::new(session))), address, server_name, identity })
    }

    /// Who the server is and who this client is to it: `server_name`, `server_version`,
    /// `peer_name`, `permission` (`read` or `control`), `state`, `has_files`, `runtime`.
    fn info<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let s = self.session()?;
        let i = s.server();
        let d = PyDict::new(py);
        d.set_item("server_name", &i.server_name)?;
        d.set_item("server_version", &i.server_version)?;
        d.set_item("peer_name", &i.peer_name)?;
        d.set_item("permission", if i.permission == wire::Permission::Control as i32 { "control" } else { "read" })?;
        d.set_item("state", state(i.state))?;
        d.set_item("has_files", i.has_files)?;
        d.set_item("runtime", &i.runtime)?;
        Ok(d)
    }

    /// The entries of `dir` under the server's data root: `name`, `kind` (`directory`,
    /// `recording`, `sorting`, `other`), `size`, `modified_unix`, `directory`.
    #[pyo3(signature = (dir=String::new()))]
    fn files<'py>(&self, py: Python<'py>, dir: String) -> PyResult<Vec<Bound<'py, PyDict>>> {
        let list = self.call(py, move |s| async move { s.list_files(&dir).await })?;
        list.entries
            .iter()
            .map(|e| {
                let d = PyDict::new(py);
                d.set_item("name", &e.name)?;
                d.set_item(
                    "kind",
                    match wire::FileKind::try_from(e.kind) {
                        Ok(wire::FileKind::Directory) => "directory",
                        Ok(wire::FileKind::Recording) => "recording",
                        Ok(wire::FileKind::Sorting) => "sorting",
                        _ => "other",
                    },
                )?;
                d.set_item("size", e.size)?;
                d.set_item("modified_unix", e.modified_unix)?;
                d.set_item("directory", e.directory)?;
                Ok(d)
            })
            .collect()
    }

    /// The recording at `path` under the data root (`source`: one of the file's sources), read on
    /// the server: a `dspk.io.Recording` like a local one, for small reads (windows, snippets);
    /// whole-recording work belongs on the server (`sort`).
    #[pyo3(signature = (path, *, source=None))]
    fn open(&self, py: Python<'_>, path: String, source: Option<String>) -> PyResult<Recording> {
        let (address, server_name, identity) = (self.address, self.server_name.clone(), self.identity.clone());
        let session = py.detach(|| {
            runtime().block_on(async move {
                let mut s = Self::connect(address, &server_name, &identity).await?;
                s.open(&path, source.as_deref()).await.map_err(|e| e.to_string())?;
                Ok::<_, String>(s)
            })
        });
        let session = session.map_err(failed)?;
        let source = RemoteSource::new(Arc::new(session), runtime().handle().clone()).map_err(failed)?;
        Ok(Recording::from_source(Arc::new(source)))
    }

    /// Configures and starts a sort on the server: `sorter` (one the server has) on `recording`
    /// (under the data root; `source` within it), into `output` (`""`: next to the recording,
    /// never overwritten). `params`: sorter flags away from their defaults (name without `--` →
    /// value); `probe`: a preset (`""`: the recording's geometry); `start_s` / `duration_s`: the
    /// part to sort (0, 0: all). Returns the status (`status()`).
    #[pyo3(signature = (sorter, recording, *, output=String::new(), params=HashMap::new(), source=String::new(), probe=String::new(), start_s=0.0, duration_s=0.0))]
    #[allow(clippy::too_many_arguments)]
    fn sort<'py>(
        &self,
        py: Python<'py>,
        sorter: String,
        recording: String,
        output: String,
        params: HashMap<String, String>,
        source: String,
        probe: String,
        start_s: f64,
        duration_s: f64,
    ) -> PyResult<Bound<'py, PyDict>> {
        let job = wire::Configure { sorter, recording, output, params, source, probe, start_sec: start_s, duration_sec: duration_s };
        let status = self.call(py, move |s| async move {
            s.configure(job).await?;
            s.start().await
        })?;
        status_dict(py, &status)
    }

    /// The server's state and its job's progress: `state` (`stopped`, `running`, `error`),
    /// `job`, `stage`, `step` of `steps`, `done` of `total` (`unit`), `message`, `output`,
    /// `started_by`, `started_unix`.
    fn status<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let s = self.call(py, |s| async move { s.status().await })?;
        status_dict(py, &s)
    }

    /// Stops the running job; returns the status.
    fn stop<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let s = self.call(py, |s| async move { s.stop().await })?;
        status_dict(py, &s)
    }

    /// The job's last `lines` log lines (0: all the server keeps).
    #[pyo3(signature = (lines=200))]
    fn logs(&self, py: Python<'_>, lines: u32) -> PyResult<Vec<String>> {
        self.call(py, move |s| async move { s.logs(lines).await })
    }

    /// Fetches `path` (a file, or a directory such as a `.sorting.zarr`) from under the data root
    /// to the new local path `dest` (never overwritten); returns `(files, bytes)`.
    fn fetch(&self, py: Python<'_>, path: String, dest: PathBuf) -> PyResult<(usize, u64)> {
        if dest.exists() {
            return Err(PyValueError::new_err(format!("{} exists: fetch writes a new path", dest.display())));
        }
        self.call(py, move |s| async move { s.fetch(path.trim_end_matches('/'), &dest).await })
    }

    /// Ends the connection (recordings opened from it have their own and stay readable).
    fn close(&self, py: Python<'_>) {
        let taken = self.inner.lock().unwrap_or_else(|e| e.into_inner()).take();
        if let Some(session) = taken.and_then(|s| Arc::try_unwrap(s).ok()) {
            py.detach(|| runtime().block_on(session.close()));
        }
    }

    fn __enter__(slf: Py<Self>) -> Py<Self> {
        slf
    }

    #[pyo3(signature = (*_args))]
    fn __exit__(&self, py: Python<'_>, _args: &Bound<'_, pyo3::types::PyTuple>) {
        self.close(py);
    }
}

pyo3_stub_gen::module_variable!("dspk._native.remote", "DEFAULT_SERVER_NAME", String);

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("DEFAULT_SERVER_NAME", DEFAULT_SERVER_NAME)?;
    m.add_class::<Session>()?;
    Ok(())
}
