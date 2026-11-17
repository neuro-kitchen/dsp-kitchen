//! A server's plugins: what it computes for a domain the stream knows nothing of (Curation of a
//! sorting, for electrophysiology). A client sends a `PluginCall` (plugin id, method, body); the
//! server hands it to the plugin of that id with its data root and the recording it has open, and
//! sends back the plugin's bytes. Plugins are registered by whoever builds the server (`dspk serve`),
//! as recordings are opened by its `RecordingOpener`.

use std::path::PathBuf;
use std::sync::Arc;

use dspk_core::RecordingSource;

use crate::error::StreamResult;

/// What a plugin call may use of the server.
pub struct PluginContext {
    /// Paths under the data root, checked (a path outside it is refused); `None`: no files served.
    pub resolve: Option<Box<dyn Fn(&str) -> StreamResult<PathBuf> + Send + Sync>>,
    /// The recording the client has open, as stored.
    pub recording: Option<Arc<dyn RecordingSource>>,
    /// The client's name, as paired (what a plugin records as who did something).
    pub peer: String,
    /// The client was paired with control: it may change files (a plugin refuses writes else).
    pub may_write: bool,
}

/// One plugin of a server.
pub trait ServerPlugin: Send + Sync {
    /// Its id, as clients name it (`"synapse"`).
    fn id(&self) -> &'static str;

    /// Answers `method` with `body`; run on a blocking thread (it may compute for seconds).
    fn call(&self, method: &str, body: &[u8], context: &PluginContext) -> StreamResult<Vec<u8>>;
}
