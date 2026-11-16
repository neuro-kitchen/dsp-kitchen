//! A QUIC [`Session`] as a [`Link`], so [`RemoteSignal`] reads through it: calls block on the
//! session's async runtime (viewers call them from worker threads).

use std::sync::Arc;

use super::client::Session;
use crate::error::{StreamError, StreamResult};
use crate::protocol::wire;
use crate::remote::{Link, RemoteSignal};

/// A session and the runtime that drives its connection.
pub struct SessionLink {
    session: Arc<Session>,
    handle: tokio::runtime::Handle,
}

impl SessionLink {
    pub fn new(session: Arc<Session>, handle: tokio::runtime::Handle) -> Self {
        Self { session, handle }
    }
}

impl Link for SessionLink {
    fn request(&self, kind: wire::request::Kind) -> StreamResult<wire::response::Kind> {
        self.handle.block_on(self.session.request(kind))
    }

    fn spawn(&self, name: &str, work: Box<dyn FnOnce() + Send>) -> StreamResult<()> {
        std::thread::Builder::new().name(name.into()).spawn(work).map(|_| ()).map_err(|e| StreamError::protocol(format!("no thread: {e}")))
    }
}

impl RemoteSignal {
    /// The recording `session` has open; calls run on `handle` (the runtime that drives the
    /// session's connection).
    pub fn new(session: Arc<Session>, handle: tokio::runtime::Handle) -> StreamResult<Self> {
        let info = session.recording().cloned().ok_or_else(|| StreamError::protocol("no recording is open in this session"))?;
        Ok(Self::over(Arc::new(SessionLink::new(session, handle)), info))
    }
}
