//! The server side: accepts clients and serves a recording to each (header, signal, views).

use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use dspk_core::RecordingSource;
use dspk_view::Pyramid;
use tokio::task::JoinHandle;
use tokio::time::{interval, MissedTickBehavior};

use super::auth::{Authority, Peer, Permission};
use super::files::{kind_of, DataRoot};
use super::jobs::Jobs;
use crate::error::{StreamError, StreamResult};
use crate::protocol::{frame_from_envelope, header_from_info, read_message, wire, write_message, PROTOCOL_VERSION};

/// Seconds of signal per frame by default.
pub const DEFAULT_FRAME_SEC: f64 = 0.02;
/// Frames read ahead per client by default (bounds memory whatever the recording length).
pub const DEFAULT_QUEUE_FRAMES: usize = 8;
/// Largest message a client may send by default: its messages (hello, subscribe, view
/// requests) are a few bytes; the length is checked before anything is allocated.
pub const DEFAULT_MAX_REQUEST_BYTES: usize = 64 << 10;
/// Clients connected at once by default; more are refused until one leaves.
pub const DEFAULT_MAX_CONNECTIONS: usize = 32;
pub use crate::protocol::{MAX_VIEW_COLUMNS, MAX_VIEW_VALUES};
pub use crate::engine::{checked_view, OpenRecording, RecordingOpener, ViewProcessor};

/// When frames are sent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pacing {
    /// At the recording's rate, as an acquisition system would.
    RealTime,
    /// As fast as the client takes them (throughput tests, bulk transfer).
    Unpaced,
}

/// How a recording is served.
#[derive(Debug, Clone)]
pub struct ServeOptions {
    /// Seconds of signal per frame.
    pub frame_sec: f64,
    pub pacing: Pacing,
    /// Start over at the end of the recording.
    pub loop_playback: bool,
    /// Frames read ahead per client.
    pub queue_frames: usize,
    /// Largest message accepted from a client (default [`DEFAULT_MAX_REQUEST_BYTES`]).
    pub max_message_bytes: usize,
}

impl Default for ServeOptions {
    fn default() -> Self {
        Self { frame_sec: DEFAULT_FRAME_SEC, pacing: Pacing::RealTime, loop_playback: false, queue_frames: DEFAULT_QUEUE_FRAMES, max_message_bytes: DEFAULT_MAX_REQUEST_BYTES }
    }
}

/// A QUIC endpoint accepting clients.
pub struct Server {
    endpoint: quinn::Endpoint,
    max_connections: usize,
    authority: Option<Arc<Authority>>,
    /// Pairing is answered (the names the server's certificate carries); see [`Self::with_pairing`].
    pairing: Option<Arc<Vec<String>>>,
    /// With pairing: whether a computer without a certificate may also open a session.
    anonymous: bool,
}

/// Closing code for a client whose certificate is not enrolled or is revoked.
pub const REFUSED_CLIENT: u32 = 1;

impl Server {
    /// Listens on `addr` (UDP), at most [`DEFAULT_MAX_CONNECTIONS`] clients at once.
    pub fn bind(addr: SocketAddr, config: quinn::ServerConfig) -> StreamResult<Self> {
        Ok(Self { endpoint: quinn::Endpoint::server(config, addr)?, max_connections: DEFAULT_MAX_CONNECTIONS, authority: None, pairing: None, anonymous: false })
    }

    /// Checks every client's certificate against `authority`'s registry after the handshake (use
    /// with a configuration from [`super::tls::server_config_with_clients`]): enrolled and not
    /// revoked, or the connection is closed.
    pub fn with_authority(mut self, authority: Arc<Authority>) -> Self {
        self.authority = Some(authority);
        self
    }

    /// Answers computers that pair with a code (`transport::pairing`; needs an authority and a
    /// configuration from [`super::tls::server_config_with_pairing`]). `names`: the names this
    /// server's certificate carries, told to the computer so it knows how to connect.
    /// `anonymous`: whether a computer without a certificate may also open a session (read
    /// only); otherwise it can only pair.
    pub fn with_pairing(mut self, names: Vec<String>, anonymous: bool) -> Self {
        self.pairing = Some(Arc::new(names));
        self.anonymous = anonymous;
        self
    }

    /// At most `n` clients at once (at least one).
    pub fn with_max_connections(mut self, n: usize) -> Self {
        self.max_connections = n.max(1);
        self
    }

    pub fn local_addr(&self) -> StreamResult<SocketAddr> {
        Ok(self.endpoint.local_addr()?)
    }

    /// The next client whose handshake succeeds and who is allowed in, and who it is; `None`
    /// once the endpoint is closed. Before any handshake work, a client must prove it owns its
    /// address (QUIC Retry: a spoofed source address gets one small packet, never the server's
    /// answers), and clients beyond the connection limit are refused. With an authority, a
    /// client presenting a certificate must be enrolled and not revoked; one presenting none is
    /// anonymous (only possible when the TLS configuration allows it).
    pub async fn accept(&self) -> Option<(quinn::Connection, Peer)> {
        loop {
            let incoming = self.endpoint.accept().await?;
            if !incoming.remote_address_validated() && incoming.may_retry() {
                if let Err(e) = incoming.retry() {
                    tracing::debug!("QUIC retry failed: {e}");
                }
                continue;
            }
            if self.endpoint.open_connections() >= self.max_connections {
                tracing::warn!("refused {}: {} clients connected (the limit)", incoming.remote_address(), self.max_connections);
                incoming.refuse();
                continue;
            }
            let connection = match incoming.await {
                Ok(connection) => connection,
                Err(e) => {
                    tracing::warn!("QUIC handshake failed: {e}");
                    continue;
                }
            };
            // A computer pairing with a code: answered here, never handed to a session
            if super::pairing::is_pairing(&connection) {
                match (&self.authority, &self.pairing) {
                    (Some(authority), Some(names)) => {
                        let (authority, names) = (authority.clone(), names.clone());
                        tokio::spawn(async move {
                            if let Err(e) = super::pairing::serve_pairing(&authority, &names, &connection).await {
                                tracing::warn!("pairing with {} failed: {e}", connection.remote_address());
                            }
                        });
                    }
                    _ => connection.close(quinn::VarInt::from_u32(REFUSED_CLIENT), b"this server does not pair"),
                }
                continue;
            }
            match self.peer(&connection) {
                Some(peer) => return Some((connection, peer)),
                None => {
                    tracing::warn!("refused {}: its certificate is not enrolled or is revoked", connection.remote_address());
                    connection.close(quinn::VarInt::from_u32(REFUSED_CLIENT), b"client not enrolled or revoked");
                }
            }
        }
    }

    /// Who `connection`'s client is: anonymous without an authority or without a certificate,
    /// else its enrolled record.
    fn peer(&self, connection: &quinn::Connection) -> Option<Peer> {
        let Some(authority) = &self.authority else { return Some(Peer::anonymous()) };
        let certificates = connection.peer_identity().and_then(|id| id.downcast::<Vec<rustls::pki_types::CertificateDer<'static>>>().ok());
        match certificates.as_deref().and_then(|c| c.first()) {
            Some(certificate) => authority.authorize(certificate),
            // With pairing the TLS layer lets certificate-less computers in: only to pair,
            // unless anonymous sessions are allowed
            None if self.pairing.is_some() && !self.anonymous => None,
            None => Some(Peer::anonymous()),
        }
    }

    /// Stops accepting and closes every connection.
    pub fn close(&self) {
        self.endpoint.close(quinn::VarInt::from_u32(0), b"server closed");
    }
}


/// What a server offers every client.
#[derive(Clone)]
pub struct Device {
    /// The server's name (its host) and version, shown to clients.
    pub name: String,
    pub version: String,
    /// The data root and how recordings under it are opened; `None`: no files.
    pub files: Option<(DataRoot, Arc<dyn RecordingOpener>)>,
    /// The recording every client starts with, if any.
    pub recording: Option<Arc<OpenRecording>>,
    /// The jobs it runs for clients (sorts); `None`: none.
    pub jobs: Option<Arc<Jobs>>,
    /// The compute runtime its jobs run on, as clients are told ("" = none or unknown).
    pub runtime: String,
    pub options: ServeOptions,
    /// Its plugins ([`crate::plugin`]): what it computes for domains (Curation of a sorting).
    pub plugins: Vec<Arc<dyn crate::plugin::ServerPlugin>>,
}

/// The recording a session has open (the initial one, or the last one it opened).
pub(crate) type Current = Arc<Mutex<Option<Arc<OpenRecording>>>>;

fn current_recording(current: &Current) -> Option<Arc<OpenRecording>> {
    current.lock().unwrap_or_else(|e| e.into_inner()).clone()
}

/// Serves `source` to one client (the protocol's single-recording form: no files, anonymous).
pub async fn serve_recording(connection: quinn::Connection, source: Arc<dyn RecordingSource>, pyramid: Option<Arc<Pyramid>>, options: ServeOptions) -> StreamResult<()> {
    let device = Device { name: "dspk".into(), version: String::new(), files: None, recording: Some(Arc::new(OpenRecording::new(source, pyramid))), jobs: None, runtime: String::new(), options, plugins: Vec::new() };
    serve_session(connection, Peer::anonymous(), Arc::new(device)).await
}

/// Serves one client until it closes the control stream or disconnects: answers its hello with
/// the server's info, streams the signal of its open recording from where it subscribes, and
/// answers every request (views, files, opening a recording) on its own stream.
pub async fn serve_session(connection: quinn::Connection, peer: Peer, device: Arc<Device>) -> StreamResult<()> {
    let options = device.options.clone();
    let (mut control_send, mut control_recv) = connection.accept_bi().await?;
    let hello: wire::ClientHello = read_message(&mut control_recv, options.max_message_bytes).await?.ok_or_else(|| StreamError::protocol("no hello"))?;
    let info = wire::ServerInfo {
        protocol_version: PROTOCOL_VERSION,
        server_name: device.name.clone(),
        server_version: device.version.clone(),
        peer_name: peer.name.clone(),
        permission: wire_permission(peer.permission) as i32,
        state: device.jobs.as_ref().map_or(wire::DeviceState::Stopped, |j| j.state()) as i32,
        has_files: device.files.is_some(),
        recording: device.recording.as_ref().map(|r| header_from_info(r.info())),
        runtime: device.runtime.clone(),
    };
    write_message(&mut control_send, &info).await?;
    if hello.protocol_version != PROTOCOL_VERSION {
        return Err(StreamError::protocol(format!("client speaks protocol {}, this server {PROTOCOL_VERSION}", hello.protocol_version)));
    }

    let current: Current = Arc::new(Mutex::new(device.recording.clone()));
    let requests = tokio::spawn(answer_requests(connection.clone(), device.clone(), current.clone(), Arc::new(peer)));
    let mut signal: Option<JoinHandle<StreamResult<()>>> = None;
    let result = loop {
        match read_message::<wire::Subscribe>(&mut control_recv, options.max_message_bytes).await {
            Ok(Some(subscribe)) => {
                let Some(recording) = current_recording(&current) else { break Err(StreamError::protocol("subscribe before any recording is open")) };
                if let Some(old) = signal.take() {
                    old.abort();
                }
                let task = stream_signal(connection.clone(), recording, subscribe.first_sample, options.clone());
                signal = Some(tokio::spawn(task));
            }
            Ok(None) => break Ok(()),
            Err(e) if e.is_closed_by_peer() => break Ok(()),
            Err(e) => break Err(e),
        }
    };
    if let Some(task) = signal {
        task.abort();
    }
    requests.abort();
    result
}

fn wire_permission(p: Permission) -> wire::Permission {
    match p {
        Permission::Read => wire::Permission::Read,
        Permission::Control => wire::Permission::Control,
    }
}

/// Answers each request stream the client opens with one response.
async fn answer_requests(connection: quinn::Connection, device: Arc<Device>, current: Current, peer: Arc<Peer>) {
    let max = device.options.max_message_bytes;
    while let Ok((mut send, mut recv)) = connection.accept_bi().await {
        let (device, current, peer) = (device.clone(), current.clone(), peer.clone());
        tokio::spawn(async move {
            let answer = async {
                let request: wire::Request = read_message(&mut recv, max).await?.ok_or_else(|| StreamError::protocol("empty request stream"))?;
                if let Some(wire::request::Kind::Fetch(fetch)) = &request.kind {
                    return send_file(&mut send, &device, &fetch.path).await;
                }
                let kind = match answer(request, &device, &current, &peer).await {
                    Ok(kind) => kind,
                    Err(e) => wire::response::Kind::Error(e.reason()),
                };
                write_message(&mut send, &wire::Response { kind: Some(kind) }).await?;
                send.finish().map_err(StreamError::protocol)?;
                StreamResult::Ok(())
            };
            if let Err(e) = answer.await {
                tracing::debug!("request stream closed: {e}");
            }
        });
    }
}

/// The answer to one request.
pub(crate) async fn answer(request: wire::Request, device: &Device, current: &Current, peer: &Peer) -> StreamResult<wire::response::Kind> {
    use wire::request::Kind as Req;
    use wire::response::Kind as Resp;
    match request.kind.ok_or_else(|| StreamError::protocol("empty request"))? {
        Req::View(view) => {
            let recording = current_recording(current).ok_or_else(|| StreamError::protocol("no recording is open"))?;
            // A view the server does not answer is refused as a request; a read that fails is
            // the envelope's error
            crate::engine::checked_view(&view, recording.info())?;
            // A processed view: its chain made in a blocking task (a whitening's fit takes
            // seconds), its block made ready without holding a thread while the device works,
            // then answered from it (as a browser's engine does)
            if let Some(processing) = view.processing.clone().filter(|p| !p.ops.is_empty()) {
                let made = recording.clone();
                let warmed = match tokio::task::spawn_blocking(move || made.processed(&processing).map(|_| ())).await.map_err(StreamError::protocol)? {
                    Ok(()) => crate::engine::warm_view(&recording, &view).await,
                    Err(e) => Err(e),
                };
                if let Err(e) = warmed {
                    return Ok(Resp::Envelope(frame_from_envelope(Err(e))));
                }
            }
            let envelope = tokio::task::spawn_blocking(move || crate::engine::answer_view(&recording, &view)).await.map_err(StreamError::protocol)?;
            Ok(Resp::Envelope(frame_from_envelope(envelope)))
        }
        Req::ListFiles(list) => {
            let (root, _) = device.files.as_ref().ok_or_else(|| StreamError::protocol("this server serves no files"))?;
            let root = root.clone();
            let files = tokio::task::spawn_blocking(move || root.list(&list.dir)).await.map_err(StreamError::protocol)??;
            Ok(Resp::Files(files))
        }
        Req::Open(open) => {
            let (root, opener) = device.files.clone().ok_or_else(|| StreamError::protocol("this server serves no files"))?;
            let recording = tokio::task::spawn_blocking(move || -> StreamResult<OpenRecording> {
                let path = root.resolve(&open.path)?;
                if kind_of(&path, path.is_dir()) != wire::FileKind::Recording {
                    return Err(StreamError::Protocol(format!("{:?} is not a recording", open.path)));
                }
                opener.open(&path, Some(open.source.as_str()).filter(|s| !s.is_empty()))
            })
            .await
            .map_err(StreamError::protocol)??;
            let header = header_from_info(recording.info());
            *current.lock().unwrap_or_else(|e| e.into_inner()) = Some(Arc::new(recording));
            Ok(Resp::Header(header))
        }
        Req::Configure(configure) => {
            let jobs = jobs_of(device)?;
            let peer = peer.clone();
            Ok(Resp::Status(tokio::task::spawn_blocking(move || jobs.configure(&peer, configure)).await.map_err(StreamError::protocol)??))
        }
        Req::Start(_) => Ok(Resp::Status(jobs_of(device)?.start(peer)?)),
        Req::Stop(_) => Ok(Resp::Status(jobs_of(device)?.stop(peer)?)),
        Req::Status(_) => Ok(Resp::Status(jobs_of(device)?.status())),
        Req::Logs(logs) => Ok(Resp::Logs(wire::LogLines { lines: jobs_of(device)?.logs(logs.lines as usize) })),
        Req::Fetch(_) => Err(StreamError::protocol("fetch is answered by send_file")),
        Req::Plugin(call) => {
            let plugin = device.plugins.iter().find(|p| p.id() == call.plugin).cloned().ok_or_else(|| StreamError::Protocol(format!("this server has no plugin {:?}", call.plugin)))?;
            let root = device.files.as_ref().map(|(root, _)| root.clone());
            let recording = current_recording(current).map(|r| r.source.clone());
            let context = crate::plugin::PluginContext { resolve: root.map(|r| Box::new(move |p: &str| r.resolve(p)) as Box<dyn Fn(&str) -> StreamResult<std::path::PathBuf> + Send + Sync>), recording };
            let body = tokio::task::spawn_blocking(move || plugin.call(&call.method, &call.body, &context)).await.map_err(StreamError::protocol)??;
            Ok(Resp::Plugin(wire::PluginReply { body }))
        }
    }
}

/// Bytes per FileChunk.
pub const FILE_CHUNK_BYTES: usize = 1 << 20;

/// Answers a Fetch: the file's header, then its bytes in chunks; a refusal as an error response.
async fn send_file(send: &mut quinn::SendStream, device: &Device, path: &str) -> StreamResult<()> {
    use tokio::io::AsyncReadExt as _;
    let opened = async {
        let (root, _) = device.files.as_ref().ok_or_else(|| StreamError::Server("this server serves no files".into()))?;
        let full = root.resolve(path)?;
        if !full.is_file() {
            return Err(StreamError::Server(format!("{path:?} is not a file (fetch a directory file by file)")));
        }
        let file = tokio::fs::File::open(&full).await?;
        let size = file.metadata().await?.len();
        Ok((file, size, root.relative(&full)))
    };
    let (mut file, size, relative) = match opened.await {
        Ok(o) => o,
        Err(e) => {
            write_message(send, &wire::Response { kind: Some(wire::response::Kind::Error(e.reason())) }).await?;
            return send.finish().map_err(StreamError::protocol);
        }
    };
    write_message(send, &wire::Response { kind: Some(wire::response::Kind::File(wire::FileHeader { path: relative, size })) }).await?;
    let mut buf = vec![0u8; FILE_CHUNK_BYTES];
    loop {
        let n = file.read(&mut buf).await?;
        if n == 0 {
            break;
        }
        write_message(send, &wire::FileChunk { data: buf[..n].to_vec() }).await?;
    }
    send.finish().map_err(StreamError::protocol)
}

fn jobs_of(device: &Device) -> StreamResult<Arc<Jobs>> {
    device.jobs.clone().ok_or_else(|| StreamError::Server("this server runs no jobs".into()))
}


/// Streams frames from `first_sample` on a new unidirectional stream. A reader thread fills a
/// bounded queue, so the recording is read ahead of the network but never held whole; the thread
/// ends by itself once the queue is dropped.
async fn stream_signal(connection: quinn::Connection, recording: Arc<OpenRecording>, first_sample: u64, options: ServeOptions) -> StreamResult<()> {
    let (source, info, stored) = (recording.source.clone(), recording.info().clone(), recording.stored);
    let total = info.samples;
    let frame = ((options.frame_sec * info.sample_rate_hz()).round() as u64).max(1);
    let channels: Vec<usize> = (0..info.channel_count()).collect();
    let bytes_per_value = info.format.bytes();
    let (tx, mut rx) = tokio::sync::mpsc::channel::<StreamResult<(u64, u32, Vec<u8>)>>(options.queue_frames.max(1));
    let loop_playback = options.loop_playback;
    std::thread::spawn(move || {
        let mut start = first_sample.min(total);
        loop {
            if start >= total {
                if !loop_playback || total == 0 {
                    return;
                }
                start = 0;
            }
            let end = (start + frame).min(total);
            let n = (end - start) as usize;
            let mut bytes = vec![0u8; channels.len() * n * bytes_per_value];
            let read = if stored {
                source.read_stored(&channels, start..end, &mut bytes)
            } else {
                let mut values = vec![0.0f32; channels.len() * n];
                source.read(&channels, start..end, &mut values).map(|()| {
                    for (dst, v) in bytes.chunks_exact_mut(bytes_per_value).zip(values) {
                        dst.copy_from_slice(&v.to_le_bytes());
                    }
                })
            };
            let message = read.map(|()| (start, n as u32, bytes)).map_err(StreamError::from);
            let failed = message.is_err();
            if tx.blocking_send(message).is_err() || failed {
                return;
            }
            start = end;
        }
    });

    let mut send = connection.open_uni().await?;
    let mut ticker = interval(Duration::from_secs_f64(frame as f64 / info.sample_rate_hz()));
    ticker.set_missed_tick_behavior(MissedTickBehavior::Skip);
    let mut sequence = 0u64;
    async {
        while let Some(chunk) = rx.recv().await {
            let (first_sample, samples, data) = chunk?;
            if options.pacing == Pacing::RealTime {
                ticker.tick().await;
            }
            let sent_unix_ns = SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_nanos() as u64);
            write_message(&mut send, &wire::SignalFrame { sequence, first_sample, samples, data, sent_unix_ns }).await?;
            sequence += 1;
        }
        send.finish().map_err(StreamError::protocol)?;
        StreamResult::Ok(())
    }
    .await
}
