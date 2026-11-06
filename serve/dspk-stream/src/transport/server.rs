//! The server side: accepts clients and serves a recording to each (header, signal, views).

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use dspk_core::{RecordingInfo, RecordingSource, SampleFormat};
use dspk_view::{Envelope, Pyramid};
use tokio::task::JoinHandle;
use tokio::time::{interval, MissedTickBehavior};

use super::auth::{Authority, Peer};
use crate::error::{StreamError, StreamResult};
use crate::protocol::{frame_from_envelope, header_from_info, read_message, view_from_request, wire, write_message, PROTOCOL_VERSION};
use dspk_view::View;

/// Seconds of signal per frame by default.
pub const DEFAULT_FRAME_SEC: f64 = 0.02;
/// Frames read ahead per client by default (bounds memory whatever the recording length).
pub const DEFAULT_QUEUE_FRAMES: usize = 8;
/// Largest message a client may send by default: its messages (hello, subscribe, view
/// requests) are a few bytes; the length is checked before anything is allocated.
pub const DEFAULT_MAX_REQUEST_BYTES: usize = 64 << 10;
/// Clients connected at once by default; more are refused until one leaves.
pub const DEFAULT_MAX_CONNECTIONS: usize = 32;
/// Largest view answered: values (channels × columns, or × samples) in one envelope, so a request
/// cannot make the server allocate without bound (32 MiB of f32, under a client's message limit).
pub const MAX_VIEW_VALUES: u64 = 8 << 20;
/// Most columns a view may ask for (a screen is a few thousand pixels wide).
pub const MAX_VIEW_COLUMNS: u32 = 1 << 16;

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
}

/// Closing code for a client whose certificate is not enrolled or is revoked.
pub const REFUSED_CLIENT: u32 = 1;

impl Server {
    /// Listens on `addr` (UDP), at most [`DEFAULT_MAX_CONNECTIONS`] clients at once.
    pub fn bind(addr: SocketAddr, config: quinn::ServerConfig) -> StreamResult<Self> {
        Ok(Self { endpoint: quinn::Endpoint::server(config, addr)?, max_connections: DEFAULT_MAX_CONNECTIONS, authority: None })
    }

    /// Checks every client's certificate against `authority`'s registry after the handshake (use
    /// with a configuration from [`super::tls::server_config_with_clients`]): enrolled and not
    /// revoked, or the connection is closed.
    pub fn with_authority(mut self, authority: Arc<Authority>) -> Self {
        self.authority = Some(authority);
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
            None => Some(Peer::anonymous()),
        }
    }

    /// Stops accepting and closes every connection.
    pub fn close(&self) {
        self.endpoint.close(quinn::VarInt::from_u32(0), b"server closed");
    }
}

/// What the signal stream carries: stored words when the source can read them, else scaled
/// `f32` (and the header then says `f32`, gain 1, offset 0).
fn streamed_info(source: &dyn RecordingSource) -> (RecordingInfo, bool) {
    let info = source.info().clone();
    let all: Vec<usize> = (0..info.channel_count()).collect();
    // One sample of every channel tells whether stored reads are supported
    let mut probe = vec![0u8; all.len() * info.format.bytes()];
    let stored = info.samples == 0 || source.read_stored(&all, 0..1, &mut probe).is_ok();
    if stored {
        return (info, true);
    }
    let mut scaled = info;
    scaled.format = SampleFormat::F32;
    for c in &mut scaled.channels {
        (c.gain, c.offset) = (1.0, 0.0);
    }
    (scaled, false)
}

/// Serves `source` to one client until it closes the control stream or disconnects: answers
/// its hello with the header, streams the signal from where it subscribes, and answers every
/// view (from `pyramid` when given, else from raw reads).
pub async fn serve_recording(connection: quinn::Connection, source: Arc<dyn RecordingSource>, pyramid: Option<Arc<Pyramid>>, options: ServeOptions) -> StreamResult<()> {
    let (mut control_send, mut control_recv) = connection.accept_bi().await?;
    let hello: wire::ClientHello = read_message(&mut control_recv, options.max_message_bytes).await?.ok_or_else(|| StreamError::protocol("no hello"))?;
    let (info, stored) = streamed_info(source.as_ref());
    write_message(&mut control_send, &header_from_info(&info)).await?;
    if hello.protocol_version != PROTOCOL_VERSION {
        return Err(StreamError::protocol(format!("client speaks protocol {}, this server {PROTOCOL_VERSION}", hello.protocol_version)));
    }

    let views = tokio::spawn(answer_views(connection.clone(), source.clone(), pyramid, options.max_message_bytes));
    let mut signal: Option<JoinHandle<StreamResult<()>>> = None;
    let result = loop {
        match read_message::<wire::Subscribe>(&mut control_recv, options.max_message_bytes).await {
            Ok(Some(subscribe)) => {
                if let Some(old) = signal.take() {
                    old.abort();
                }
                let task = stream_signal(connection.clone(), source.clone(), info.clone(), stored, subscribe.first_sample, options.clone());
                signal = Some(tokio::spawn(task));
            }
            Ok(None) => break Ok(()),
            Err(e) => break Err(e),
        }
    };
    if let Some(task) = signal {
        task.abort();
    }
    views.abort();
    result
}

/// Answers each view stream the client opens with one envelope.
async fn answer_views(connection: quinn::Connection, source: Arc<dyn RecordingSource>, pyramid: Option<Arc<Pyramid>>, max_message_bytes: usize) {
    while let Ok((mut send, mut recv)) = connection.accept_bi().await {
        let (source, pyramid) = (source.clone(), pyramid.clone());
        tokio::spawn(async move {
            let answer = async {
                let request: wire::ViewRequest = read_message(&mut recv, max_message_bytes).await?.ok_or_else(|| StreamError::protocol("empty view stream"))?;
                let envelope = match checked_view(&request, source.info()) {
                    Ok(view) => tokio::task::spawn_blocking(move || -> StreamResult<Envelope> { Ok(view.read(source.as_ref(), pyramid.as_deref())?) })
                        .await
                        .map_err(StreamError::protocol)?,
                    Err(e) => Err(e),
                };
                write_message(&mut send, &frame_from_envelope(envelope)).await?;
                send.finish().map_err(StreamError::protocol)?;
                StreamResult::Ok(())
            };
            if let Err(e) = answer.await {
                tracing::debug!("view stream closed: {e}");
            }
        });
    }
}

/// The view a client asks for, refused when it is outside the recording or larger than the
/// server answers: channels in range and each once, at most [`MAX_VIEW_COLUMNS`] columns and
/// [`MAX_VIEW_VALUES`] values (channels × the columns, or the samples when fewer).
pub fn checked_view(request: &wire::ViewRequest, info: &RecordingInfo) -> StreamResult<View> {
    let view = view_from_request(request);
    view.validate(info.channel_count(), info.samples).map_err(|e| StreamError::protocol(e.to_string()))?;
    let mut seen = vec![false; info.channel_count()];
    for &c in &view.channels {
        if std::mem::replace(&mut seen[c], true) {
            return Err(StreamError::protocol(format!("channel {c} asked for twice")));
        }
    }
    if request.width > MAX_VIEW_COLUMNS {
        return Err(StreamError::protocol(format!("{} columns asked for: at most {MAX_VIEW_COLUMNS}", request.width)));
    }
    let values = view.channels.len() as u64 * (view.end - view.start).min(u64::from(request.width));
    if values > MAX_VIEW_VALUES {
        return Err(StreamError::protocol(format!("a view of {values} values: at most {MAX_VIEW_VALUES} (fewer channels or columns)")));
    }
    Ok(view)
}

/// Streams frames from `first_sample` on a new unidirectional stream. A reader thread fills a
/// bounded queue, so the recording is read ahead of the network but never held whole; the thread
/// ends by itself once the queue is dropped.
async fn stream_signal(connection: quinn::Connection, source: Arc<dyn RecordingSource>, info: RecordingInfo, stored: bool, first_sample: u64, options: ServeOptions) -> StreamResult<()> {
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
