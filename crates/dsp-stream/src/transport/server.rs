//! The server side: accepts clients and serves a recording to each (header, signal, views).

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use dsp_core::{RecordingInfo, RecordingSource, SampleFormat};
use dsp_view::{Envelope, Pyramid};
use tokio::task::JoinHandle;
use tokio::time::{interval, MissedTickBehavior};

use crate::error::{StreamError, StreamResult};
use crate::protocol::{
    frame_from_envelope, header_from_info, read_message, view_from_request, wire, write_message, DEFAULT_MAX_MESSAGE_BYTES, PROTOCOL_VERSION,
};

/// Seconds of signal per frame by default.
pub const DEFAULT_FRAME_SEC: f64 = 0.02;
/// Frames read ahead per client by default (bounds memory whatever the recording length).
pub const DEFAULT_QUEUE_FRAMES: usize = 8;

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
    /// Largest message accepted from a client.
    pub max_message_bytes: usize,
}

impl Default for ServeOptions {
    fn default() -> Self {
        Self { frame_sec: DEFAULT_FRAME_SEC, pacing: Pacing::RealTime, loop_playback: false, queue_frames: DEFAULT_QUEUE_FRAMES, max_message_bytes: DEFAULT_MAX_MESSAGE_BYTES }
    }
}

/// A QUIC endpoint accepting clients.
pub struct Server {
    endpoint: quinn::Endpoint,
}

impl Server {
    /// Listens on `addr` (UDP).
    pub fn bind(addr: SocketAddr, config: quinn::ServerConfig) -> StreamResult<Self> {
        Ok(Self { endpoint: quinn::Endpoint::server(config, addr)? })
    }

    pub fn local_addr(&self) -> StreamResult<SocketAddr> {
        Ok(self.endpoint.local_addr()?)
    }

    /// The next client whose handshake succeeds; `None` once the endpoint is closed.
    pub async fn accept(&self) -> Option<quinn::Connection> {
        loop {
            match self.endpoint.accept().await?.await {
                Ok(connection) => return Some(connection),
                Err(e) => tracing::warn!("QUIC handshake failed: {e}"),
            }
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
                let view = view_from_request(&request);
                let envelope = tokio::task::spawn_blocking(move || -> StreamResult<Envelope> { Ok(view.read(source.as_ref(), pyramid.as_deref())?) })
                    .await
                    .map_err(StreamError::protocol)?;
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
