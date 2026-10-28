//! The client side: a session with one server (its recording's description, the signal, views).

use std::net::{Ipv4Addr, Ipv6Addr, SocketAddr};

use dsp_core::RecordingInfo;
use dsp_view::{Envelope, View};

use crate::error::{StreamError, StreamResult};
use crate::protocol::{
    decode_signal, envelope_from_frame, info_from_header, read_message, request_from_view, wire, write_message, DEFAULT_MAX_MESSAGE_BYTES, PROTOCOL_VERSION,
};

/// A connected session: the server's recording, as described by its header.
pub struct Session {
    /// Kept for the connection's lifetime (the endpoint drives it).
    _endpoint: quinn::Endpoint,
    connection: quinn::Connection,
    control: quinn::SendStream,
    /// Kept open: dropping it would ask the server to stop the control stream.
    _control_recv: quinn::RecvStream,
    info: RecordingInfo,
    max_message_bytes: usize,
}

impl Session {
    /// Connects to `addr` (certificate checked against `server_name` per `config`) and reads the
    /// header. The local endpoint uses the address family of `addr`.
    pub async fn connect(addr: SocketAddr, server_name: &str, config: quinn::ClientConfig) -> StreamResult<Self> {
        let local: SocketAddr = if addr.is_ipv6() { (Ipv6Addr::UNSPECIFIED, 0).into() } else { (Ipv4Addr::UNSPECIFIED, 0).into() };
        let mut endpoint = quinn::Endpoint::client(local)?;
        endpoint.set_default_client_config(config);
        let connection = endpoint.connect(addr, server_name)?.await?;
        let (mut control, mut control_recv) = connection.open_bi().await?;
        write_message(&mut control, &wire::ClientHello { protocol_version: PROTOCOL_VERSION }).await?;
        let header: wire::Header = read_message(&mut control_recv, DEFAULT_MAX_MESSAGE_BYTES).await?.ok_or_else(|| StreamError::protocol("no header"))?;
        let info = info_from_header(&header)?;
        Ok(Self { _endpoint: endpoint, connection, control, _control_recv: control_recv, info, max_message_bytes: DEFAULT_MAX_MESSAGE_BYTES })
    }

    /// Accepts messages up to `bytes` long (default [`DEFAULT_MAX_MESSAGE_BYTES`]).
    pub fn with_max_message_bytes(mut self, bytes: usize) -> Self {
        self.max_message_bytes = bytes;
        self
    }

    /// The recording the server streams (channels, units, exact rate, stored format).
    pub fn info(&self) -> &RecordingInfo {
        &self.info
    }

    /// Starts (or restarts) the signal at `first_sample`.
    pub async fn subscribe(&mut self, first_sample: u64) -> StreamResult<SignalReceiver> {
        write_message(&mut self.control, &wire::Subscribe { first_sample }).await?;
        let recv = self.connection.accept_uni().await?;
        Ok(SignalReceiver { recv, info: self.info.clone(), max_message_bytes: self.max_message_bytes })
    }

    /// The server's answer to `view`, on a stream of its own (never behind signal data).
    pub async fn view(&self, view: &View) -> StreamResult<Envelope> {
        let (mut send, mut recv) = self.connection.open_bi().await?;
        write_message(&mut send, &request_from_view(view)?).await?;
        send.finish().map_err(StreamError::protocol)?;
        let frame: wire::EnvelopeFrame = read_message(&mut recv, self.max_message_bytes).await?.ok_or_else(|| StreamError::protocol("no envelope"))?;
        envelope_from_frame(frame)
    }

    /// Ends the session (the server stops streaming).
    pub async fn close(mut self) {
        let _ = self.control.finish();
        self.connection.close(quinn::VarInt::from_u32(0), b"client closed");
        self._endpoint.wait_idle().await;
    }
}

/// Frames of the signal stream.
pub struct SignalReceiver {
    recv: quinn::RecvStream,
    info: RecordingInfo,
    max_message_bytes: usize,
}

impl SignalReceiver {
    /// The next frame as sent (stored words); `None` at the end of the recording.
    pub async fn next_frame(&mut self) -> StreamResult<Option<wire::SignalFrame>> {
        read_message(&mut self.recv, self.max_message_bytes).await
    }

    /// The next frame's first sample and scaled values (`channels × samples`, channel-major).
    pub async fn next_values(&mut self) -> StreamResult<Option<(u64, Vec<f32>)>> {
        match self.next_frame().await? {
            Some(frame) => Ok(Some((frame.first_sample, decode_signal(&frame, &self.info)?))),
            None => Ok(None),
        }
    }
}
