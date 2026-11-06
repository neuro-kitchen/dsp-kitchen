//! The client side: a session with one server (who it is, its files, the recording this client
//! opened, the signal, views).

use std::net::{Ipv4Addr, Ipv6Addr, SocketAddr};

use dspk_core::RecordingInfo;
use dspk_view::{Envelope, View};

use crate::error::{StreamError, StreamResult};
use crate::protocol::{
    decode_signal, envelope_from_frame, info_from_header, read_message, request_from_view, wire, write_message, DEFAULT_MAX_MESSAGE_BYTES, PROTOCOL_VERSION,
};

/// A connected session: the server's info, and the recording this client has open.
pub struct Session {
    /// Kept for the connection's lifetime (the endpoint drives it).
    _endpoint: quinn::Endpoint,
    connection: quinn::Connection,
    control: quinn::SendStream,
    /// Kept open: dropping it would ask the server to stop the control stream.
    _control_recv: quinn::RecvStream,
    server: wire::ServerInfo,
    recording: Option<RecordingInfo>,
    max_message_bytes: usize,
}

impl Session {
    /// Connects to `addr` (certificate checked against `server_name` per `config`) and reads the
    /// server's info. The local endpoint uses the address family of `addr`.
    pub async fn connect(addr: SocketAddr, server_name: &str, config: quinn::ClientConfig) -> StreamResult<Self> {
        let local: SocketAddr = if addr.is_ipv6() { (Ipv6Addr::UNSPECIFIED, 0).into() } else { (Ipv4Addr::UNSPECIFIED, 0).into() };
        let mut endpoint = quinn::Endpoint::client(local)?;
        endpoint.set_default_client_config(config);
        let connection = endpoint.connect(addr, server_name)?.await?;
        let (mut control, mut control_recv) = connection.open_bi().await?;
        write_message(&mut control, &wire::ClientHello { protocol_version: PROTOCOL_VERSION }).await?;
        let server: wire::ServerInfo = read_message(&mut control_recv, DEFAULT_MAX_MESSAGE_BYTES).await?.ok_or_else(|| StreamError::protocol("no server info"))?;
        if server.protocol_version != PROTOCOL_VERSION {
            return Err(StreamError::protocol(format!("server speaks protocol {}, this client {PROTOCOL_VERSION}", server.protocol_version)));
        }
        let recording = server.recording.as_ref().map(info_from_header).transpose()?;
        Ok(Self { _endpoint: endpoint, connection, control, _control_recv: control_recv, server, recording, max_message_bytes: DEFAULT_MAX_MESSAGE_BYTES })
    }

    /// Accepts messages up to `bytes` long (default [`DEFAULT_MAX_MESSAGE_BYTES`]).
    pub fn with_max_message_bytes(mut self, bytes: usize) -> Self {
        self.max_message_bytes = bytes;
        self
    }

    /// Who the server is, who this client is to it, and what it may do.
    pub fn server(&self) -> &wire::ServerInfo {
        &self.server
    }

    /// The recording this client has open (channels, units, exact rate, stored format): the
    /// server's initial one, or the last [`Self::open`]ed.
    pub fn recording(&self) -> Option<&RecordingInfo> {
        self.recording.as_ref()
    }

    /// Starts (or restarts) the signal of the open recording at `first_sample`.
    pub async fn subscribe(&mut self, first_sample: u64) -> StreamResult<SignalReceiver> {
        let info = self.recording.clone().ok_or_else(|| StreamError::protocol("no recording is open: open one first"))?;
        write_message(&mut self.control, &wire::Subscribe { first_sample }).await?;
        let recv = self.connection.accept_uni().await?;
        Ok(SignalReceiver { recv, info, max_message_bytes: self.max_message_bytes })
    }

    /// Sends `kind` on a stream of its own (never behind signal data) and reads the answer; a
    /// refusal or failure on the server is an error here.
    async fn request(&self, kind: wire::request::Kind) -> StreamResult<wire::response::Kind> {
        let (mut send, mut recv) = self.connection.open_bi().await?;
        write_message(&mut send, &wire::Request { kind: Some(kind) }).await?;
        send.finish().map_err(StreamError::protocol)?;
        let response: wire::Response = read_message(&mut recv, self.max_message_bytes).await?.ok_or_else(|| StreamError::protocol("no response"))?;
        match response.kind {
            Some(wire::response::Kind::Error(message)) => Err(StreamError::Server(message)),
            Some(kind) => Ok(kind),
            None => Err(StreamError::protocol("empty response")),
        }
    }

    /// The server's answer to `view` of the open recording.
    pub async fn view(&self, view: &View) -> StreamResult<Envelope> {
        match self.request(wire::request::Kind::View(request_from_view(view)?)).await? {
            wire::response::Kind::Envelope(frame) => envelope_from_frame(frame),
            _ => Err(StreamError::protocol("unexpected answer to a view")),
        }
    }

    /// The entries of directory `dir` under the server's data root ("" = the root).
    pub async fn list_files(&self, dir: &str) -> StreamResult<wire::FileList> {
        match self.request(wire::request::Kind::ListFiles(wire::ListFiles { dir: dir.into() })).await? {
            wire::response::Kind::Files(list) => Ok(list),
            _ => Err(StreamError::protocol("unexpected answer to a listing")),
        }
    }

    async fn status_request(&self, kind: wire::request::Kind) -> StreamResult<wire::Status> {
        match self.request(kind).await? {
            wire::response::Kind::Status(status) => Ok(status),
            _ => Err(StreamError::protocol("unexpected answer to a job request")),
        }
    }

    /// Sets the server's job (control permission; only while nothing runs).
    pub async fn configure(&self, job: wire::Configure) -> StreamResult<wire::Status> {
        self.status_request(wire::request::Kind::Configure(job)).await
    }

    /// Starts the configured job (control permission).
    pub async fn start(&self) -> StreamResult<wire::Status> {
        self.status_request(wire::request::Kind::Start(wire::Start {})).await
    }

    /// Asks the running job to stop (control permission).
    pub async fn stop(&self) -> StreamResult<wire::Status> {
        self.status_request(wire::request::Kind::Stop(wire::Stop {})).await
    }

    /// The server's state and its job's progress.
    pub async fn status(&self) -> StreamResult<wire::Status> {
        self.status_request(wire::request::Kind::Status(wire::GetStatus {})).await
    }

    /// The job's last `lines` log lines (0 = all the server keeps).
    pub async fn logs(&self, lines: u32) -> StreamResult<Vec<String>> {
        match self.request(wire::request::Kind::Logs(wire::GetLogs { lines })).await? {
            wire::response::Kind::Logs(logs) => Ok(logs.lines),
            _ => Err(StreamError::protocol("unexpected answer to logs")),
        }
    }

    /// Opens the recording at `path` under the server's data root (`source`: one of the file's
    /// sources, `None` = its default); views and the signal are then of it.
    pub async fn open(&mut self, path: &str, source: Option<&str>) -> StreamResult<&RecordingInfo> {
        let open = wire::Open { path: path.into(), source: source.unwrap_or_default().into() };
        match self.request(wire::request::Kind::Open(open)).await? {
            wire::response::Kind::Header(header) => {
                self.recording = Some(info_from_header(&header)?);
                Ok(self.recording.as_ref().expect("just set"))
            }
            _ => Err(StreamError::protocol("unexpected answer to open")),
        }
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
