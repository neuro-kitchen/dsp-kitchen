//! The client's side of a device connection.

use std::net::{Ipv4Addr, Ipv6Addr, SocketAddr};

use tokio::sync::{broadcast, Mutex};

use std::sync::Arc;
use std::time::Duration;

use super::super::{unix_now_ns, wire, ClockEstimate, ClockSample, LiveError, PROTOCOL_VERSION};
use crate::error::{StreamError, StreamResult};
use crate::protocol::{read_message, write_message, DEFAULT_MAX_MESSAGE_BYTES};

/// Device events buffered per client before the oldest are dropped.
const EVENT_QUEUE: usize = 256;
/// How often the clocks are compared (each exchange also reports the last estimate to the device).
const CLOCK_EVERY: Duration = Duration::from_secs(1);

/// A connection to one device.
pub struct DeviceClient {
    /// Kept for the connection's lifetime (the endpoint drives it).
    _endpoint: quinn::Endpoint,
    connection: quinn::Connection,
    welcome: wire::DeviceWelcome,
    events: broadcast::Sender<wire::DeviceEvent>,
    /// One subscription at a time, so the next stream the device opens is the one asked for.
    subscribing: Mutex<()>,
    clock: Arc<std::sync::Mutex<ClockEstimate>>,
    max_message_bytes: usize,
}

/// One clock exchange on `connection`, reporting `estimate`'s best so far, and its result added to it.
async fn exchange(connection: &quinn::Connection, estimate: &std::sync::Mutex<ClockEstimate>, max: usize) -> StreamResult<ClockSample> {
    let best = estimate.lock().unwrap_or_else(|e| e.into_inner()).best();
    let ping = wire::ClockPing { client_sent_ns: unix_now_ns(), last_offset_ns: best.map_or(0, |b| b.offset_ns), last_round_trip_ns: best.map_or(0, |b| b.round_trip_ns) };
    let (mut send, mut recv) = connection.open_bi().await?;
    write_message(&mut send, &wire::LiveRequest { kind: Some(wire::live_request::Kind::Clock(ping)) }).await?;
    send.finish().map_err(StreamError::protocol)?;
    let response: wire::LiveResponse = read_message(&mut recv, max).await?.ok_or_else(|| StreamError::protocol("no answer"))?;
    let Some(wire::live_response::Kind::Clock(p)) = response.kind else { return Err(StreamError::protocol("expected a clock answer")) };
    let sample = ClockSample::from_exchange(p.client_sent_ns, p.device_received_unix_ns, p.device_sent_unix_ns, unix_now_ns());
    estimate.lock().unwrap_or_else(|e| e.into_inner()).add(sample);
    Ok(sample)
}

impl DeviceClient {
    /// Connects to `addr` (its certificate checked against `server_name` per `config`, from
    /// [`crate::transport::live_client_config`]) and reads the device's welcome.
    pub async fn connect(addr: SocketAddr, server_name: &str, config: quinn::ClientConfig) -> StreamResult<Self> {
        let local: SocketAddr = if addr.is_ipv6() { (Ipv6Addr::UNSPECIFIED, 0).into() } else { (Ipv4Addr::UNSPECIFIED, 0).into() };
        let mut endpoint = quinn::Endpoint::client(local)?;
        endpoint.set_default_client_config(config);
        let connection = endpoint.connect(addr, server_name)?.await?;
        let (mut control, mut control_recv) = connection.open_bi().await?;
        write_message(&mut control, &wire::DeviceHello { protocol_version: PROTOCOL_VERSION, client_version: env!("CARGO_PKG_VERSION").into() }).await?;
        let welcome: wire::DeviceWelcome = read_message(&mut control_recv, DEFAULT_MAX_MESSAGE_BYTES).await?.ok_or_else(|| StreamError::protocol("no welcome"))?;
        if welcome.protocol_version != PROTOCOL_VERSION {
            return Err(StreamError::protocol(format!("the device speaks live protocol {}, this client {PROTOCOL_VERSION}", welcome.protocol_version)));
        }
        let (events, _) = broadcast::channel(EVENT_QUEUE);
        let sender = events.clone();
        tokio::spawn(async move {
            // The control stream's send half lives as long as this task: the device keeps writing
            let _control = control;
            while let Ok(Some(event)) = read_message::<wire::DeviceEvent>(&mut control_recv, DEFAULT_MAX_MESSAGE_BYTES).await {
                let _ = sender.send(event);
            }
        });
        // The clocks, compared every second while connected
        let clock = Arc::new(std::sync::Mutex::new(ClockEstimate::default()));
        let (c, est) = (connection.clone(), clock.clone());
        tokio::spawn(async move {
            while c.close_reason().is_none() {
                if exchange(&c, &est, DEFAULT_MAX_MESSAGE_BYTES).await.is_err() && c.close_reason().is_some() {
                    break;
                }
                tokio::time::sleep(CLOCK_EVERY).await;
            }
        });
        Ok(Self { _endpoint: endpoint, connection, welcome, events, subscribing: Mutex::new(()), clock, max_message_bytes: DEFAULT_MAX_MESSAGE_BYTES })
    }

    /// The device, and who this client is to it.
    pub fn welcome(&self) -> &wire::DeviceWelcome {
        &self.welcome
    }

    /// The device's events from now on (status changes, control requests, logs, marks).
    pub fn events(&self) -> broadcast::Receiver<wire::DeviceEvent> {
        self.events.subscribe()
    }

    /// Whether the connection is still open (else: connect again; the device kept running).
    pub fn is_open(&self) -> bool {
        self.connection.close_reason().is_none()
    }

    pub fn close(&self) {
        self.connection.close(quinn::VarInt::from_u32(0), b"bye");
    }

    /// Closes the connection and waits until the device has been told (a short-lived client: the
    /// device sees it leave at once instead of after the idle timeout).
    pub async fn finish(self) {
        self.connection.close(quinn::VarInt::from_u32(0), b"bye");
        let _ = tokio::time::timeout(std::time::Duration::from_secs(2), self._endpoint.wait_idle()).await;
    }

    /// Sends one request on a stream of its own and reads the answer; a refusal is an error
    /// ([`StreamError::Server`] with the device's words, [`Self::request_coded`] keeps the code).
    pub async fn request(&self, kind: wire::live_request::Kind) -> StreamResult<wire::live_response::Kind> {
        self.request_coded(kind).await?.map_err(|e| StreamError::Server(e.message))
    }

    /// As [`Self::request`], with a refusal's code kept.
    pub async fn request_coded(&self, kind: wire::live_request::Kind) -> StreamResult<Result<wire::live_response::Kind, LiveError>> {
        let (mut send, mut recv) = self.connection.open_bi().await?;
        write_message(&mut send, &wire::LiveRequest { kind: Some(kind) }).await?;
        send.finish().map_err(StreamError::protocol)?;
        let response: wire::LiveResponse = read_message(&mut recv, self.max_message_bytes).await?.ok_or_else(|| StreamError::protocol("no answer"))?;
        Ok(match response.kind {
            Some(wire::live_response::Kind::Error(e)) => Err(LiveError::new(wire::ErrorCode::try_from(e.code).unwrap_or(wire::ErrorCode::Unspecified), e.message)),
            Some(kind) => Ok(kind),
            None => return Err(StreamError::protocol("an empty answer")),
        })
    }

    async fn status_of(&self, kind: wire::live_request::Kind) -> StreamResult<wire::DeviceStatus> {
        match self.request(kind).await? {
            wire::live_response::Kind::Status(s) => Ok(s),
            _ => Err(StreamError::protocol("expected the device's status")),
        }
    }

    pub async fn status(&self) -> StreamResult<wire::DeviceStatus> {
        self.status_of(wire::live_request::Kind::Status(wire::GetStatus {})).await
    }

    /// Configures (or with `dry_run` only checks) a chain: a pipeline spec, JSON or TOML.
    pub async fn configure(&self, spec: &str, dry_run: bool) -> StreamResult<wire::ConfigureResult> {
        match self.request(wire::live_request::Kind::Configure(wire::ConfigureChain { spec_json: spec.into(), dry_run })).await? {
            wire::live_response::Kind::Configured(r) => Ok(r),
            _ => Err(StreamError::protocol("expected a configure result")),
        }
    }

    pub async fn start(&self, record: bool, label: &str, notes: &str) -> StreamResult<wire::DeviceStatus> {
        self.status_of(wire::live_request::Kind::Start(wire::Start { record, label: label.into(), notes: notes.into() })).await
    }

    pub async fn stop(&self) -> StreamResult<wire::DeviceStatus> {
        self.status_of(wire::live_request::Kind::Stop(wire::Stop {})).await
    }

    pub async fn request_control(&self) -> StreamResult<wire::DeviceStatus> {
        self.status_of(wire::live_request::Kind::RequestControl(wire::RequestControl {})).await
    }

    pub async fn hand_over(&self, to: &str) -> StreamResult<wire::DeviceStatus> {
        self.status_of(wire::live_request::Kind::HandOver(wire::HandOverControl { to: to.into() })).await
    }

    pub async fn release_control(&self) -> StreamResult<wire::DeviceStatus> {
        self.status_of(wire::live_request::Kind::ReleaseControl(wire::ReleaseControl {})).await
    }

    pub async fn mark(&self, sample: u64, label: &str) -> StreamResult<wire::DeviceStatus> {
        self.status_of(wire::live_request::Kind::Mark(wire::Mark { sample, label: label.into() })).await
    }

    pub async fn set_parameter(&self, node: &str, parameter: &str, value_json: &str) -> StreamResult<wire::DeviceStatus> {
        self.status_of(wire::live_request::Kind::SetParameter(wire::SetParameter { node: node.into(), parameter: parameter.into(), value_json: value_json.into() })).await
    }

    pub async fn list_taps(&self) -> StreamResult<Vec<wire::TapInfo>> {
        match self.request(wire::live_request::Kind::ListTaps(wire::ListTaps {})).await? {
            wire::live_response::Kind::Taps(t) => Ok(t.taps),
            _ => Err(StreamError::protocol("expected the taps")),
        }
    }

    /// A stretch of a tap of the run (from the device's ring or its disk).
    pub async fn read_range(&self, tap: &str, first_sample: u64, count: u32, channels: &[u32]) -> StreamResult<wire::TapFrame> {
        let request = wire::ReadRange { tap: tap.into(), first_sample, count, channels: channels.to_vec() };
        match self.request(wire::live_request::Kind::ReadRange(request)).await? {
            wire::live_response::Kind::Frame(f) => Ok(f),
            _ => Err(StreamError::protocol("expected a frame")),
        }
    }

    /// A view of a stretch of a tap of the run (scrolling back).
    pub async fn view(&self, tap: &str, plot_key: u64, view: crate::protocol::wire::ViewRequest) -> StreamResult<crate::protocol::wire::EnvelopeFrame> {
        match self.request(wire::live_request::Kind::View(wire::TapView { tap: tap.into(), plot_key, view: Some(view) })).await? {
            wire::live_response::Kind::Envelope(e) => Ok(e),
            _ => Err(StreamError::protocol("expected a view")),
        }
    }

    /// The last log entries of the device at `min_level` and above.
    pub async fn log(&self, entries: u32, min_level: wire::LogLevel) -> StreamResult<Vec<wire::LogEntry>> {
        match self.request(wire::live_request::Kind::Log(wire::GetLog { entries, min_level: min_level as i32 })).await? {
            wire::live_response::Kind::Log(l) => Ok(l.entries),
            _ => Err(StreamError::protocol("expected log entries")),
        }
    }

    /// One clock exchange now: the device's clock minus this computer's, and the round trip.
    pub async fn clock(&self) -> StreamResult<ClockSample> {
        exchange(&self.connection, &self.clock, self.max_message_bytes).await
    }

    /// The best clock estimate so far (`None` before the first exchange, a second after connecting).
    pub fn clock_estimate(&self) -> Option<ClockSample> {
        self.clock.lock().unwrap_or_else(|e| e.into_inner()).best()
    }

    /// A one-time pairing code for another computer (director only); `control`: it may direct.
    pub async fn pairing_code(&self, control: bool) -> StreamResult<wire::PairingCode> {
        match self.request(wire::live_request::Kind::PairingCode(wire::NewPairingCode { control })).await? {
            wire::live_response::Kind::PairingCode(c) => Ok(c),
            _ => Err(StreamError::protocol("expected a pairing code")),
        }
    }

    /// Every value of `tap` from the head on (`channels` empty: all).
    pub async fn subscribe(&self, tap: &str, channels: &[u32]) -> StreamResult<TapReceiver> {
        let (info, recv) = self.open_subscription(wire::live_request::Kind::Subscribe(wire::Subscribe { tap: tap.into(), channels: channels.to_vec() })).await?;
        Ok(TapReceiver { recv, info, max_message_bytes: self.max_message_bytes })
    }

    /// A live view of `tap` for plot `plot_key`: `columns_per_second` [min, max] columns.
    pub async fn subscribe_view(&self, request: wire::SubscribeView) -> StreamResult<ViewReceiver> {
        let (info, recv) = self.open_subscription(wire::live_request::Kind::SubscribeView(request)).await?;
        Ok(ViewReceiver { recv, info, max_message_bytes: self.max_message_bytes })
    }

    pub async fn unsubscribe(&self, taps: &[String], plot_keys: &[u64]) -> StreamResult<()> {
        self.request(wire::live_request::Kind::Unsubscribe(wire::Unsubscribe { taps: taps.to_vec(), plot_keys: plot_keys.to_vec() })).await?;
        Ok(())
    }

    async fn open_subscription(&self, kind: wire::live_request::Kind) -> StreamResult<(wire::TapInfo, quinn::RecvStream)> {
        let _one_at_a_time = self.subscribing.lock().await;
        let answer = self.request(kind).await?;
        let wire::live_response::Kind::Taps(list) = answer else { return Err(StreamError::protocol("expected the tap")) };
        let mut recv = self.connection.accept_uni().await?;
        let opened: wire::TapOpened = read_message(&mut recv, self.max_message_bytes).await?.ok_or_else(|| StreamError::protocol("the tap's stream ended at once"))?;
        let info = opened.info.or_else(|| list.taps.into_iter().next()).ok_or_else(|| StreamError::protocol("a tap without its description"))?;
        Ok((info, recv))
    }
}

/// The frames of one tap.
pub struct TapReceiver {
    recv: quinn::RecvStream,
    info: wire::TapInfo,
    max_message_bytes: usize,
}

impl TapReceiver {
    pub fn info(&self) -> &wire::TapInfo {
        &self.info
    }

    /// The next frame; `None` when the device ended the tap (the run stopped, or unsubscribed).
    pub async fn next(&mut self) -> StreamResult<Option<wire::TapFrame>> {
        read_message(&mut self.recv, self.max_message_bytes).await
    }
}

/// The columns of one live view.
pub struct ViewReceiver {
    recv: quinn::RecvStream,
    info: wire::TapInfo,
    max_message_bytes: usize,
}

impl ViewReceiver {
    pub fn info(&self) -> &wire::TapInfo {
        &self.info
    }

    pub async fn next(&mut self) -> StreamResult<Option<wire::LiveViewFrame>> {
        read_message(&mut self.recv, self.max_message_bytes).await
    }
}
