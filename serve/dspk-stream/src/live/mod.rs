//! Devices: live sources, one chain, taps, one director and many viewers (`proto/dspk_live.proto`).
//!
//! The protocol does not depend on its transport:
//! - [`Device`] is what a device does (`dspk-device` implements it); every method takes the
//!   [`Caller`] so permission and director checks live in one place, the device.
//! - [`dispatch`] turns a [`wire::LiveRequest`] into a [`wire::LiveResponse`] (and a
//!   [`Subscription`] to pump), for any transport.
//! - [`quic`] carries it over QUIC + TLS (the transport dspk ships). Another one (gRPC + ZeroMQ, as
//!   Synapse) is a module of its own on the same [`Device`] and [`dispatch`].
//! - [`Director`]: one director, many viewers, the role kept for a disconnected director for a
//!   while. [`Lifecycle`]: the state machine. [`ClockEstimate`]: the offset between two clocks.

mod clock;
mod director;
pub mod quic;
mod state;

pub use clock::{ClockEstimate, ClockSample};
pub use director::{ControlOutcome, Director, DIRECTOR_GRACE};
pub use state::Lifecycle;

/// Protocol version sent in `DeviceHello` and `DeviceWelcome`.
pub const PROTOCOL_VERSION: u32 = 1;

/// TLS application protocol (ALPN) of a device connection over QUIC.
pub const ALPN: &[u8] = b"dspk-live/1";

/// Frames queued per subscription before the oldest are dropped for that client (a slow client
/// never slows the device; the `sequence` gap tells it).
pub const SUBSCRIPTION_QUEUE: usize = 256;

/// Messages generated from `proto/dspk_live.proto` (package `dspk_live.v1`); the types it shares
/// with sessions (`Channel`, `Ratio`, `SampleFormat`, views) are [`crate::protocol::wire`]'s.
pub mod wire {
    include!(concat!(env!("OUT_DIR"), "/dspk_live.v1.rs"));
}

/// Who sent a request: the client's name (from its certificate) and whether it may direct.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Caller {
    pub name: String,
    /// The control permission (from pairing): may become the director.
    pub may_control: bool,
}

/// A refused or failed request: a code to act on, words to show.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LiveError {
    pub code: wire::ErrorCode,
    pub message: String,
}

impl LiveError {
    pub fn new(code: wire::ErrorCode, message: impl Into<String>) -> Self {
        Self { code, message: message.into() }
    }
    pub fn not_director(director: &str) -> Self {
        let who = if director.is_empty() { "nobody: request control first".to_string() } else { format!("{director}; request control") };
        Self::new(wire::ErrorCode::NotDirector, format!("only the director may do this; the director is {who}"))
    }
    pub fn wrong_state(message: impl Into<String>) -> Self {
        Self::new(wire::ErrorCode::WrongState, message)
    }
    pub fn invalid(message: impl Into<String>) -> Self {
        Self::new(wire::ErrorCode::Invalid, message)
    }
    pub fn not_found(message: impl Into<String>) -> Self {
        Self::new(wire::ErrorCode::NotFound, message)
    }
    pub fn internal(message: impl Into<String>) -> Self {
        Self::new(wire::ErrorCode::Internal, message)
    }
    fn to_wire(&self) -> wire::Error {
        wire::Error { code: self.code as i32, message: self.message.clone() }
    }
}

impl std::fmt::Display for LiveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for LiveError {}

pub type LiveResult<T> = Result<T, LiveError>;

/// A frame pushed on a subscription.
#[derive(Debug, Clone, PartialEq)]
pub enum Pushed {
    Frame(wire::TapFrame),
    View(wire::LiveViewFrame),
}

/// An accepted subscription: what it is, and the frames to send. The device keeps the sending
/// side and stops when the receiver is dropped (the client unsubscribed or left).
pub struct Subscription {
    pub info: wire::TapInfo,
    /// The plot it is for (`SubscribeView`; 0 for `Subscribe`).
    pub plot_key: u64,
    pub frames: tokio::sync::mpsc::Receiver<Pushed>,
}

/// What a device does. Every request method checks the caller itself (permission, director,
/// state) and returns a [`LiveError`] with the reason when it refuses. Methods may block for a
/// moment (a configure compiles kernels); transports call them off their network threads.
pub trait Device: Send + Sync + 'static {
    fn info(&self) -> wire::DeviceInfo;
    fn status(&self) -> wire::DeviceStatus;
    /// Events for one connected client (status changes, control requests, logs, marks).
    fn events(&self) -> tokio::sync::broadcast::Receiver<wire::DeviceEvent>;
    /// A client connected (a disconnected director takes its role back) / left (its role is kept
    /// for it for [`DIRECTOR_GRACE`]).
    fn connected(&self, caller: &Caller);
    fn disconnected(&self, caller: &Caller);

    fn configure(&self, caller: &Caller, request: wire::ConfigureChain) -> LiveResult<wire::ConfigureResult>;
    fn start(&self, caller: &Caller, request: wire::Start) -> LiveResult<wire::DeviceStatus>;
    fn stop(&self, caller: &Caller) -> LiveResult<wire::DeviceStatus>;
    fn set_parameter(&self, caller: &Caller, request: wire::SetParameter) -> LiveResult<wire::DeviceStatus>;
    fn request_control(&self, caller: &Caller) -> LiveResult<wire::DeviceStatus>;
    fn hand_over(&self, caller: &Caller, to: &str) -> LiveResult<wire::DeviceStatus>;
    fn release_control(&self, caller: &Caller) -> LiveResult<wire::DeviceStatus>;
    fn mark(&self, caller: &Caller, mark: wire::Mark) -> LiveResult<wire::DeviceStatus>;

    fn list_taps(&self) -> Vec<wire::TapInfo>;
    fn subscribe(&self, caller: &Caller, request: wire::Subscribe) -> LiveResult<Subscription>;
    fn subscribe_view(&self, caller: &Caller, request: wire::SubscribeView) -> LiveResult<Subscription>;
    fn read_range(&self, caller: &Caller, request: wire::ReadRange) -> LiveResult<wire::TapFrame>;
    fn view(&self, caller: &Caller, request: wire::TapView) -> LiveResult<crate::protocol::wire::EnvelopeFrame>;
    fn log(&self, request: wire::GetLog) -> wire::LogEntries;
}

/// The device's clock (Unix ns).
pub fn unix_now_ns() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(0)
}

/// What a request asks a transport to do besides answering.
pub enum Followup {
    None,
    /// Pump this subscription to the client.
    Subscribe(Subscription),
    /// Stop these subscriptions (by tap name, by plot key).
    Unsubscribe { taps: Vec<String>, plot_keys: Vec<u64> },
}

/// Answers one request, for any transport.
pub fn dispatch(device: &dyn Device, caller: &Caller, request: wire::LiveRequest) -> (wire::LiveResponse, Followup) {
    use wire::live_request::Kind as Rq;
    use wire::live_response::Kind as Rs;
    let status = |r: LiveResult<wire::DeviceStatus>| r.map(Rs::Status);
    let mut followup = Followup::None;
    let answer: LiveResult<Rs> = match request.kind {
        None => Err(LiveError::invalid("an empty request")),
        Some(Rq::Status(_)) => Ok(Rs::Status(device.status())),
        Some(Rq::Configure(c)) => device.configure(caller, c).map(Rs::Configured),
        Some(Rq::Start(s)) => status(device.start(caller, s)),
        Some(Rq::Stop(_)) => status(device.stop(caller)),
        Some(Rq::SetParameter(p)) => status(device.set_parameter(caller, p)),
        Some(Rq::RequestControl(_)) => status(device.request_control(caller)),
        Some(Rq::HandOver(h)) => status(device.hand_over(caller, &h.to)),
        Some(Rq::ReleaseControl(_)) => status(device.release_control(caller)),
        Some(Rq::Mark(m)) => status(device.mark(caller, m)),
        Some(Rq::ListTaps(_)) => Ok(Rs::Taps(wire::TapList { taps: device.list_taps() })),
        Some(Rq::Subscribe(s)) => device.subscribe(caller, s).map(|sub| {
            let info = sub.info.clone();
            followup = Followup::Subscribe(sub);
            Rs::Taps(wire::TapList { taps: vec![info] })
        }),
        Some(Rq::SubscribeView(s)) => device.subscribe_view(caller, s).map(|sub| {
            let info = sub.info.clone();
            followup = Followup::Subscribe(sub);
            Rs::Taps(wire::TapList { taps: vec![info] })
        }),
        Some(Rq::Unsubscribe(u)) => {
            followup = Followup::Unsubscribe { taps: u.taps, plot_keys: u.plot_keys };
            Ok(Rs::Status(device.status()))
        }
        Some(Rq::ReadRange(r)) => device.read_range(caller, r).map(Rs::Frame),
        Some(Rq::View(v)) => device.view(caller, v).map(Rs::Envelope),
        Some(Rq::Clock(ping)) => {
            let received = unix_now_ns();
            Ok(Rs::Clock(wire::ClockPong { client_sent_ns: ping.client_sent_ns, device_received_unix_ns: received, device_sent_unix_ns: unix_now_ns() }))
        }
        Some(Rq::Log(l)) => Ok(Rs::Log(device.log(l))),
    };
    let kind = match answer {
        Ok(kind) => kind,
        Err(e) => Rs::Error(e.to_wire()),
    };
    (wire::LiveResponse { kind: Some(kind) }, followup)
}
