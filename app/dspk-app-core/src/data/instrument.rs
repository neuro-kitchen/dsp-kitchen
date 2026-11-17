//! The app's side of a device (`dspk-device`): connect with a profile (as a server's: an address,
//! its certificate name, this app's identity from `dspk pair`), follow its status, direct it, and
//! receive live views for plots. Desktop only for now (the page has no QUIC).
//!
//! The screens that use it (a "Connect to a device" list, Explore following the head of a run) are
//! the next step; this is the data they draw from. Named `instrument` while `data::device` is the
//! compute device (to be settled once this works).

#![cfg(not(target_family = "wasm"))]

use std::collections::VecDeque;
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};

use anyhow::Context as _;
use dspk_stream::live::quic::DeviceClient;
use dspk_stream::live::wire;
use dspk_stream::transport::live_client_config;
use dspk_stream::{ClientIdentity, ServerTrust};

use crate::data::remote::ServerProfile;
use crate::platform::lock;
use crate::platform::remote::runtime;

/// Columns a live plot keeps (its newest).
pub const LIVE_COLUMNS_KEPT: usize = 8_192;

/// Called when something new arrived (a status, columns): the UI asks for a redraw.
pub type OnChange = Arc<dyn Fn() + Send + Sync>;

/// A live view's newest columns: one entry per column, a `(min, max)` per channel.
#[derive(Default)]
pub struct LiveColumns {
    /// Sample index of the first kept column, and samples per column.
    pub first_sample: u64,
    pub samples_per_column: f64,
    pub columns: VecDeque<Vec<(f32, f32)>>,
    /// Frames the device dropped for this plot (a slow connection).
    pub dropped: u64,
    last_sequence: Option<u64>,
}

impl LiveColumns {
    /// One past the newest sample shown.
    pub fn head_sample(&self) -> u64 {
        self.first_sample + (self.columns.len() as f64 * self.samples_per_column).round() as u64
    }

    fn push(&mut self, frame: wire::LiveViewFrame, channels: usize) {
        if let Some(prev) = self.last_sequence
            && frame.sequence > prev + 1
        {
            self.dropped += frame.sequence - prev - 1;
        }
        self.last_sequence = Some(frame.sequence);
        let Some(columns) = frame.columns else { return };
        let width = columns.min_max.len() / (2 * channels.max(1));
        if width == 0 {
            return;
        }
        if self.columns.is_empty() {
            self.first_sample = frame.first_sample;
        }
        self.samples_per_column = frame.samples as f64 / width as f64;
        for col in 0..width {
            self.columns.push_back((0..channels).map(|row| (columns.min_max[2 * (row * width + col)], columns.min_max[2 * (row * width + col) + 1])).collect());
        }
        while self.columns.len() > LIVE_COLUMNS_KEPT {
            self.columns.pop_front();
            self.first_sample += self.samples_per_column.round() as u64;
        }
    }
}

/// A connected device. A lost connection is made again when next used (the device runs on; views
/// read what was missed from its ring or its recording, so a plot has no hole).
pub struct Instrument {
    pub profile: ServerProfile,
    client: Mutex<Arc<DeviceClient>>,
    status: Arc<Mutex<wire::DeviceStatus>>,
    on_change: OnChange,
    /// The last attempt to connect again (one a second at most).
    retried: Mutex<Option<std::time::Instant>>,
}

/// Connects to `profile`'s device and follows its events into `status`.
fn connect_client(profile: &ServerProfile, status: Option<&Arc<Mutex<wire::DeviceStatus>>>, on_change: &OnChange) -> anyhow::Result<(Arc<DeviceClient>, Arc<Mutex<wire::DeviceStatus>>)> {
    let identity = ClientIdentity::from_file(&profile.identity).with_context(|| format!("reading {}", profile.identity.display()))?;
    let config = live_client_config(&ServerTrust::from_identity(&identity), Some(&identity))?;
    let addr: SocketAddr = profile.address.parse()?;
    let client = runtime().block_on(DeviceClient::connect(addr, &profile.server_name, config)).with_context(|| format!("connecting to device {} ({})", profile.name, profile.address))?;
    let client = Arc::new(client);
    let first = client.welcome().device.as_ref().and_then(|d| d.status.clone()).unwrap_or_default();
    let status = match status {
        Some(s) => {
            *lock(s) = first;
            s.clone()
        }
        None => Arc::new(Mutex::new(first)),
    };
    let mut events = client.events();
    let (s, notify) = (status.clone(), on_change.clone());
    runtime().spawn(async move {
        while let Ok(event) = events.recv().await {
            if let Some(wire::device_event::Kind::Status(new)) = event.kind {
                *lock(&s) = new;
            }
            notify();
        }
    });
    Ok((client, status))
}

impl Instrument {
    /// Connects (blocking: from a background thread) and follows the device's events.
    pub fn connect(profile: ServerProfile, on_change: OnChange) -> anyhow::Result<Self> {
        profile.check().map_err(anyhow::Error::msg)?;
        let (client, status) = connect_client(&profile, None, &on_change)?;
        Ok(Self { profile, client: Mutex::new(client), status, on_change, retried: Mutex::new(None) })
    }

    /// The device and who this app is to it (as it said when this connection was made).
    pub fn welcome(&self) -> wire::DeviceWelcome {
        lock(&self.client).welcome().clone()
    }

    /// Whether the connection is up now (no attempt to make it again: for drawing).
    pub fn connected(&self) -> bool {
        lock(&self.client).is_open()
    }

    /// The connection, made again when it was lost (blocking: from a background thread; one try a
    /// second at most; the old one is returned while the device cannot be reached).
    pub fn client(&self) -> Arc<DeviceClient> {
        let current = lock(&self.client).clone();
        if current.is_open() {
            return current;
        }
        {
            let mut retried = lock(&self.retried);
            if retried.is_some_and(|t| t.elapsed() < std::time::Duration::from_secs(1)) {
                return current;
            }
            *retried = Some(std::time::Instant::now());
        }
        match connect_client(&self.profile, Some(&self.status), &self.on_change) {
            Ok((client, _)) => {
                *lock(&self.client) = client.clone();
                (self.on_change)();
                client
            }
            Err(e) => {
                tracing::warn!("device {}: {e:#}", self.profile.name);
                current
            }
        }
    }

    /// The device's last known status.
    pub fn status(&self) -> wire::DeviceStatus {
        lock(&self.status).clone()
    }

    pub fn is_director(&self) -> bool {
        lock(&self.status).director == lock(&self.client).welcome().client_name
    }

    /// Runs a request on the network runtime (blocking: from a background thread).
    pub fn block_on<T>(&self, f: impl std::future::Future<Output = T>) -> T {
        runtime().block_on(f)
    }

    /// Takes control (asks the director when there is one).
    pub fn request_control(&self) -> anyhow::Result<()> {
        let s = self.block_on(self.client().request_control())?;
        *lock(&self.status) = s;
        Ok(())
    }

    /// Checks a chain (a pipeline spec) without applying it: the refusals, for greying out ops.
    pub fn check_chain(&self, spec_json: &str) -> anyhow::Result<Vec<wire::Refusal>> {
        Ok(self.block_on(self.client().configure(spec_json, true))?.refusals)
    }

    /// Configures (or with `dry_run` checks) a chain: the refusals, or the chain as it will run
    /// (its block length and delay).
    pub fn configure_full(&self, spec_json: &str, dry_run: bool) -> anyhow::Result<wire::ConfigureResult> {
        Ok(self.block_on(self.client().configure(spec_json, dry_run))?)
    }

    pub fn configure(&self, spec_json: &str) -> anyhow::Result<Vec<wire::Refusal>> {
        Ok(self.block_on(self.client().configure(spec_json, false))?.refusals)
    }

    pub fn start(&self, record: bool, label: &str) -> anyhow::Result<()> {
        *lock(&self.status) = self.block_on(self.client().start(record, label, ""))?;
        Ok(())
    }

    pub fn stop(&self) -> anyhow::Result<()> {
        *lock(&self.status) = self.block_on(self.client().stop())?;
        Ok(())
    }

    pub fn mark(&self, label: &str) -> anyhow::Result<()> {
        self.block_on(self.client().mark(0, label))?;
        Ok(())
    }

    /// A live view of `tap` for plot `plot_key` at `columns_per_second`: its newest columns,
    /// filled in the background (the UI is told through `on_change`).
    pub fn live_view(&self, tap: &str, plot_key: u64, channels: &[u32], columns_per_second: f64) -> anyhow::Result<Arc<Mutex<LiveColumns>>> {
        let request = wire::SubscribeView { tap: tap.into(), plot_key, channels: channels.to_vec(), columns_per_second, processing: None };
        let mut rx = self.block_on(self.client().subscribe_view(request))?;
        let rows = if channels.is_empty() { rx.info().channels.len() } else { channels.len() };
        let columns = Arc::new(Mutex::new(LiveColumns::default()));
        let (c, notify) = (columns.clone(), self.on_change.clone());
        runtime().spawn(async move {
            while let Ok(Some(frame)) = rx.next().await {
                lock(&c).push(frame, rows);
                notify();
            }
        });
        Ok(columns)
    }
}

/// A tap of a device's run in progress read as a recording (Explore's live mode): views are asked
/// of the device (from its ring, its recording, or its overview when zoomed out), samples read
/// there. `info().samples` is how far the run was when this snapshot was made; [`Self::grown`]
/// makes a longer one as the run goes on.
pub struct LiveTapSignal {
    instrument: Arc<Instrument>,
    tap: String,
    info: dspk_core::RecordingInfo,
}

impl LiveTapSignal {
    /// Tap `tap` of the device as far as its run has gone (blocking: from a background thread).
    pub fn open(instrument: Arc<Instrument>, tap: &str) -> anyhow::Result<Self> {
        let client = instrument.client();
        let taps = instrument.block_on(client.list_taps())?;
        let t = taps.into_iter().find(|t| t.name == tap).with_context(|| format!("the device has no tap {tap}"))?;
        let head = instrument.status().run.map_or(0, |r| r.head_sample);
        let header = dspk_stream::protocol::wire::Header {
            protocol_version: dspk_stream::protocol::PROTOCOL_VERSION,
            name: format!("{} · {tap}", instrument.profile.name),
            channels: t.channels.clone(),
            samples: head,
            sample_rate_hz: t.rate_hz,
            format: t.format,
            ..Default::default()
        };
        let info = dspk_stream::protocol::info_from_header(&header)?;
        Ok(Self { instrument, tap: tap.to_string(), info })
    }

    /// This tap as far as the run has gone now; `None` when it has not moved.
    pub fn grown(&self) -> Option<Self> {
        let head = self.instrument.status().run.map_or(0, |r| r.head_sample);
        (head > self.info.samples).then(|| {
            let mut info = self.info.clone();
            info.samples = head;
            Self { instrument: self.instrument.clone(), tap: self.tap.clone(), info }
        })
    }
}

impl dspk_view::SignalBackend for LiveTapSignal {
    fn info(&self) -> &dspk_core::RecordingInfo {
        &self.info
    }

    fn view(&self, view: &dspk_view::View, out: &mut dspk_view::Envelope) -> dspk_core::DspResult<()> {
        let err = |e: dspk_stream::StreamError| dspk_core::DspError::Io(e.to_string());
        let request = dspk_stream::protocol::request_from_view(view).map_err(err)?;
        // Plot key 0: views of one plot are not superseded by each other (traces and heatmap
        // ask together)
        let frame = self.instrument.block_on(self.instrument.client().view(&self.tap, 0, request)).map_err(err)?;
        *out = dspk_stream::protocol::envelope_from_frame(frame).map_err(err)?;
        Ok(())
    }

    fn read(&self, channels: &[usize], samples: std::ops::Range<u64>, out: &mut [f32]) -> dspk_core::DspResult<()> {
        let n = (samples.end - samples.start) as usize;
        let chans: Vec<u32> = channels.iter().map(|&c| c as u32).collect();
        let frame = self
            .instrument
            .block_on(self.instrument.client().read_range(&self.tap, samples.start, n as u32, &chans))
            .map_err(|e| dspk_core::DspError::Io(e.to_string()))?;
        let size = self.info.format.bytes();
        for (row, &ch) in channels.iter().enumerate() {
            let c = &self.info.channels[ch];
            let bytes = frame.data.get(row * n * size..(row + 1) * n * size).ok_or_else(|| dspk_core::DspError::Io("a short read from the device".into()))?;
            self.info.format.decode(bytes, &mut out[row * n..(row + 1) * n], c.gain, c.offset);
        }
        Ok(())
    }

    fn focus(&self, _sample: u64) {}

    fn prepare(&self, _start: u64, _end: u64) -> dspk_core::DspResult<()> {
        Ok(())
    }

    fn progress(&self) -> dspk_view::Progress {
        let n = self.info.samples;
        dspk_view::Progress { filled: 0..n, done: n, total: n }
    }

    fn watch(&self, _on_progress: dspk_view::OnProgress) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use dspk_stream::protocol::wire::Columns;

    #[test]
    fn columns_accumulate_and_count_drops() {
        let mut c = LiveColumns::default();
        let frame = |sequence, first_sample, values: Vec<f32>| wire::LiveViewFrame {
            tap: 0,
            sequence,
            first_sample,
            samples: 20,
            sent_unix_ns: 0,
            columns: Some(Columns { min_max: values, complete: true }),
        };
        // 2 channels × 2 columns of 10 samples
        c.push(frame(0, 0, vec![0.0, 1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 13.0]), 2);
        c.push(frame(2, 40, vec![4.0, 5.0, 6.0, 7.0, 14.0, 15.0, 16.0, 17.0]), 2);
        assert_eq!(c.columns.len(), 4);
        assert_eq!(c.columns[1], vec![(2.0, 3.0), (12.0, 13.0)]);
        assert_eq!(c.dropped, 1);
        assert_eq!(c.samples_per_column, 10.0);
    }
}
