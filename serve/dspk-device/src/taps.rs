//! Taps and their subscribers: every block of every tap goes to each client subscribed to it, as
//! frames (every value) or as the columns of a live view (min / max per column, what a plot draws).
//! A client that falls behind loses frames (its queue is full); the device never waits for it, and
//! the frame counter's gap tells the client.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

use dspk_core::SampleFormat;
use dspk_stream::live::{wire, LiveError, LiveResult, Pushed, Subscription, SUBSCRIPTION_QUEUE};
use dspk_stream::protocol::wire as session;
use tokio::sync::mpsc;

/// Most columns per second a live view may ask for.
const MAX_COLUMNS_PER_SECOND: f64 = 2_000.0;

/// One block of one tap, as the device has it.
pub enum TapData<'a> {
    /// Stored values (a source's raw data): channel-major, `format`, with each channel's gain and
    /// offset for views.
    Stored { bytes: &'a [u8], format: SampleFormat, gains: &'a [f32], offsets: &'a [f32] },
    /// Values in the signal's unit, channel-major.
    Values(&'a [f32]),
    /// Events: (sample, channel).
    Events(&'a [(u64, u32)]),
}

struct ViewState {
    plot_key: u64,
    samples_per_column: f64,
    /// The column being filled and its min / max per subscribed channel.
    column: Option<u64>,
    min: Vec<f32>,
    max: Vec<f32>,
}

struct Subscriber {
    tap: String,
    /// Channel indices of the tap (empty: all).
    channels: Vec<usize>,
    sender: mpsc::Sender<Pushed>,
    sequence: u64,
    view: Option<ViewState>,
}

#[derive(Default)]
pub struct TapHub {
    taps: Mutex<Vec<wire::TapInfo>>,
    subscribers: Mutex<Vec<Subscriber>>,
    dropped: AtomicU64,
}

impl TapHub {
    pub fn taps(&self) -> Vec<wire::TapInfo> {
        self.taps.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }

    pub fn tap(&self, name: &str) -> Option<wire::TapInfo> {
        self.taps.lock().unwrap_or_else(|e| e.into_inner()).iter().find(|t| t.name == name).cloned()
    }

    /// Replaces the taps (a new chain); subscribers of taps that are gone are ended.
    pub fn set_taps(&self, taps: Vec<wire::TapInfo>) {
        self.subscribers.lock().unwrap_or_else(|e| e.into_inner()).retain(|s| taps.iter().any(|t| t.name == s.tap));
        *self.taps.lock().unwrap_or_else(|e| e.into_inner()) = taps;
    }

    /// Frames dropped for slow clients since the device started.
    pub fn dropped(&self) -> u64 {
        self.dropped.load(Ordering::Relaxed)
    }

    fn check_channels(info: &wire::TapInfo, channels: &[u32]) -> LiveResult<Vec<usize>> {
        let total = info.channels.len();
        let picked: Vec<usize> = channels.iter().map(|&c| c as usize).collect();
        if let Some(bad) = picked.iter().find(|&&c| c >= total) {
            return Err(LiveError::invalid(format!("tap {}: channel {bad} does not exist ({total} channels)", info.name)));
        }
        Ok(picked)
    }

    pub fn subscribe(&self, request: &wire::Subscribe) -> LiveResult<Subscription> {
        let info = self.tap(&request.tap).ok_or_else(|| LiveError::not_found(format!("no tap {}", request.tap)))?;
        let channels = Self::check_channels(&info, &request.channels)?;
        let (sender, frames) = mpsc::channel(SUBSCRIPTION_QUEUE);
        self.subscribers.lock().unwrap_or_else(|e| e.into_inner()).push(Subscriber { tap: info.name.clone(), channels, sender, sequence: 0, view: None });
        Ok(Subscription { info, plot_key: 0, frames })
    }

    pub fn subscribe_view(&self, request: &wire::SubscribeView) -> LiveResult<Subscription> {
        let info = self.tap(&request.tap).ok_or_else(|| LiveError::not_found(format!("no tap {}", request.tap)))?;
        if info.kind != wire::TapKind::Samples as i32 {
            return Err(LiveError::invalid(format!("tap {} carries events: subscribe to it, views are of signals", info.name)));
        }
        if request.processing.as_ref().is_some_and(|p| !p.ops.is_empty()) {
            return Err(LiveError::invalid("viewer ops on a live view are planned; put them in the device's chain for now"));
        }
        if !(request.columns_per_second > 0.0 && request.columns_per_second <= MAX_COLUMNS_PER_SECOND) {
            return Err(LiveError::invalid(format!("columns_per_second {}: between 0 and {MAX_COLUMNS_PER_SECOND}", request.columns_per_second)));
        }
        let channels = Self::check_channels(&info, &request.channels)?;
        let rate = info.rate_hz.as_ref().map_or(0.0, |r| r.numerator as f64 / r.denominator.max(1) as f64);
        let samples_per_column = (rate / request.columns_per_second).max(1.0);
        let rows = if channels.is_empty() { info.channels.len() } else { channels.len() };
        let (sender, frames) = mpsc::channel(SUBSCRIPTION_QUEUE);
        let view = ViewState { plot_key: request.plot_key, samples_per_column, column: None, min: vec![f32::INFINITY; rows], max: vec![f32::NEG_INFINITY; rows] };
        let mut subs = self.subscribers.lock().unwrap_or_else(|e| e.into_inner());
        // A plot subscribes again (zoomed): the new one replaces the old
        subs.retain(|s| s.view.as_ref().is_none_or(|v| v.plot_key != request.plot_key));
        subs.push(Subscriber { tap: info.name.clone(), channels, sender, sequence: 0, view: Some(view) });
        Ok(Subscription { info, plot_key: request.plot_key, frames })
    }

    /// Hands one block of `tap` (`channels` rows of `samples`) to its subscribers.
    pub fn publish(&self, tap: &str, channels: usize, first_sample: u64, samples: usize, source_time_ns: u64, lost_before: u64, data: TapData<'_>) {
        let mut subs = self.subscribers.lock().unwrap_or_else(|e| e.into_inner());
        if !subs.iter().any(|s| s.tap == tap) {
            return;
        }
        // Values for views, decoded once
        let mut values: Option<Vec<f32>> = None;
        let now = dspk_stream::live::unix_now_ns();
        subs.retain_mut(|sub| {
            if sub.tap != tap {
                return true;
            }
            let pushed = match &mut sub.view {
                None => {
                    let (bytes, count) = frame_bytes(&data, channels, samples, &sub.channels);
                    Some(Pushed::Frame(wire::TapFrame {
                        tap: 0,
                        sequence: sub.sequence,
                        first_sample,
                        count,
                        source_time_ns,
                        sent_unix_ns: now,
                        lost_before,
                        data: bytes,
                    }))
                }
                Some(view) => {
                    let all = values.get_or_insert_with(|| scaled(&data, channels, samples));
                    view_columns(view, all, channels, samples, first_sample, &sub.channels).map(|(first, covered, columns)| {
                        Pushed::View(wire::LiveViewFrame { tap: 0, sequence: sub.sequence, first_sample: first, samples: covered, sent_unix_ns: now, columns: Some(columns) })
                    })
                }
            };
            let Some(pushed) = pushed else { return true };
            sub.sequence += 1;
            match sub.sender.try_send(pushed) {
                Ok(()) => true,
                Err(mpsc::error::TrySendError::Full(_)) => {
                    self.dropped.fetch_add(1, Ordering::Relaxed);
                    true
                }
                // The client unsubscribed or left
                Err(mpsc::error::TrySendError::Closed(_)) => false,
            }
        });
    }
}

/// The bytes of a frame of `data` for the subscribed channels, and its count.
fn frame_bytes(data: &TapData<'_>, channels: usize, samples: usize, picked: &[usize]) -> (Vec<u8>, u32) {
    let rows = |row_bytes: usize, bytes: &[u8]| -> Vec<u8> {
        if picked.is_empty() {
            return bytes.to_vec();
        }
        picked.iter().flat_map(|&c| bytes[c * row_bytes..(c + 1) * row_bytes].iter().copied()).collect()
    };
    match data {
        TapData::Stored { bytes, format, .. } => (rows(samples * format.bytes(), bytes), samples as u32),
        TapData::Values(values) => {
            let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
            (rows(samples * 4, &bytes), samples as u32)
        }
        TapData::Events(events) => {
            let kept = events.iter().filter(|(_, c)| picked.is_empty() || picked.contains(&(*c as usize)));
            let mut out = Vec::new();
            let mut n = 0u32;
            for (s, c) in kept {
                out.extend_from_slice(&s.to_le_bytes());
                out.extend_from_slice(&c.to_le_bytes());
                n += 1;
            }
            let _ = channels;
            (out, n)
        }
    }
}

/// Every channel's values in its unit, channel-major (events: none).
fn scaled(data: &TapData<'_>, channels: usize, samples: usize) -> Vec<f32> {
    match data {
        TapData::Stored { bytes, format, gains, offsets } => {
            let row = samples * format.bytes();
            let mut out = vec![0.0f32; channels * samples];
            for c in 0..channels {
                format.decode(&bytes[c * row..(c + 1) * row], &mut out[c * samples..(c + 1) * samples], gains[c], offsets[c]);
            }
            out
        }
        TapData::Values(values) => values.to_vec(),
        TapData::Events(_) => Vec::new(),
    }
}

/// Feeds a block into a view; returns the columns it completed (their first sample, the samples
/// they cover, and the [min, max] rows).
fn view_columns(view: &mut ViewState, values: &[f32], channels: usize, samples: usize, first_sample: u64, picked: &[usize]) -> Option<(u64, u64, session::Columns)> {
    if values.len() != channels * samples {
        return None;
    }
    let rows: Vec<usize> = if picked.is_empty() { (0..channels).collect() } else { picked.to_vec() };
    let mut done: Vec<Vec<(f32, f32)>> = Vec::new();
    let mut first_done: Option<u64> = None;
    for t in 0..samples {
        let g = first_sample + t as u64;
        let col = (g as f64 / view.samples_per_column).floor() as u64;
        if view.column != Some(col) {
            if let Some(prev) = view.column {
                first_done.get_or_insert(prev);
                done.push(view.min.iter().zip(&view.max).map(|(&a, &b)| (a, b)).collect());
            }
            view.column = Some(col);
            view.min.iter_mut().for_each(|m| *m = f32::INFINITY);
            view.max.iter_mut().for_each(|m| *m = f32::NEG_INFINITY);
        }
        for (r, &c) in rows.iter().enumerate() {
            let v = values[c * samples + t];
            if v < view.min[r] {
                view.min[r] = v;
            }
            if v > view.max[r] {
                view.max[r] = v;
            }
        }
    }
    let first_col = first_done?;
    let width = done.len();
    let mut min_max = vec![0.0f32; 2 * rows.len() * width];
    for (col, column) in done.iter().enumerate() {
        for (row, &(lo, hi)) in column.iter().enumerate() {
            let at = 2 * (row * width + col);
            min_max[at] = lo;
            min_max[at + 1] = hi;
        }
    }
    let spc = view.samples_per_column;
    let start = (first_col as f64 * spc).ceil() as u64;
    let end = ((first_col + width as u64) as f64 * spc).ceil() as u64;
    Some((start, end - start, session::Columns { min_max, complete: true }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn info(name: &str, channels: usize) -> wire::TapInfo {
        wire::TapInfo {
            name: name.into(),
            kind: wire::TapKind::Samples as i32,
            direction: wire::TapDirection::Out as i32,
            channels: (0..channels).map(|i| session::Channel { name: format!("ch{i}"), gain: 1.0, ..Default::default() }).collect(),
            rate_hz: Some(session::Ratio { numerator: 1000, denominator: 1 }),
            format: session::SampleFormat::F32 as i32,
            ..Default::default()
        }
    }

    #[tokio::test]
    async fn frames_views_and_a_slow_client() {
        let hub = TapHub::default();
        hub.set_taps(vec![info("x", 2)]);
        let mut raw = hub.subscribe(&wire::Subscribe { tap: "x".into(), channels: vec![1] }).unwrap();
        let mut view = hub.subscribe_view(&wire::SubscribeView { tap: "x".into(), plot_key: 7, channels: vec![], columns_per_second: 100.0, processing: None }).unwrap();
        // 10 samples per column; two blocks of 15
        for k in 0..2u64 {
            let values: Vec<f32> = (0..2).flat_map(|c| (0..15).map(move |t| (c * 100 + k as usize * 15 + t) as f32)).collect();
            hub.publish("x", 2, k * 15, 15, 0, 0, TapData::Values(&values));
        }
        let Some(Pushed::Frame(f)) = raw.frames.recv().await else { panic!() };
        assert_eq!((f.count, f.data.len()), (15, 15 * 4));
        assert_eq!(f32::from_le_bytes(f.data[..4].try_into().unwrap()), 100.0, "channel 1 only");
        // First block completes column 0 (samples 0..10), the second column 1 (10..20)
        let Some(Pushed::View(v1)) = view.frames.recv().await else { panic!() };
        assert_eq!((v1.first_sample, v1.samples), (0, 10));
        let c = v1.columns.unwrap().min_max;
        assert_eq!(&c[..2], &[0.0, 9.0]);
        assert_eq!(&c[2..4], &[100.0, 109.0]);
        let Some(Pushed::View(v2)) = view.frames.recv().await else { panic!() };
        assert_eq!((v2.first_sample, v2.sequence), (10, 1));

        // A client that never reads: frames beyond its queue are dropped and counted
        let _slow = hub.subscribe(&wire::Subscribe { tap: "x".into(), channels: vec![] }).unwrap();
        let values = vec![0.0f32; 2];
        for k in 0..(SUBSCRIPTION_QUEUE as u64 + 10) {
            hub.publish("x", 2, 30 + k, 1, 0, 0, TapData::Values(&values));
        }
        assert!(hub.dropped() >= 10);
    }
}
