//! Onsets: when each channel rises above a level (or falls below it) and when it comes back, as
//! events — e.g. muscle contractions on an EMG envelope (band-pass → rectify → low-pass →
//! onsets).
//!
//! - An event **starts** where the signal crosses `level` and **ends** where it crosses back
//!   past `release` (hysteresis: a release between the baseline and the level keeps noise around
//!   the level from splitting one event into many; by default `release = level`).
//! - Events shorter than `min_duration` are dropped; after an event, onsets within `refractory`
//!   of its end are ignored (the signal must then cross the level again).
//! - Each event reports its peak (the largest value, or the lowest when falling) and where it is.
//!
//! The work is split as `find_peak_candidates` is: kernels find the crossings of every channel and
//! compact them on the device, and the extremum of each stretch between crossings; only those are
//! downloaded. [`OnsetTracker`] then pairs them into events on the host, window after window, so a
//! long recording streams through in chunks and events may span chunks: the result does not depend
//! on the chunk size.
//!
//! - [`detect_onsets`] (host): a whole `[channels, samples]` buffer in memory.
//! - [`detect_onsets_on_device`] (device): a whole device buffer.
//! - [`OnsetTracker::push_device_window`]: one window of a longer run (what pipelines use).

mod device;
pub mod kernels;

use crate::error::{OpsError, OpsResult};

pub use device::detect_onsets_on_device;

fn invalid(msg: String) -> OpsError {
    OpsError::Core(dspk_core::DspError::InvalidConfig(msg))
}

/// Which way the signal crosses the level at an onset.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Direction {
    /// Up through `level` (a level above the baseline; envelopes, power).
    #[default]
    Rising,
    /// Down through `level` (a level below the baseline).
    Falling,
}

impl Direction {
    /// `+1` rising, `−1` falling: values are compared as `sign · x`.
    pub fn sign(self) -> f32 {
        match self {
            Direction::Rising => 1.0,
            Direction::Falling => -1.0,
        }
    }
}

/// Options of [`detect_onsets`] and [`OnsetTracker`].
#[derive(Debug, Clone, PartialEq)]
pub struct OnsetOptions {
    /// Level an onset crosses, in the signal's unit: one value for every channel, or one per
    /// channel.
    pub levels: Vec<f32>,
    /// Level the signal crosses back past to end the event (one, or one per channel); empty:
    /// `levels`. Rising: at most the level; falling: at least the level.
    pub releases: Vec<f32>,
    pub direction: Direction,
    /// Events shorter than this many samples are dropped (default 0: none).
    pub min_duration_samples: u64,
    /// Onsets within this many samples after an event's end are ignored (default 0).
    pub refractory_samples: u64,
}

impl OnsetOptions {
    /// One level for every channel, released at the same level, rising.
    pub fn new(level: f32) -> Self {
        Self { levels: vec![level], releases: Vec::new(), direction: Direction::Rising, min_duration_samples: 0, refractory_samples: 0 }
    }

    /// Per-channel levels and releases, in signed space (`sign · value`), checked.
    fn resolve(&self, channels: usize) -> OpsResult<(Vec<f32>, Vec<f32>)> {
        let spread = |v: &[f32], what: &str| -> OpsResult<Vec<f32>> {
            match v.len() {
                1 => Ok(vec![v[0]; channels]),
                n if n == channels => Ok(v.to_vec()),
                n => Err(invalid(format!("onsets: {n} {what} for {channels} channels (give one, or one per channel)"))),
            }
        };
        let levels = spread(&self.levels, "levels")?;
        let releases = if self.releases.is_empty() { levels.clone() } else { spread(&self.releases, "releases")? };
        let s = self.direction.sign();
        for (ch, (&l, &r)) in levels.iter().zip(&releases).enumerate() {
            if !l.is_finite() || !r.is_finite() {
                return Err(invalid(format!("onsets: channel {ch}: level and release must be finite")));
            }
            if s * r > s * l {
                let side = if s > 0.0 { "at most" } else { "at least" };
                return Err(invalid(format!("onsets: channel {ch}: the release ({r}) must be {side} the level ({l})")));
            }
        }
        Ok((levels.iter().map(|v| s * v).collect(), releases.iter().map(|v| s * v).collect()))
    }
}

/// One event of one channel, in samples of the whole run.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OnsetEvent {
    pub channel: usize,
    /// First sample at or past the level.
    pub onset: u64,
    /// First sample back past the release; `None` when the run ended during the event.
    pub offset: Option<u64>,
    /// Where the peak is (the first sample of the largest value; the lowest when falling).
    pub peak_sample: u64,
    pub peak_value: f32,
}

/// Where one channel is between windows.
#[derive(Debug, Clone, Copy)]
struct ChannelState {
    on: bool,
    onset: u64,
    /// Signed peak so far and where.
    peak: f32,
    peak_at: u64,
    /// No onset before this sample (refractory).
    ready_at: u64,
}

/// Pairs crossings into events, window after window (see the module docs).
#[derive(Debug, Clone)]
pub struct OnsetTracker {
    sign: f32,
    /// Signed levels and releases, per channel.
    levels: Vec<f32>,
    releases: Vec<f32>,
    min_duration: u64,
    refractory: u64,
    state: Vec<ChannelState>,
    /// Last value of each channel in the previous window (NaN before the first).
    prev: Vec<f32>,
    events: Vec<OnsetEvent>,
}

/// Crossings of one channel in one window: local sample and kind
/// ([`kernels::CROSSING_UP`] / [`kernels::CROSSING_DOWN`]).
type Crossings<'a> = &'a [(u32, u32)];
/// Signed extremum and its local sample of each stretch: before the first crossing, then after
/// each crossing (one more than the crossings).
type Stretches<'a> = &'a [(f32, u32)];

impl OnsetTracker {
    pub fn new(channels: usize, options: &OnsetOptions) -> OpsResult<Self> {
        let (levels, releases) = options.resolve(channels)?;
        let idle = ChannelState { on: false, onset: 0, peak: f32::NEG_INFINITY, peak_at: 0, ready_at: 0 };
        Ok(Self {
            sign: options.direction.sign(),
            levels,
            releases,
            min_duration: options.min_duration_samples,
            refractory: options.refractory_samples,
            state: vec![idle; channels],
            prev: vec![f32::NAN; channels],
            events: Vec::new(),
        })
    }

    pub fn channels(&self) -> usize {
        self.state.len()
    }

    /// Takes one channel's crossings of a window whose local sample 0 is sample `origin` of the run.
    fn take(&mut self, ch: usize, origin: u64, crossings: Crossings, stretches: Stretches) {
        debug_assert_eq!(stretches.len(), crossings.len() + 1);
        let st = &mut self.state[ch];
        let grow = |st: &mut ChannelState, (v, t): (f32, u32)| {
            if v > st.peak {
                (st.peak, st.peak_at) = (v, origin + t as u64);
            }
        };
        if st.on {
            grow(st, stretches[0]);
        }
        for (k, &(t, kind)) in crossings.iter().enumerate() {
            let at = origin + t as u64;
            let after = stretches[k + 1];
            if kind == kernels::CROSSING_UP {
                if st.on {
                    grow(st, after);
                } else if at >= st.ready_at {
                    (st.on, st.onset, st.peak, st.peak_at) = (true, at, f32::NEG_INFINITY, at);
                    grow(st, after);
                }
            } else if st.on {
                st.on = false;
                if at - st.onset >= self.min_duration {
                    self.events.push(OnsetEvent { channel: ch, onset: st.onset, offset: Some(at), peak_sample: st.peak_at, peak_value: self.sign * st.peak });
                    st.ready_at = at + self.refractory;
                }
            }
        }
    }

    /// A window held on the host: `rows` is `[channels, samples]`, local sample 0 is sample
    /// `origin` of the run, and `scan` the local samples this window owns (later windows continue
    /// from `scan.end`).
    pub fn push_host_window(&mut self, rows: &[f32], samples: usize, scan: std::ops::Range<usize>, origin: u64) {
        let mut crossings = Vec::new();
        let mut stretches = Vec::new();
        for ch in 0..self.channels() {
            let row = &rows[ch * samples..(ch + 1) * samples];
            let (l, r, s) = (self.levels[ch], self.releases[ch], self.sign);
            crossings.clear();
            for t in scan.clone() {
                let p = if t > 0 { row[t - 1] } else { self.prev[ch] };
                let (sp, sv) = (s * p, s * row[t]);
                if sp < l && sv >= l {
                    crossings.push((t as u32, kernels::CROSSING_UP));
                } else if sp >= r && sv < r {
                    crossings.push((t as u32, kernels::CROSSING_DOWN));
                }
            }
            stretches.clear();
            let bounds: Vec<u32> = std::iter::once(scan.start as u32).chain(crossings.iter().map(|c| c.0)).chain(std::iter::once(scan.end as u32)).collect();
            for w in bounds.windows(2) {
                let mut best = (-3.0e38f32, w[0]);
                for t in w[0]..w[1] {
                    let v = s * row[t as usize];
                    if v > best.0 {
                        best = (v, t);
                    }
                }
                stretches.push(best);
            }
            self.take(ch, origin, &crossings, &stretches);
            if scan.end > 0 {
                self.prev[ch] = row[scan.end - 1];
            }
        }
    }

    /// Ends the run: events still open get no offset. Returns every event, by onset then channel.
    pub fn finish(mut self) -> Vec<OnsetEvent> {
        for (ch, st) in self.state.iter().enumerate() {
            if st.on {
                self.events.push(OnsetEvent { channel: ch, onset: st.onset, offset: None, peak_sample: st.peak_at, peak_value: self.sign * st.peak });
            }
        }
        self.events.sort_by_key(|e| (e.onset, e.channel));
        self.events
    }
}

/// Onsets of every channel of a `[channels, samples]` buffer in memory.
pub fn detect_onsets(rows: &[f32], channels: usize, samples: usize, options: &OnsetOptions) -> OpsResult<Vec<OnsetEvent>> {
    if rows.len() != channels * samples {
        return Err(invalid(format!("onsets: {} values for {channels} × {samples}", rows.len())));
    }
    let mut tracker = OnsetTracker::new(channels, options)?;
    tracker.push_host_window(rows, samples, 0..samples, 0);
    Ok(tracker.finish())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two bumps on channel 0 (the second dips to 4 in the middle), one on channel 1.
    fn signal() -> (Vec<f32>, usize) {
        let n = 40;
        let mut x = vec![0.0f32; 2 * n];
        for (t, v) in [(5, 6.0), (6, 9.0), (7, 7.0), (8, 1.0), (20, 8.0), (21, 4.0), (22, 10.0), (23, 6.0)] {
            x[t] = v;
        }
        for t in 30..35 {
            x[n + t] = 3.0 + t as f32 - 30.0;
        }
        (x, n)
    }

    #[test]
    fn events_with_hysteresis_and_peaks() {
        let (x, n) = signal();
        let mut o = OnsetOptions::new(5.0);
        o.releases = vec![2.0];
        let ev = detect_onsets(&x, 2, n, &o).unwrap();
        let got: Vec<_> = ev.iter().map(|e| (e.channel, e.onset, e.offset, e.peak_sample, e.peak_value)).collect();
        // The dip to 4 stays above the release: one event 20..24, peak 10 at 22
        assert_eq!(got, vec![(0, 5, Some(8), 6, 9.0), (0, 20, Some(24), 22, 10.0), (1, 32, Some(35), 34, 7.0)]);
        // Without hysteresis the dip splits the second event
        let ev = detect_onsets(&x, 2, n, &OnsetOptions::new(5.0)).unwrap();
        assert_eq!(ev.iter().filter(|e| e.channel == 0).count(), 3);
    }

    #[test]
    fn min_duration_refractory_and_open_events() {
        let (x, n) = signal();
        let mut o = OnsetOptions::new(5.0);
        o.min_duration_samples = 4;
        let ev = detect_onsets(&x, 2, n, &o).unwrap();
        assert!(ev.iter().all(|e| e.offset.unwrap() - e.onset >= 4), "{ev:?}");
        let mut o = OnsetOptions::new(5.0);
        o.releases = vec![2.0];
        o.refractory_samples = 20;
        let ev = detect_onsets(&x[..n], 1, n, &o).unwrap();
        assert_eq!(ev.len(), 1, "the second bump starts within 20 samples of the first's end");
        // A run ending inside an event
        let ev = detect_onsets(&x[..21], 1, 21, &OnsetOptions::new(5.0)).unwrap();
        assert_eq!(ev.last().map(|e| (e.onset, e.offset)), Some((20, None)));
    }

    #[test]
    fn falling_and_chunks_give_the_same_events() {
        let (x, n) = signal();
        let neg: Vec<f32> = x.iter().map(|v| -v).collect();
        let mut o = OnsetOptions::new(-5.0);
        o.releases = vec![-2.0];
        o.direction = Direction::Falling;
        let whole = detect_onsets(&neg, 2, n, &o).unwrap();
        assert_eq!(whole.iter().map(|e| e.peak_value).collect::<Vec<_>>(), vec![-9.0, -10.0, -7.0]);
        // Fed in windows of 7 samples (rows of each window copied out)
        let mut t = OnsetTracker::new(2, &o).unwrap();
        for start in (0..n).step_by(7) {
            let end = (start + 7).min(n);
            let w = end - start;
            let rows: Vec<f32> = (0..2).flat_map(|c| neg[c * n + start..c * n + end].to_vec()).collect();
            t.push_host_window(&rows, w, 0..w, start as u64);
        }
        assert_eq!(t.finish(), whole);
    }

    #[test]
    fn bad_options_are_refused() {
        assert!(detect_onsets(&[0.0; 4], 2, 2, &OnsetOptions { levels: vec![1.0, 2.0, 3.0], ..OnsetOptions::new(0.0) }).is_err());
        let mut o = OnsetOptions::new(5.0);
        o.releases = vec![6.0];
        assert!(detect_onsets(&[0.0; 4], 2, 2, &o).is_err(), "a rising release above the level");
    }
}
