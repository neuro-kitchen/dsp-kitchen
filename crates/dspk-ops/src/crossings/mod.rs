//! Threshold crossings as events: where each channel goes past a level (above it, below it, or
//! either) and where it comes back. One tool for any signal: muscle contractions on an EMG
//! envelope (band-pass → rectify → low-pass → crossings), excursions of a raw trace past ±level,
//! a temperature above a limit, a trigger line going high.
//!
//! - An event **starts** (`onset`) where the signal crosses `level` and **ends** (`offset`) where
//!   it crosses back past `release` (hysteresis: a release between the baseline and the level
//!   keeps noise around the level from splitting one event into many; by default `release =
//!   level`, a plain threshold crossing).
//! - While an event is open further crossings are part of it; a new event needs a fresh crossing.
//! - Events shorter than `min_duration` are dropped; after an event, onsets within `refractory`
//!   of its end are ignored (the signal must then cross the level again).
//! - Each event reports its peak (the largest value; the lowest for a crossing downwards) and
//!   where it is.
//!
//! The work is split as `find_peak_candidates` is: kernels find the crossings of every channel and
//! compact them on the device, and the extremum of each stretch between crossings; only those are
//! downloaded. [`CrossingTracker`] then pairs them into events on the host, window after window, so
//! a long recording streams through in chunks and events may span chunks: the result does not
//! depend on the chunk size.
//!
//! - [`detect_crossings`] (host): a whole `[channels, samples]` buffer in memory.
//! - [`detect_crossings_on_device`] (device): a whole device buffer.
//! - [`CrossingTracker::push_device_window`]: one window of a longer run (what pipelines use).

mod device;
pub mod kernels;

use crate::error::{OpsError, OpsResult};

pub use device::detect_crossings_on_device;

fn invalid(msg: String) -> OpsError {
    OpsError::Core(dspk_core::DspError::InvalidConfig(msg))
}

/// Which way the signal goes past the level at an onset.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Direction {
    /// Up through `level` (a level above the baseline; envelopes, power).
    #[default]
    Rising,
    /// Down through `level` (a level below the baseline).
    Falling,
    /// Past `+|level|` upwards or `−|level|` downwards (deflections either way on a raw trace);
    /// `release` is a magnitude too. An event's sign is its peak's.
    Both,
}

/// Options of [`detect_crossings`] and [`CrossingTracker`].
#[derive(Debug, Clone, PartialEq)]
pub struct CrossingOptions {
    /// Level an onset crosses, in the signal's unit: one value for every channel, or one per
    /// channel (`Both`: magnitudes).
    pub levels: Vec<f32>,
    /// Level the signal crosses back past to end the event (one, or one per channel); empty:
    /// `levels`. Rising: at most the level; falling: at least the level; both: a magnitude at most
    /// the level's.
    pub releases: Vec<f32>,
    pub direction: Direction,
    /// Events shorter than this many samples are dropped (default 0: none).
    pub min_duration_samples: u64,
    /// Onsets within this many samples after an event's end are ignored (default 0).
    pub refractory_samples: u64,
}

impl CrossingOptions {
    /// One level for every channel, released at the same level, rising.
    pub fn new(level: f32) -> Self {
        Self { levels: vec![level], releases: Vec::new(), direction: Direction::Rising, min_duration_samples: 0, refractory_samples: 0 }
    }

    /// The sides to watch: each a sign and per-channel levels and releases in signed space
    /// (`sign · value`), checked.
    fn sides(&self, channels: usize) -> OpsResult<Vec<Side>> {
        let spread = |v: &[f32], what: &str| -> OpsResult<Vec<f32>> {
            match v.len() {
                1 => Ok(vec![v[0]; channels]),
                n if n == channels => Ok(v.to_vec()),
                n => Err(invalid(format!("crossings: {n} {what} for {channels} channels (give one, or one per channel)"))),
            }
        };
        let levels = spread(&self.levels, "levels")?;
        let releases = if self.releases.is_empty() { levels.clone() } else { spread(&self.releases, "releases")? };
        for (ch, (&l, &r)) in levels.iter().zip(&releases).enumerate() {
            if !l.is_finite() || !r.is_finite() {
                return Err(invalid(format!("crossings: channel {ch}: level and release must be finite")));
            }
            let (bad, rule) = match self.direction {
                Direction::Rising => (r > l, "at most"),
                Direction::Falling => (r < l, "at least"),
                Direction::Both => (r.abs() > l.abs(), "in magnitude at most"),
            };
            if bad {
                return Err(invalid(format!("crossings: channel {ch}: the release ({r}) must be {rule} the level ({l})")));
            }
        }
        let side = |sign: f32, magnitude: bool| {
            let to = |v: &f32| if magnitude { v.abs() } else { sign * v };
            Side::new(sign, levels.iter().map(to).collect(), releases.iter().map(to).collect())
        };
        Ok(match self.direction {
            Direction::Rising => vec![side(1.0, false)],
            Direction::Falling => vec![side(-1.0, false)],
            Direction::Both => vec![side(1.0, true), side(-1.0, true)],
        })
    }
}

/// One event of one channel, in samples of the whole run.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CrossingEvent {
    pub channel: usize,
    /// First sample at or past the level.
    pub onset: u64,
    /// First sample back past the release; `None` when the run ended during the event.
    pub offset: Option<u64>,
    /// Where the peak is (the first sample of the largest value; the lowest for a crossing
    /// downwards).
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

/// One direction watched: a sign, levels and releases in signed space, each channel's state.
#[derive(Debug, Clone)]
struct Side {
    sign: f32,
    levels: Vec<f32>,
    releases: Vec<f32>,
    state: Vec<ChannelState>,
}

impl Side {
    fn new(sign: f32, levels: Vec<f32>, releases: Vec<f32>) -> Self {
        let idle = ChannelState { on: false, onset: 0, peak: f32::NEG_INFINITY, peak_at: 0, ready_at: 0 };
        let n = levels.len();
        Self { sign, levels, releases, state: vec![idle; n] }
    }
}

/// Pairs crossings into events, window after window (see the module docs).
#[derive(Debug, Clone)]
pub struct CrossingTracker {
    /// One side per direction watched (two for `Both`).
    sides: Vec<Side>,
    min_duration: u64,
    refractory: u64,
    /// Last value of each channel in the previous window (NaN before the first).
    prev: Vec<f32>,
    events: Vec<CrossingEvent>,
}

/// Crossings of one channel in one window: local sample and kind
/// ([`kernels::CROSSING_UP`] / [`kernels::CROSSING_DOWN`]).
type Crossings<'a> = &'a [(u32, u32)];
/// Signed extremum and its local sample of each stretch: before the first crossing, then after
/// each crossing (one more than the crossings).
type Stretches<'a> = &'a [(f32, u32)];

impl CrossingTracker {
    pub fn new(channels: usize, options: &CrossingOptions) -> OpsResult<Self> {
        Ok(Self {
            sides: options.sides(channels)?,
            min_duration: options.min_duration_samples,
            refractory: options.refractory_samples,
            prev: vec![f32::NAN; channels],
            events: Vec::new(),
        })
    }

    pub fn channels(&self) -> usize {
        self.prev.len()
    }

    /// Takes one channel's crossings on side `side` of a window whose local sample 0 is sample
    /// `origin` of the run.
    fn take(&mut self, side: usize, ch: usize, origin: u64, crossings: Crossings, stretches: Stretches) {
        debug_assert_eq!(stretches.len(), crossings.len() + 1);
        let sign = self.sides[side].sign;
        let st = &mut self.sides[side].state[ch];
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
                    self.events.push(CrossingEvent { channel: ch, onset: st.onset, offset: Some(at), peak_sample: st.peak_at, peak_value: sign * st.peak });
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
        for side in 0..self.sides.len() {
            for ch in 0..self.channels() {
                let row = &rows[ch * samples..(ch + 1) * samples];
                let Side { sign: s, ref levels, ref releases, .. } = self.sides[side];
                let (l, r) = (levels[ch], releases[ch]);
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
                self.take(side, ch, origin, &crossings, &stretches);
            }
        }
        if scan.end > 0 {
            for ch in 0..self.channels() {
                self.prev[ch] = rows[ch * samples + scan.end - 1];
            }
        }
    }

    /// Ends the run: events still open get no offset. Returns every event, by onset then channel.
    pub fn finish(mut self) -> Vec<CrossingEvent> {
        for side in &self.sides {
            for (ch, st) in side.state.iter().enumerate() {
                if st.on {
                    self.events.push(CrossingEvent { channel: ch, onset: st.onset, offset: None, peak_sample: st.peak_at, peak_value: side.sign * st.peak });
                }
            }
        }
        self.events.sort_by(|a, b| (a.onset, a.channel).cmp(&(b.onset, b.channel)).then(a.peak_value.total_cmp(&b.peak_value)));
        self.events
    }
}

/// Threshold-crossing events of every channel of a `[channels, samples]` buffer in memory.
pub fn detect_crossings(rows: &[f32], channels: usize, samples: usize, options: &CrossingOptions) -> OpsResult<Vec<CrossingEvent>> {
    if rows.len() != channels * samples {
        return Err(invalid(format!("crossings: {} values for {channels} × {samples}", rows.len())));
    }
    let mut tracker = CrossingTracker::new(channels, options)?;
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
        let mut o = CrossingOptions::new(5.0);
        o.releases = vec![2.0];
        let ev = detect_crossings(&x, 2, n, &o).unwrap();
        let got: Vec<_> = ev.iter().map(|e| (e.channel, e.onset, e.offset, e.peak_sample, e.peak_value)).collect();
        // The dip to 4 stays above the release: one event 20..24, peak 10 at 22
        assert_eq!(got, vec![(0, 5, Some(8), 6, 9.0), (0, 20, Some(24), 22, 10.0), (1, 32, Some(35), 34, 7.0)]);
        // Without hysteresis the dip splits the second event
        let ev = detect_crossings(&x, 2, n, &CrossingOptions::new(5.0)).unwrap();
        assert_eq!(ev.iter().filter(|e| e.channel == 0).count(), 3);
    }

    #[test]
    fn min_duration_refractory_and_open_events() {
        let (x, n) = signal();
        let mut o = CrossingOptions::new(5.0);
        o.min_duration_samples = 4;
        let ev = detect_crossings(&x, 2, n, &o).unwrap();
        assert!(ev.iter().all(|e| e.offset.unwrap() - e.onset >= 4), "{ev:?}");
        let mut o = CrossingOptions::new(5.0);
        o.releases = vec![2.0];
        o.refractory_samples = 20;
        let ev = detect_crossings(&x[..n], 1, n, &o).unwrap();
        assert_eq!(ev.len(), 1, "the second bump starts within 20 samples of the first's end");
        // A run ending inside an event
        let ev = detect_crossings(&x[..21], 1, 21, &CrossingOptions::new(5.0)).unwrap();
        assert_eq!(ev.last().map(|e| (e.onset, e.offset)), Some((20, None)));
    }

    #[test]
    fn falling_and_chunks_give_the_same_events() {
        let (x, n) = signal();
        let neg: Vec<f32> = x.iter().map(|v| -v).collect();
        let mut o = CrossingOptions::new(-5.0);
        o.releases = vec![-2.0];
        o.direction = Direction::Falling;
        let whole = detect_crossings(&neg, 2, n, &o).unwrap();
        assert_eq!(whole.iter().map(|e| e.peak_value).collect::<Vec<_>>(), vec![-9.0, -10.0, -7.0]);
        // Fed in windows of 7 samples (rows of each window copied out)
        let mut t = CrossingTracker::new(2, &o).unwrap();
        for start in (0..n).step_by(7) {
            let end = (start + 7).min(n);
            let w = end - start;
            let rows: Vec<f32> = (0..2).flat_map(|c| neg[c * n + start..c * n + end].to_vec()).collect();
            t.push_host_window(&rows, w, 0..w, start as u64);
        }
        assert_eq!(t.finish(), whole);
    }

    #[test]
    fn both_directions_at_once() {
        let (x, n) = signal();
        // Channel 0 as is, plus a negative bump at 30..33
        let mut y = x[..n].to_vec();
        for t in 30..33 {
            y[t] = -8.0;
        }
        let mut o = CrossingOptions::new(5.0);
        o.releases = vec![2.0];
        o.direction = Direction::Both;
        let ev = detect_crossings(&y, 1, n, &o).unwrap();
        let got: Vec<_> = ev.iter().map(|e| (e.onset, e.offset, e.peak_value)).collect();
        assert_eq!(got, vec![(5, Some(8), 9.0), (20, Some(24), 10.0), (30, Some(33), -8.0)]);
        // A negative level is taken as its magnitude
        o.levels = vec![-5.0];
        assert_eq!(detect_crossings(&y, 1, n, &o).unwrap(), ev);
    }

    #[test]
    fn bad_options_are_refused() {
        assert!(detect_crossings(&[0.0; 4], 2, 2, &CrossingOptions { levels: vec![1.0, 2.0, 3.0], ..CrossingOptions::new(0.0) }).is_err());
        let mut o = CrossingOptions::new(5.0);
        o.releases = vec![6.0];
        assert!(detect_crossings(&[0.0; 4], 2, 2, &o).is_err(), "a rising release above the level");
    }
}
