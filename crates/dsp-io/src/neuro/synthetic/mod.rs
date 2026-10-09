//! Procedural recordings of any length, computed on demand (no disk, no memory per sample).
//!
//! Every sample is a deterministic function of (seed, channel, sample index): noise, line hum,
//! and spike trains of a few units whose location drifts slowly across channels. Ground-truth
//! spike times are available for tests.

use std::ops::Range;

use dsp_core::recording::check_read;
use dsp_core::{DspResult, MemoryOrder, RecordingInfo, RecordingSource, SampleFormat, SampleRate};

/// Shape of a procedural recording.
#[derive(Debug, Clone, PartialEq)]
pub struct SyntheticParams {
    pub channels: usize,
    pub sample_rate_hz: f64,
    pub duration_sec: f64,
    /// Noise RMS.
    pub noise_uv: f32,
    /// Line-noise (60 Hz hum) amplitude.
    pub line_noise_uv: f32,
    /// Number of units (0 = no spikes).
    pub units: usize,
    /// Peak unit displacement over time, in channels.
    pub drift_channels: f32,
    pub drift_period_sec: f64,
    pub seed: u64,
}

impl Default for SyntheticParams {
    fn default() -> Self {
        Self {
            channels: 32,
            sample_rate_hz: 30_000.0,
            duration_sec: 60.0,
            noise_uv: 10.0,
            line_noise_uv: 20.0,
            units: 8,
            drift_channels: 2.0,
            drift_period_sec: 600.0,
            seed: 0x5eed,
        }
    }
}

#[derive(Debug, Clone)]
struct Unit {
    /// Channel position at t = 0.
    center: f32,
    amplitude_uv: f32,
    /// At most one spike per slot of this many samples (sets the maximum rate).
    slot: u64,
    /// Probability that a slot holds a spike.
    fire_prob: f32,
}

/// Channels on each side of a unit's center that receive its spikes.
const SPREAD: f32 = 4.0;
/// Width (channels) of the Gaussian fall-off of a spike's amplitude across channels.
const SPREAD_SIGMA: f32 = 1.5;

/// Spike waveform: window before and after the trough (s), and the trough's Gaussian width (s).
const WAVEFORM_PRE_SEC: f64 = 0.000_5;
const WAVEFORM_POST_SEC: f64 = 0.001_5;
const TROUGH_WIDTH_SEC: f32 = 0.000_25;
/// The positive rebound after the trough: amplitude (relative to the trough), and its delay and
/// width in trough widths.
const REBOUND_AMPLITUDE: f32 = 0.3;
const REBOUND_DELAY: f32 = 2.0;
const REBOUND_WIDTH: f32 = 1.5;

/// Units' mean firing rates are uniform in `MIN_RATE_HZ .. MIN_RATE_HZ + RATE_SPAN_HZ` (2–20 Hz),
/// their peak amplitudes in `MIN_AMPLITUDE_UV .. + AMPLITUDE_SPAN_UV` (60–200 µV).
const MIN_RATE_HZ: f32 = 2.0;
const RATE_SPAN_HZ: f32 = 18.0;
const MIN_AMPLITUDE_UV: f32 = 60.0;
const AMPLITUDE_SPAN_UV: f32 = 140.0;
/// Probability that a slot holds a spike; slots last `1 / (2 · rate)`, so `rate` spikes per second
/// on average.
const FIRE_PROBABILITY: f32 = 0.5;
/// A slot spans at least this many waveforms, so spikes of one unit never overlap.
const MIN_SLOT_WAVEFORMS: u64 = 2;

/// Line-noise frequency (Hz) and the phase step between channels (cycles per channel).
const LINE_FREQUENCY_HZ: f64 = 60.0;
const LINE_PHASE_PER_CHANNEL: f64 = 0.1;

/// Salts that separate the random streams (units, spike slots, channel noise) drawn from one seed.
const UNIT_SALT: u64 = 0x9e37_79b9;
const SLOT_SALT: u64 = 0xa24b_aed4_963e_e407;
const CHANNEL_SALT: u64 = 0xd6e8_feb8_6659_fd93;
/// Bit offset of the unit index in a slot's random key.
const UNIT_KEY_SHIFT: u32 = 48;

pub struct SyntheticRecording {
    info: RecordingInfo,
    params: SyntheticParams,
    units: Vec<Unit>,
    /// Unit-amplitude spike waveform; the trough sits at index `pre`.
    template: Vec<f32>,
    pre: usize,
}

impl SyntheticRecording {
    pub fn new(params: SyntheticParams) -> DspResult<Self> {
        let rate = SampleRate::new(params.sample_rate_hz)?;
        let samples = (params.duration_sec * params.sample_rate_hz).round().max(0.0) as u64;
        let mut info = RecordingInfo::new(
            format!("synthetic_{}ch_{}", params.channels, format_duration(params.duration_sec)),
            params.channels,
            samples,
            rate,
            SampleFormat::F32,
            MemoryOrder::ChannelMajor,
        );
        info.metadata.insert("source".into(), "procedural".into());

        let sr = params.sample_rate_hz;
        let (pre, post) = ((sr * WAVEFORM_PRE_SEC).round() as usize, (sr * WAVEFORM_POST_SEC).round() as usize);
        let template = (0..pre + post)
            .map(|i| {
                let t = (i as f32 - pre as f32) / (sr as f32 * TROUGH_WIDTH_SEC);
                -(-0.5 * t * t).exp() + REBOUND_AMPLITUDE * (-0.5 * ((t - REBOUND_DELAY) / REBOUND_WIDTH).powi(2)).exp()
            })
            .collect();

        let units = (0..params.units)
            .map(|u| {
                let h = splitmix(params.seed ^ (u as u64 + 1).wrapping_mul(UNIT_SALT));
                let rate_hz = MIN_RATE_HZ + unit_frac(h) * RATE_SPAN_HZ;
                Unit {
                    center: (u as f32 + 0.5) * params.channels as f32 / params.units as f32,
                    amplitude_uv: MIN_AMPLITUDE_UV + unit_frac(h >> 16) * AMPLITUDE_SPAN_UV,
                    slot: ((sr / (2.0 * rate_hz as f64)) as u64).max((pre + post) as u64 * MIN_SLOT_WAVEFORMS),
                    fire_prob: FIRE_PROBABILITY,
                }
            })
            .collect();

        Ok(Self { info, params, units, template, pre })
    }

    pub fn params(&self) -> &SyntheticParams {
        &self.params
    }

    pub fn unit_count(&self) -> usize {
        self.units.len()
    }

    /// Ground-truth trough sample indices of `unit` within `range`.
    pub fn spike_times(&self, unit: usize, range: Range<u64>) -> Vec<u64> {
        let mut out = Vec::new();
        self.for_each_spike(unit, range, |s| out.push(s));
        out
    }

    fn for_each_spike(&self, unit: usize, range: Range<u64>, mut f: impl FnMut(u64)) {
        let u = &self.units[unit];
        if range.start >= range.end {
            return;
        }
        let first = range.start / u.slot;
        let last = (range.end - 1) / u.slot;
        for k in first..=last {
            let h = splitmix(self.params.seed ^ ((unit as u64) << UNIT_KEY_SHIFT) ^ k.wrapping_mul(SLOT_SALT));
            if unit_frac(h) >= u.fire_prob {
                continue;
            }
            // Jitter within the first half of the slot keeps a refractory gap between spikes
            let t = k * u.slot + (h >> 32) % (u.slot / 2).max(1);
            if range.contains(&t) && t < self.info.samples {
                f(t);
            }
        }
    }

    /// Unit center in channels at sample `s`.
    fn center_at(&self, u: &Unit, s: u64) -> f32 {
        let p = &self.params;
        if p.drift_channels == 0.0 || p.drift_period_sec <= 0.0 {
            return u.center;
        }
        let phase = std::f64::consts::TAU * s as f64 / (p.drift_period_sec * p.sample_rate_hz);
        u.center + p.drift_channels * phase.sin() as f32
    }
}

impl RecordingSource for SyntheticRecording {
    fn info(&self) -> &RecordingInfo {
        &self.info
    }

    fn read(&self, channels: &[usize], samples: Range<u64>, out: &mut [f32]) -> DspResult<()> {
        let n = check_read(&self.info, channels, &samples, out.len())?;
        if n == 0 {
            return Ok(());
        }
        let p = &self.params;
        let omega = std::f64::consts::TAU * LINE_FREQUENCY_HZ / p.sample_rate_hz;

        // Background: noise + hum
        for (dst, &ch) in out.chunks_exact_mut(n).zip(channels) {
            let phase = (ch as f64 * LINE_PHASE_PER_CHANNEL).fract() * std::f64::consts::TAU;
            let ch_seed = p.seed ^ (ch as u64).wrapping_mul(CHANNEL_SALT);
            for (i, o) in dst.iter_mut().enumerate() {
                let s = samples.start + i as u64;
                let hum = p.line_noise_uv * (omega * s as f64 + phase).sin() as f32;
                *o = hum + p.noise_uv * gaussian(splitmix(ch_seed ^ s));
            }
        }

        // Spikes overlapping the window (a spike at t spans t - pre .. t + post)
        let (pre, len) = (self.pre as u64, self.template.len() as u64);
        let lo = samples.start.saturating_sub(len - pre);
        let hi = samples.end + pre;
        for (ui, u) in self.units.iter().enumerate() {
            self.for_each_spike(ui, lo..hi, |t| {
                let center = self.center_at(u, t);
                for (dst, &ch) in out.chunks_exact_mut(n).zip(channels) {
                    let d = ch as f32 - center;
                    if d.abs() > SPREAD {
                        continue;
                    }
                    let a = u.amplitude_uv * (-0.5 * (d / SPREAD_SIGMA).powi(2)).exp();
                    let t0 = t as i64 - pre as i64;
                    for (j, &w) in self.template.iter().enumerate() {
                        let s = t0 + j as i64 - samples.start as i64;
                        if s >= 0 && (s as usize) < n {
                            dst[s as usize] += a * w;
                        }
                    }
                }
            });
        }
        Ok(())
    }
}

fn splitmix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

/// Uniform in [0, 1) from the low 24 bits.
fn unit_frac(h: u64) -> f32 {
    (h & 0xff_ffff) as f32 / (1u64 << 24) as f32
}

/// Approximately standard normal (Irwin–Hall sum of four 16-bit uniforms).
fn gaussian(h: u64) -> f32 {
    let sum: u32 = (0..4).map(|i| ((h >> (16 * i)) & 0xffff) as u32).sum();
    (sum as f32 / 65536.0 - 2.0) * 3.0f32.sqrt()
}

fn format_duration(sec: f64) -> String {
    if sec >= 3600.0 && sec % 3600.0 == 0.0 {
        format!("{}h", sec / 3600.0)
    } else if sec >= 60.0 && sec % 60.0 == 0.0 {
        format!("{}m", sec / 60.0)
    } else {
        format!("{sec}s")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deterministic_and_chunk_independent() {
        let rec = SyntheticRecording::new(SyntheticParams { channels: 8, duration_sec: 2.0, ..Default::default() }).unwrap();
        let mut whole = vec![0.0; 3 * 6000];
        rec.read(&[1, 4, 7], 1000..7000, &mut whole).unwrap();

        // Same samples read in two pieces (split inside spikes) must match exactly
        let mut a = vec![0.0; 3 * 2500];
        let mut b = vec![0.0; 3 * 3500];
        rec.read(&[1, 4, 7], 1000..3500, &mut a).unwrap();
        rec.read(&[1, 4, 7], 3500..7000, &mut b).unwrap();
        for c in 0..3 {
            assert_eq!(&whole[c * 6000..c * 6000 + 2500], &a[c * 2500..(c + 1) * 2500]);
            assert_eq!(&whole[c * 6000 + 2500..(c + 1) * 6000], &b[c * 3500..(c + 1) * 3500]);
        }
    }

    #[test]
    fn test_spikes_appear_at_ground_truth() {
        let p = SyntheticParams { channels: 8, units: 1, noise_uv: 0.0, line_noise_uv: 0.0, drift_channels: 0.0, duration_sec: 5.0, ..Default::default() };
        let rec = SyntheticRecording::new(p).unwrap();
        let times = rec.spike_times(0, 0..rec.info().samples);
        // 2–20 Hz over 5 s
        assert!(times.len() >= 3, "{} spikes", times.len());
        assert!(times.windows(2).all(|w| w[1] > w[0]));

        let t = times[1];
        let center = rec.units[0].center.round() as usize;
        let mut out = vec![0.0; 1];
        rec.read(&[center], t..t + 1, &mut out).unwrap();
        assert!(out[0] < -30.0, "trough {}", out[0]);
    }

    #[test]
    fn test_hours_long_without_memory() {
        let rec = SyntheticRecording::new(SyntheticParams { channels: 384, duration_sec: 4.0 * 3600.0, ..Default::default() }).unwrap();
        assert_eq!(rec.info().samples, 432_000_000);
        assert_eq!(rec.info().name, "synthetic_384ch_4h");
        let mut out = vec![0.0; 2 * 300];
        rec.read(&[0, 383], rec.info().samples - 300..rec.info().samples, &mut out).unwrap();
        assert!(out.iter().all(|v| v.is_finite()));
    }
}
