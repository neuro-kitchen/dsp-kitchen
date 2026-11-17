//! The generator: noise, spikes and a sine ([`SyntheticRecording`]), handed out in blocks at the
//! pace of its sample rate. The first test source of a device (`device.toml`: `kind = "generator"`).

use std::time::{Duration, Instant};

use dspk_core::{ChannelInfo, DspError, DspResult, RecordingSource, SampleFormat, SignalUnit, SyntheticParams, SyntheticRecording};
use serde::{Deserialize, Serialize};

use super::{block_samples, encode_stored, sample_time_ns, unix_now_ns, LiveBlock, LiveEvent, LiveLayout, LiveSource};

/// How long a generator can run: the procedural recording is made this long (samples are computed
/// on demand, so its length costs nothing).
const GENERATOR_SPAN_SEC: f64 = 365.0 * 24.0 * 3600.0;

/// The generator's settings (a `device.toml` source of `kind = "generator"`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct GeneratorParams {
    pub channels: usize,
    pub sample_rate_hz: f64,
    /// The stored type of its blocks: `int16`, `int32` or `float32`.
    pub format: SampleFormat,
    /// µV per stored step for integer formats (ignored for float32).
    pub gain_uv: f32,
    /// Same seed, same signal.
    pub seed: u64,
    pub noise_uv: f32,
    /// Simulated units (0: no spikes).
    pub units: usize,
    /// Every unit's mean rate (Hz); absent: each unit its own, 2 – 20 Hz.
    pub spike_rate_hz: Option<f32>,
    /// Every unit's peak amplitude (µV); absent: each unit its own, 60 – 200 µV.
    pub spike_amplitude_uv: Option<f32>,
    /// A sine on every channel (Hz; 0: none) and its amplitude.
    pub sine_hz: f64,
    pub sine_uv: f32,
    /// 60 Hz line hum (µV).
    pub line_noise_uv: f32,
    /// Length of each block handed out (ms).
    pub block_ms: f64,
}

impl Default for GeneratorParams {
    fn default() -> Self {
        Self {
            channels: 32,
            sample_rate_hz: 30_000.0,
            format: SampleFormat::I16,
            // A typical extracellular headstage's step
            gain_uv: 0.195,
            seed: 0x5eed,
            noise_uv: 10.0,
            units: 4,
            spike_rate_hz: None,
            spike_amplitude_uv: None,
            sine_hz: 10.0,
            sine_uv: 20.0,
            line_noise_uv: 0.0,
            block_ms: 1.0,
        }
    }
}

pub struct Generator {
    layout: LiveLayout,
    signal: SyntheticRecording,
    block: usize,
    next: u64,
    started: Option<Instant>,
}

impl Generator {
    pub fn new(name: impl Into<String>, params: &GeneratorParams) -> DspResult<Self> {
        if !matches!(params.format, SampleFormat::I16 | SampleFormat::I32 | SampleFormat::F32) {
            return Err(DspError::InvalidConfig(format!("generator format {}: int16, int32 or float32", params.format.name())));
        }
        if params.channels == 0 || params.block_ms <= 0.0 {
            return Err(DspError::InvalidConfig("a generator needs channels and a positive block length".into()));
        }
        let signal = SyntheticRecording::new(SyntheticParams {
            channels: params.channels,
            sample_rate_hz: params.sample_rate_hz,
            duration_sec: GENERATOR_SPAN_SEC,
            noise_uv: params.noise_uv,
            line_noise_uv: params.line_noise_uv,
            units: params.units,
            drift_channels: 0.0,
            seed: params.seed,
            spike_rate_hz: params.spike_rate_hz,
            spike_amplitude_uv: params.spike_amplitude_uv,
            sine_hz: params.sine_hz,
            sine_uv: params.sine_uv,
            ..Default::default()
        })?;
        let gain = if params.format == SampleFormat::F32 { 1.0 } else { params.gain_uv };
        let channels = (0..params.channels).map(|i| ChannelInfo { name: format!("ch{i}"), gain, offset: 0.0, unit: SignalUnit::Microvolt }).collect();
        let sample_rate = signal.info().sample_rate;
        let layout = LiveLayout { name: name.into(), channels, sample_rate, format: params.format };
        Ok(Self { block: block_samples(params.block_ms, sample_rate), layout, signal, next: 0, started: None })
    }

    /// The block starting at `first`, as a device would receive it (without waiting).
    pub fn block_at(&self, first: u64, samples: usize) -> DspResult<LiveBlock> {
        let all: Vec<usize> = (0..self.layout.channel_count()).collect();
        let mut values = vec![0.0f32; all.len() * samples];
        self.signal.read(&all, first..first + samples as u64, &mut values)?;
        let data = encode_stored(&values, &self.layout.channels, samples, self.layout.format)?;
        Ok(LiveBlock {
            first_sample: first,
            samples,
            source_time_ns: sample_time_ns(first, self.layout.sample_rate),
            received_unix_ns: unix_now_ns(),
            lost_before: 0,
            data,
        })
    }
}

impl LiveSource for Generator {
    fn layout(&self) -> &LiveLayout {
        &self.layout
    }

    fn next(&mut self, wait: Duration) -> DspResult<LiveEvent> {
        let started = *self.started.get_or_insert_with(Instant::now);
        let end = self.next + self.block as u64;
        // A block is ready once its last sample's time has passed
        let due = started + Duration::from_nanos(sample_time_ns(end, self.layout.sample_rate));
        let now = Instant::now();
        if due > now {
            let pause = due - now;
            if pause > wait {
                std::thread::sleep(wait);
                return Ok(LiveEvent::Idle);
            }
            std::thread::sleep(pause);
        }
        let block = self.block_at(self.next, self.block)?;
        self.next = end;
        Ok(LiveEvent::Block(block))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blocks_follow_each_other_and_repeat_with_the_seed() {
        let params = GeneratorParams { channels: 4, sample_rate_hz: 1000.0, block_ms: 10.0, ..Default::default() };
        let mut g = Generator::new("gen", &params).unwrap();
        let mut blocks = Vec::new();
        while blocks.len() < 3 {
            if let LiveEvent::Block(b) = g.next(Duration::from_millis(50)).unwrap() {
                blocks.push(b);
            }
        }
        assert_eq!(blocks[0].first_sample, 0);
        assert_eq!(blocks[1].first_sample, 10);
        assert_eq!(blocks[2].end_sample(), 30);
        for b in &blocks {
            b.check(g.layout()).unwrap();
        }
        let again = Generator::new("gen", &params).unwrap().block_at(10, 10).unwrap();
        assert_eq!(again.data, blocks[1].data);
    }

    #[test]
    fn paced_at_its_rate() {
        let params = GeneratorParams { channels: 1, sample_rate_hz: 1000.0, block_ms: 20.0, ..Default::default() };
        let mut g = Generator::new("gen", &params).unwrap();
        let t0 = Instant::now();
        let mut got = 0;
        while got < 5 {
            if let LiveEvent::Block(_) = g.next(Duration::from_millis(100)).unwrap() {
                got += 1;
            }
        }
        // 5 blocks of 20 ms: 100 ms of signal, not much faster or slower
        let elapsed = t0.elapsed();
        assert!(elapsed >= Duration::from_millis(95) && elapsed < Duration::from_millis(400), "{elapsed:?}");
    }

    #[test]
    fn refuses_other_formats() {
        let params = GeneratorParams { format: SampleFormat::F64, ..Default::default() };
        assert!(Generator::new("gen", &params).is_err());
    }
}
