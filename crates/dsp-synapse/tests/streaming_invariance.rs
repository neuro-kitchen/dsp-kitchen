//! The out-of-core sorter returns the same spikes for any batch size, and the same as one pass over
//! the whole recording (filter → detect → deduplicate).

use cubecl::Runtime;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use dsp_base::pipeline::{Pipeline, PipelineStage, PipelineWorkspace};
use dsp_core::layout::{Position3D, SensorLayout, SensorSite};
use dsp_core::{MemoryRecording, RecordingSource};
use dsp_synapse::detection::{DeduplicatedSpike, deduplicate_spikes_spatial, detect_spikes_with_sigma};
use dsp_synapse::streaming::{StreamingSortConfig, StreamingSpikeRunner, calibrate_noise};

const FS: f64 = 30_000.0;
const CHANNELS: usize = 16;
const SECONDS: usize = 12;

fn probe() -> SensorLayout {
    // Two-column staggered layout, 20 µm pitch.
    let sites = (0..CHANNELS)
        .map(|c| SensorSite::new(c, Position3D::new((c % 2) as f32 * 32.0, (c / 2) as f32 * 20.0, 0.0), 0))
        .collect();
    SensorLayout::new("synthetic-16", sites)
}

/// Deterministic Gaussian noise (xorshift + Box-Muller).
fn noise(n: usize, seed: u64, sigma: f32) -> Vec<f32> {
    let mut state = seed | 1;
    let mut uniform = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        ((state >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    };
    (0..n)
        .map(|_| {
            let (u1, u2) = (uniform(), uniform());
            ((-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()) as f32 * sigma
        })
        .collect()
}

/// Spike times: on and around every boundary of the tested batch sizes, plus a regular train.
fn spike_times() -> Vec<(usize, usize)> {
    let n = SECONDS * FS as usize;
    let mut times = Vec::new();
    for batch in [30_000usize, 111_000, 300_000] {
        for edge in (batch..n).step_by(batch) {
            for dt in [-40isize, -3, 0, 2, 37] {
                times.push((edge as isize + dt) as usize);
            }
        }
    }
    times.extend((5_000..n - 5_000).step_by(7_919));
    times.retain(|t| burst_edges().iter().all(|e| t.abs_diff(*e) > 300));
    times.sort_unstable();
    times.dedup_by(|a, b| *a - *b < 90); // keep spikes further apart than 3 ms
    times.into_iter().enumerate().map(|(i, t)| (t, (i * 5) % CHANNELS)).collect()
}

fn burst_edges() -> [usize; 2] {
    [60_000, 150_000]
}

fn recording() -> MemoryRecording {
    let n = SECONDS * FS as usize;
    let mut data = noise(CHANNELS * n, 0x5eed, 4.0);
    // Offsets like real probes.
    for ch in 0..CHANNELS {
        for v in &mut data[ch * n..(ch + 1) * n] {
            *v += -300.0 + 20.0 * ch as f32;
        }
    }
    let layout = probe();
    for (t, ch) in spike_times() {
        let p = layout.contacts[ch].position;
        for other in 0..CHANNELS {
            let d = p.distance_to(&layout.contacts[other].position);
            let gain = (-d / 25.0).exp();
            if gain < 0.05 {
                continue;
            }
            for k in 0..40usize {
                let x = k as f32 - 12.0;
                let w = -150.0 * (-0.5 * (x / 2.5).powi(2)).exp() + 40.0 * (-0.5 * ((x - 9.0) / 5.0).powi(2)).exp();
                data[other * n + t - 12 + k] += gain * w;
            }
        }
    }
    // Bursts on channel 3 straddling 1 s boundaries: troughs 20 and 35 samples apart, the middle one
    // deepest. One pass keeps the first and third (refractory greedy); a window split between the
    // first two without carried refractory state keeps only the middle one.
    for edge in burst_edges() {
        for (dt, amp) in [(-10isize, -110.0f32), (10, -160.0), (25, -100.0)] {
            let t = (edge as isize + dt) as usize;
            for k in 0..9usize {
                let x = k as f32 - 4.0;
                data[3 * n + t - 4 + k] += amp * (-0.5 * (x / 1.2).powi(2)).exp();
            }
        }
    }
    MemoryRecording::new("synthetic", data, CHANNELS, FS).unwrap()
}

fn pipeline() -> Pipeline {
    Pipeline::with_stages(vec![PipelineStage::bandpass(300.0, 6000.0)])
}

fn config(batch_sec: f64) -> StreamingSortConfig {
    StreamingSortConfig { batch_duration_sec: batch_sec, spatial_radius_um: 40.0, ..Default::default() }
}

fn key(spikes: &[DeduplicatedSpike]) -> Vec<(u64, usize, Vec<usize>)> {
    spikes.iter().map(|s| (s.sample_index, s.primary_channel, s.participating_channels.clone())).collect()
}

/// Filter the whole recording in one chunk, detect and deduplicate on the host.
fn whole_recording(rec: &MemoryRecording, cfg: &StreamingSortConfig) -> Vec<DeduplicatedSpike> {
    let n = rec.info().samples as usize;
    let client = WgpuRuntime::client(&WgpuDevice::default());
    let mut ws = PipelineWorkspace::<WgpuRuntime>::new(client, pipeline(), CHANNELS, n, FS).unwrap();
    let halos = cfg.compute_halos(FS, &pipeline()).unwrap();
    let sigmas = calibrate_noise(rec, &mut ws, cfg, halos).unwrap();
    let mut filtered = vec![0.0; rec.data().len()];
    ws.process_chunk(rec.data(), n, &mut filtered);
    let range = cfg.detection_range(FS, n as u64);
    let crossings: Vec<_> = detect_spikes_with_sigma(&filtered, CHANNELS, n, &sigmas, cfg.threshold_factor, cfg.refractory_samples(FS))
        .into_iter()
        .filter(|s| range.contains(&s.sample_index))
        .collect();
    deduplicate_spikes_spatial(&crossings, &probe(), cfg.spatial_radius_um, cfg.refractory_samples(FS) as u64)
}

#[test]
fn spikes_are_identical_for_any_batch_size_and_whole_recording() {
    let rec = recording();
    let layout = probe();
    let injected = spike_times().len();

    let reference = whole_recording(&rec, &config(10.0));
    assert!(reference.len() >= injected, "reference found {} of {injected} spikes", reference.len());

    let mut templates = Vec::new();
    for batch in [1.0, 3.7, 10.0] {
        let result = StreamingSpikeRunner::new(config(batch)).run(&rec, &pipeline(), &layout).unwrap();
        assert_eq!(key(&result.spikes), key(&reference), "batch {batch} s differs from the whole-recording run");
        for (a, b) in result.spikes.iter().zip(&reference) {
            assert!((a.peak_amplitude_uv - b.peak_amplitude_uv).abs() < 0.05, "amplitude differs at {}", a.sample_index);
        }
        templates.push(result.channel_templates);
    }
    for other in &templates[1..] {
        for (a, b) in templates[0].iter().zip(other) {
            match (a, b) {
                (Some(a), Some(b)) => {
                    let err = a.mean.iter().zip(&b.mean).fold(0.0f32, |m, (x, y)| m.max((x - y).abs()));
                    assert!(err < 0.05, "template mean differs by {err} µV");
                }
                (None, None) => {}
                _ => panic!("template present for one batch size only"),
            }
        }
    }

    // Bursts: the refractory chain is decided as in one pass, whatever the window split.
    for edge in burst_edges() {
        let hits: Vec<i64> = reference
            .iter()
            .filter(|s| s.primary_channel == 3 && s.sample_index.abs_diff(edge as u64) < 60)
            .map(|s| s.sample_index as i64 - edge as i64)
            .collect();
        assert_eq!(hits.len(), 2, "burst at {edge}: {hits:?}");
    }

    // Every injected spike is found once, on its own channel.
    for (t, ch) in spike_times() {
        let hits: Vec<_> = reference.iter().filter(|s| s.sample_index.abs_diff(t as u64) <= 3).collect();
        assert_eq!(hits.len(), 1, "spike at {t} found {} times", hits.len());
        assert_eq!(hits[0].primary_channel, ch, "spike at {t}");
    }
}

#[test]
fn every_runtime_returns_the_same_spikes() {
    use dsp_core::ComputeTarget;
    let rec = recording();
    let reference = whole_recording(&rec, &config(10.0));
    for target in ComputeTarget::available() {
        let result = StreamingSpikeRunner::new(config(3.7)).run_with(target, &rec, &pipeline(), &probe()).unwrap();
        assert_eq!(key(&result.spikes), key(&reference), "{target}");
    }
}

/// int16 recording (0.25 µV per unit, per-channel offsets) with native stored reads.
struct I16Recording {
    info: dsp_core::RecordingInfo,
    data: Vec<i16>,
}

impl I16Recording {
    fn from(rec: &MemoryRecording) -> Self {
        let n = rec.info().samples as usize;
        let mut values = vec![0.0f32; CHANNELS * n];
        rec.read(&(0..CHANNELS).collect::<Vec<_>>(), 0..n as u64, &mut values).unwrap();
        let offsets: Vec<f32> = (0..CHANNELS).map(|c| -300.0 + 20.0 * c as f32).collect();
        let data = values.iter().enumerate().map(|(i, v)| ((v - offsets[i / n]) / 0.25).round() as i16).collect();
        let mut info = dsp_core::RecordingInfo::new(
            "i16",
            CHANNELS,
            n as u64,
            dsp_core::SampleRate::new(FS).unwrap(),
            dsp_core::SampleFormat::I16,
            dsp_core::MemoryOrder::ChannelMajor,
        )
        .with_gain_uv(0.25);
        for (c, ch) in info.channels.iter_mut().enumerate() {
            ch.offset_uv = offsets[c];
        }
        Self { info, data }
    }
}

impl RecordingSource for I16Recording {
    fn info(&self) -> &dsp_core::RecordingInfo {
        &self.info
    }
    fn read(&self, channels: &[usize], samples: std::ops::Range<u64>, out: &mut [f32]) -> dsp_core::DspResult<()> {
        let n = dsp_core::recording::check_read(&self.info, channels, &samples, out.len())?;
        let total = self.info.samples as usize;
        for (dst, &c) in out.chunks_exact_mut(n.max(1)).zip(channels) {
            let row = &self.data[c * total + samples.start as usize..c * total + samples.end as usize];
            for (o, &v) in dst.iter_mut().zip(row) {
                *o = v as f32 * 0.25 + self.info.channels[c].offset_uv;
            }
        }
        Ok(())
    }
    fn read_stored(&self, channels: &[usize], samples: std::ops::Range<u64>, out: &mut [u8]) -> dsp_core::DspResult<()> {
        let n = dsp_core::recording::check_read_stored(&self.info, channels, &samples, out.len())?;
        let total = self.info.samples as usize;
        for (dst, &c) in out.chunks_exact_mut((2 * n).max(1)).zip(channels) {
            let row = &self.data[c * total + samples.start as usize..c * total + samples.end as usize];
            for (o, &v) in dst.chunks_exact_mut(2).zip(row) {
                o.copy_from_slice(&v.to_le_bytes());
            }
        }
        Ok(())
    }
}

/// The same recording without stored reads (forces the f32 upload path).
struct F32Only<'a>(&'a I16Recording);

impl RecordingSource for F32Only<'_> {
    fn info(&self) -> &dsp_core::RecordingInfo {
        self.0.info()
    }
    fn read(&self, channels: &[usize], samples: std::ops::Range<u64>, out: &mut [f32]) -> dsp_core::DspResult<()> {
        self.0.read(channels, samples, out)
    }
}

#[test]
fn stored_int16_upload_gives_the_same_spikes_as_f32_upload() {
    use dsp_core::ComputeTarget;
    let rec = I16Recording::from(&recording());
    assert!(F32Only(&rec).read_stored(&[0], 0..1, &mut [0u8; 2]).is_err(), "wrapper has no stored reads");
    for target in ComputeTarget::available() {
        let runner = StreamingSpikeRunner::new(config(3.7));
        let stored = runner.run_with(target, &rec, &pipeline(), &probe()).unwrap();
        let f32_path = runner.run_with(target, &F32Only(&rec), &pipeline(), &probe()).unwrap();
        assert!(!stored.spikes.is_empty());
        assert_eq!(key(&stored.spikes), key(&f32_path.spikes), "{target}");
    }
}
