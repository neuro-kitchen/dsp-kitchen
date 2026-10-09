//! MountainSort 5 over a whole recording: halo windows streamed through the device, as the
//! Kilosort4 runner (`kilosort4::runner`).
//!
//! 1. **Preprocessing** (SpikeInterface's wrapper): CAR (off), band-pass, notch (off), then global
//!    ZCA whitening fitted on `whitening_chunks` evenly spaced chunks (the second moment `X Xᵀ / n`
//!    of the filtered chunks, uncentred as SpikeInterface's `whiten`; it draws its chunks at
//!    random).
//! 2. **Phase 1** (scheme 1, or scheme 2's training stretch): detection and masked snippets per
//!    window ([`Detector`], [`MaskedSnippets`]), then [`super::cluster_events`].
//! 3. **Scheme 2**: a pass over the training stretch for each unit's snippets and the noise
//!    snippets, [`Classifiers::fit`], then every window of the recording detected and classified
//!    on the device.
//!
//! Spike amplitudes are the whitened trace at the detection (sample and channel, before the
//! alignment offsets), locations the detection channel's position: MountainSort 5 reports neither.

use std::ops::Range;

use cubecl::prelude::Client;
use dsp_base::filter::FilterSpec;
use dsp_base::linalg::SecondMomentAccumulator;
use dsp_base::pipeline::{Pipeline, PipelineStage, PipelineWorkspace};
use dsp_base::spatial::SpatialWhitening;
use dsp_core::progress::Stages;
use dsp_core::{DspError, DspResult, HaloWindow, ProgressSink, RecordingSource};
use dsp_io::neuro::probe::SensorLayout;
use dsp_synapse::core::{SortingOutput, WaveformTemplate};
use dsp_synapse::metrics::{CompositeScoreOptions, RefractoryOptions};

use super::detect::{DetectOptions, Detector};
use super::scheme1::{cluster_events_with_progress, Events, Scheme1Output};
use super::scheme2::{noise_times, remove_duplicate_events, subsample_indices, Classifiers};
use super::snippets::MaskedSnippets;
use super::templates::median_templates;
use super::{Mountainsort5Config, Scheme, TrainingSampling};
use crate::sorters::kilosort4::runner::DeviceWindows;
use crate::sorters::result::{AmplitudeScale, SorterResult};

/// Sorter name in [`SortingOutput`].
pub const MOUNTAINSORT5_SORTER: &str = "mountainsort5";

pub const STAGE_WHITENING: &str = "Fitting whitening";
pub const STAGE_PHASE1: &str = "Detecting (phase 1)";
pub const STAGE_CLUSTERING: &str = "Clustering (phase 1)";
pub const STAGE_TRAINING: &str = "Training classifiers";
pub const STAGE_CLASSIFYING: &str = "Classifying spikes";

/// Samples of context beyond the filters' settling: upstream's classification padding (1000).
const PADDING: u64 = 1000;
/// Upstream's training chunk (s).
const TRAINING_CHUNK_SEC: f64 = 10.0;
/// Values per classification chunk (upstream `100e6 / channels` samples).
const CHUNK_VALUES: f64 = 100e6;

/// A MountainSort 5 run.
#[derive(Debug, Clone)]
pub struct Mountainsort5Result {
    pub sample_rate_hz: f64,
    pub total_samples: u64,
    pub n_units: usize,
    /// Global samples, sorted.
    pub spike_samples: Vec<u64>,
    /// `0..n_units`.
    pub spike_units: Vec<u32>,
    /// Whitened value at the detection.
    pub spike_amplitudes: Vec<f32>,
    pub spike_channels: Vec<u32>,
    /// `[n_units, T, M]` median templates (whitened; zeros off the computed channels).
    pub templates: Vec<f32>,
    pub peak_channels: Vec<usize>,
    pub width: usize,
    pub channels: usize,
    /// `(x, y)` µm per channel.
    pub positions: Vec<[f32; 2]>,
    /// Channels of each unit's template (the snippet mask of its peak channel).
    pub template_channels: Vec<Vec<usize>>,
    /// Phase 1 spikes (scheme 2's training stretch; all spikes for scheme 1).
    pub phase1_spikes: usize,
    pub pipeline: Pipeline,
}

impl Mountainsort5Result {
    /// The sorter-independent result ([`SorterResult`]).
    pub fn sorter_result(&self) -> SorterResult {
        let (t, m) = (self.width, self.channels);
        let mut counts = vec![0usize; self.n_units];
        self.spike_units.iter().for_each(|&u| counts[u as usize] += 1);
        let templates = (0..self.n_units)
            .map(|u| {
                let chans = &self.template_channels[u];
                (!chans.is_empty()).then(|| {
                    let tp = &self.templates[u * t * m..(u + 1) * t * m];
                    let mean: Vec<f32> = chans.iter().flat_map(|&c| (0..t).map(move |s| tp[s * m + c])).collect();
                    WaveformTemplate::with_count(chans.clone(), t, counts[u].max(1), mean, vec![0.0; chans.len() * t])
                })
            })
            .collect();
        SorterResult {
            sorter: MOUNTAINSORT5_SORTER.into(),
            sample_rate_hz: self.sample_rate_hz,
            total_samples: self.total_samples,
            n_units: self.n_units,
            spike_samples: self.spike_samples.clone(),
            spike_units: self.spike_units.clone(),
            spike_amplitudes: self.spike_amplitudes.clone(),
            spike_locations: self.spike_channels.iter().map(|&c| [self.positions[c as usize][0], self.positions[c as usize][1], 0.0]).collect(),
            templates,
            amplitude_scale: AmplitudeScale::Whitened,
            label_refractory: RefractoryOptions::default(),
            composite: CompositeScoreOptions::default(),
        }
    }

    pub fn to_sorting_output(&self, probe: Option<SensorLayout>) -> SortingOutput {
        self.sorter_result().to_sorting_output(probe)
    }
}

/// The filter chain of `cfg` (CAR, band-pass or high-pass, notch; whitening is fitted after).
pub fn filter_stages(cfg: &Mountainsort5Config) -> Vec<PipelineStage> {
    let mut stages = Vec::new();
    if cfg.do_car {
        stages.push(PipelineStage::CommonAverageReference);
    }
    if cfg.do_bandpass {
        stages.push(PipelineStage::Filter(match cfg.bandpass_high_hz {
            Some(high) => FilterSpec::bandpass(cfg.bandpass_low_hz, high),
            None => FilterSpec::highpass(cfg.bandpass_low_hz),
        }));
    }
    if cfg.do_notch {
        stages.push(PipelineStage::Filter(FilterSpec::notch(cfg.notch_hz, cfg.notch_q)));
    }
    stages
}

/// Scheme 2's training segments (`[start, end)` global samples), upstream
/// `get_sampled_recording_for_training`.
pub fn training_segments(total: u64, fs: f64, duration_sec: Option<f64>, mode: TrainingSampling) -> Vec<Range<u64>> {
    let Some(sec) = duration_sec else { return vec![0..total] };
    let wanted = (sec * fs) as u64;
    if (sec * fs) >= total as f64 {
        return vec![0..total];
    }
    let initial = vec![0..wanted];
    if mode == TrainingSampling::Initial {
        return initial;
    }
    let chunk = (fs * TRAINING_CHUNK_SEC.min(sec)) as u64;
    let n = ((sec * fs) / chunk as f64).ceil() as u64;
    if n <= 1 {
        return initial;
    }
    let mut sizes = vec![chunk; n as usize];
    sizes[n as usize - 1] = wanted - (n - 1) * chunk;
    let spacing = (total - sizes.iter().sum::<u64>()) / (n - 1);
    let mut out = Vec::with_capacity(n as usize);
    let mut t = 0u64;
    for &s in &sizes {
        out.push(t..t + s);
        t += s + spacing;
    }
    out
}

fn filter_error(e: dsp_base::filter::FilterError) -> DspError {
    DspError::InvalidConfig(e.to_string())
}

/// Windows of `len` valid samples over each segment.
fn segment_windows(segments: &[Range<u64>], len: u64, halos: (u64, u64), total: u64) -> Vec<(HaloWindow, usize)> {
    let mut out = Vec::new();
    for (si, seg) in segments.iter().enumerate() {
        let mut s = seg.start;
        while s < seg.end {
            let e = (s + len).min(seg.end);
            out.push((HaloWindow::around(out.len(), s..e, halos.0, halos.1, total), si));
            s = e;
        }
    }
    out
}

/// `global ∩ window` in local samples.
fn local(range: &Range<u64>, window: &HaloWindow) -> Range<usize> {
    let a = range.start.max(window.read_global.start);
    let b = range.end.min(window.read_global.end);
    if a >= b {
        return 0..0;
    }
    (a - window.read_global.start) as usize..(b - window.read_global.start) as usize
}

/// Runs MountainSort 5 (module docs).
pub fn run(client: &Client, source: &dyn RecordingSource, probe: &SensorLayout, cfg: &Mountainsort5Config, progress: &dyn ProgressSink) -> DspResult<Mountainsort5Result> {
    let info = source.info();
    let (channels, total, fs) = (info.channel_count(), info.samples, info.sample_rate_hz());
    if probe.total_channels() != channels {
        return Err(DspError::InvalidConfig(format!("probe has {} sites, recording {channels} channels", probe.total_channels())));
    }
    let (t1, t2) = (cfg.snippet_t1, cfg.snippet_t2);
    if total < (t1 + t2) as u64 {
        return Err(DspError::InvalidConfig(format!("{total} samples: shorter than a snippet ({} samples)", t1 + t2)));
    }
    let positions: Vec<[f32; 2]> = probe.sites().iter().map(|s| [s.position.x_um, s.position.y_um]).collect();
    let mut names = Vec::new();
    if cfg.do_whiten {
        names.push((STAGE_WHITENING, "chunks"));
    }
    names.extend([(STAGE_PHASE1, "windows"), (STAGE_CLUSTERING, "spikes")]);
    if cfg.scheme == Scheme::Two {
        names.extend([(STAGE_TRAINING, "windows"), (STAGE_CLASSIFYING, "windows")]);
    }
    let stages = Stages::new(progress, &names);

    // 1. Preprocessing
    let mut pipeline_stages = filter_stages(cfg);
    let filtering = Pipeline::with_stages(pipeline_stages.clone());
    let (settle_l, settle_r) = filtering.settling(fs).map_err(filter_error)?;
    let radius = |ms: f64| (ms / 1000.0 * fs).ceil() as u64;
    let r_max = radius(cfg.detect_time_radius_ms).max(radius(cfg.phase1_detect_time_radius_ms));
    let margin = PADDING + (t1 + t2) as u64 + r_max;
    let halos = (settle_l as u64 + margin, settle_r as u64 + margin);
    let chunk = match cfg.classification_chunk_sec {
        Some(sec) => (sec * fs).ceil() as u64,
        None => (CHUNK_VALUES / channels.max(1) as f64).ceil() as u64,
    }
    .max(1);
    let max_read = (chunk + halos.0 + halos.1).min(total) as usize;
    dsp_core::compute::device_elements("a window of [channels, samples]", &[channels, max_read])?;

    if cfg.do_whiten {
        let n = cfg.whitening_chunks.max(1) as u64;
        let size = ((cfg.whitening_chunk_ms / 1000.0 * fs).round() as u64).clamp(1, total);
        let windows: Vec<HaloWindow> = (0..n)
            .map(|i| {
                let start = if n > 1 { i * (total - size) / (n - 1) } else { 0 };
                HaloWindow::around(i as usize, start..start + size, settle_l as u64, settle_r as u64, total)
            })
            .collect();
        let read = windows.iter().map(HaloWindow::read_len).max().unwrap_or(1);
        let workspace = PipelineWorkspace::<f32>::new(client.clone(), filtering.clone(), channels, read, fs).map_err(filter_error)?;
        let mut device = DeviceWindows::new(source, workspace, None);
        let mut moment = SecondMomentAccumulator::<f32>::new(client, channels);
        let mut done = 0u64;
        stages.report(STAGE_WHITENING, 0, n);
        device.for_each_while(&windows, |w, filtered| {
            moment.add(filtered, w.read_len(), w.valid_local.clone());
            done += 1;
            stages.report(STAGE_WHITENING, done, n);
            Ok(true)
        })?;
        let w = SpatialWhitening::zca_from_covariance::<f32>(client, &moment.finish(), channels, cfg.whitening_epsilon);
        pipeline_stages.push(PipelineStage::SpatialWhitening(w));
    }
    let pipeline = Pipeline::with_stages(pipeline_stages);
    let workspace = PipelineWorkspace::<f32>::new(client.clone(), pipeline.clone(), channels, max_read, fs).map_err(filter_error)?;
    let mut device = DeviceWindows::new(source, workspace, None);

    // 2. Phase 1
    let (segments, detect1) = match cfg.scheme {
        Scheme::One => (
            vec![0..total],
            DetectOptions {
                threshold: cfg.detect_threshold,
                sign: cfg.detect_sign,
                time_radius_ms: cfg.detect_time_radius_ms,
                channel_radius_um: cfg.scheme1_detect_channel_radius_um,
            },
        ),
        Scheme::Two => (
            training_segments(total, fs, cfg.training_duration_sec, cfg.training_sampling),
            DetectOptions {
                threshold: cfg.phase1_detect_threshold,
                sign: cfg.detect_sign,
                time_radius_ms: cfg.phase1_detect_time_radius_ms,
                channel_radius_um: cfg.phase1_detect_channel_radius_um,
            },
        ),
    };
    let training = segment_windows(&segments, chunk, halos, total);
    let windows: Vec<HaloWindow> = training.iter().map(|(w, _)| w.clone()).collect();
    let detector = Detector::new(client, &positions, detect1);
    let mut events = Events { samples: Vec::new(), segments: Vec::new(), snippets: MaskedSnippets::new(&positions, cfg.snippet_mask_radius_um, t1, t2) };
    let (mut values, mut det_channels) = (Vec::new(), Vec::new());
    let mut done = 0u64;
    let n_windows = windows.len() as u64;
    stages.report(STAGE_PHASE1, 0, n_windows);
    device.for_each_while(&windows, |w, prepared| {
        let si = training[w.index].1;
        let seg = &segments[si];
        let valid = local(&(seg.start + t1 as u64..seg.end.saturating_sub(t2 as u64)), w);
        let det = detector.detect(client, prepared, w.read_len(), valid, w.valid_local.clone(), w.read_global.start, fs);
        let local_samples: Vec<u32> = det.samples.iter().map(|&s| (s - w.read_global.start) as u32).collect();
        events.snippets.extract(client, prepared, w.read_len(), &local_samples, &det.channels);
        events.segments.extend(std::iter::repeat_n(si as u32, det.samples.len()));
        events.samples.extend(&det.samples);
        values.extend(&det.values);
        det_channels.extend(&det.channels);
        done += 1;
        stages.report(STAGE_PHASE1, done, n_windows);
        Ok(true)
    })?;
    let phase1: Scheme1Output =
        cluster_events_with_progress(client, events, &segments, &cfg.clustering(), &mut |done, total| stages.report(STAGE_CLUSTERING, done, total));
    let mask_rows = MaskedSnippets::new(&positions, cfg.snippet_mask_radius_um, t1, t2).mask;
    let unit_channels = |peak: usize| -> Vec<usize> { mask_rows.of(peak).collect() };

    let width = t1 + t2;
    let result = |n_units: usize, samples, units, amps, chans, templates, peaks: Vec<usize>, phase1_spikes| Mountainsort5Result {
        sample_rate_hz: fs,
        total_samples: total,
        n_units,
        spike_samples: samples,
        spike_units: units,
        spike_amplitudes: amps,
        spike_channels: chans,
        templates,
        template_channels: peaks.iter().map(|&p| unit_channels(p)).collect(),
        peak_channels: peaks,
        width,
        channels,
        positions: positions.clone(),
        phase1_spikes,
        pipeline: pipeline.clone(),
    };
    if cfg.scheme == Scheme::One {
        let amps = phase1.events.iter().map(|&e| values[e]).collect();
        let chans = phase1.events.iter().map(|&e| det_channels[e]).collect();
        let units = phase1.labels.iter().map(|&l| l - 1).collect();
        let n = phase1.samples.len();
        return Ok(result(phase1.k, phase1.samples, units, amps, chans, phase1.templates, phase1.peak_channels, n));
    }

    // 3. Scheme 2: unit and noise snippets over the training stretch
    let k = phase1.k;
    let r = cfg.snippet_mask_radius_um;
    let near = |a: usize, b: usize, limit: Option<f32>| {
        limit.is_none_or(|l| (positions[a][0] - positions[b][0]).hypot(positions[a][1] - positions[b][1]) <= l)
    };
    let rows_within = |limit: Option<f32>| -> Vec<Vec<u32>> {
        phase1.peak_channels.iter().map(|&p| (0..channels).filter(|&c| near(p, c, limit)).map(|c| c as u32).collect()).collect()
    };
    let mut all = MaskedSnippets::with_rows(&rows_within(r), channels, t1, t2);
    let mut sub = MaskedSnippets::with_rows(&rows_within(r.map(|x| 2.0 * x)), channels, t1, t2);
    let mut noise = MaskedSnippets::new(&positions, None, t1, t2);
    let mut members: Vec<Vec<usize>> = vec![Vec::new(); k];
    for (i, &l) in phase1.labels.iter().enumerate() {
        members[l as usize - 1].push(i);
    }
    let mut selected = vec![false; phase1.samples.len()];
    for list in &members {
        for j in subsample_indices(list.len(), cfg.max_num_snippets_per_training_batch) {
            selected[list[j]] = true;
        }
    }
    // Noise times in the concatenated training stretch, mapped to global samples
    let n_train: u64 = segments.iter().map(|s| s.end - s.start).sum();
    let noise_global: Vec<u64> = noise_times(n_train as usize, t1, t2, cfg.max_num_snippets_per_training_batch)
        .into_iter()
        .map(|c| {
            let mut c = c as u64;
            for s in &segments {
                let len = s.end - s.start;
                if c < len {
                    return s.start + c;
                }
                c -= len;
            }
            segments.last().map_or(0, |s| s.end - 1)
        })
        .collect();
    let mut done = 0u64;
    stages.report(STAGE_TRAINING, 0, n_windows);
    let (mut next_spike, mut next_noise) = (0usize, 0usize);
    device.for_each_while(&windows, |w, prepared| {
        let owned = |s: u64| w.valid_global.contains(&s);
        let snippet_ok = |s: u64| s >= w.read_global.start + t1 as u64 && s + t2 as u64 <= w.read_global.end;
        let (mut a_s, mut a_u, mut b_s, mut b_u) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        while next_spike < phase1.samples.len() && phase1.samples[next_spike] < w.valid_global.end {
            let s = phase1.samples[next_spike];
            if owned(s) && snippet_ok(s) {
                let (ls, u) = ((s - w.read_global.start) as u32, phase1.labels[next_spike] - 1);
                a_s.push(ls);
                a_u.push(u);
                if selected[next_spike] {
                    b_s.push(ls);
                    b_u.push(u);
                }
            }
            next_spike += 1;
        }
        all.extract(client, prepared, w.read_len(), &a_s, &a_u);
        sub.extract(client, prepared, w.read_len(), &b_s, &b_u);
        let mut n_s = Vec::new();
        while next_noise < noise_global.len() && noise_global[next_noise] < w.valid_global.end {
            let s = noise_global[next_noise];
            if owned(s) && snippet_ok(s) {
                n_s.push((s - w.read_global.start) as u32);
            }
            next_noise += 1;
        }
        let zeros = vec![0u32; n_s.len()];
        noise.extract(client, prepared, w.read_len(), &n_s, &zeros);
        done += 1;
        stages.report(STAGE_TRAINING, done, n_windows);
        Ok(true)
    })?;
    let unit_labels: Vec<u32> = all.event_channels.iter().map(|&u| u + 1).collect();
    let templates = median_templates(&all, &unit_labels, k);
    drop(all);
    let classifiers = Classifiers::fit(client, &mask_rows, &noise, &templates, k, &sub, &cfg.classifier());
    drop(sub);

    // Phase 2: every window detected and classified
    let detector = Detector::new(
        client,
        &positions,
        DetectOptions {
            threshold: cfg.detect_threshold,
            sign: cfg.detect_sign,
            time_radius_ms: cfg.detect_time_radius_ms,
            channel_radius_um: cfg.detect_channel_radius_um,
        },
    );
    let tol = radius(cfg.detect_time_radius_ms) as i64;
    let snippet_mask = MaskedSnippets::new(&positions, cfg.snippet_mask_radius_um, t1, t2);
    let schedule = dsp_core::ChunkSchedule::full_recording(total, chunk, halos.0, halos.1);
    let (mut samples, mut units, mut amps, mut chans) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let windows = schedule.windows();
    let mut done = 0u64;
    stages.report(STAGE_CLASSIFYING, 0, windows.len() as u64);
    device.for_each_while(windows, |w, prepared| {
        let n = w.read_len();
        let valid = local(&(t1 as u64..total - t2 as u64), w);
        let emit = t1..n.saturating_sub(t2);
        let det = detector.detect(client, prepared, n, valid, emit, w.read_global.start, fs);
        if !det.samples.is_empty() {
            let local_samples: Vec<u32> = det.samples.iter().map(|&s| (s - w.read_global.start) as u32).collect();
            let snippets = snippet_mask.gather(client, prepared, n, &local_samples, &det.channels);
            let (labels, offsets) = classifiers.classify(client, &snippets, &det.channels);
            let mut found: Vec<(i64, u32, usize)> = (0..labels.len())
                .filter(|&i| labels[i] > 0)
                .map(|i| (det.samples[i] as i64 - offsets[i] as i64, labels[i], i))
                .collect();
            found.sort_by_key(|f| (f.0, f.2));
            let times: Vec<i64> = found.iter().map(|f| f.0).collect();
            let labs: Vec<u32> = found.iter().map(|f| f.1).collect();
            for j in remove_duplicate_events(&times, &labs, tol) {
                let (t, l, i) = found[j];
                if t >= w.valid_global.start as i64 && t < w.valid_global.end as i64 {
                    samples.push(t as u64);
                    units.push(l - 1);
                    amps.push(det.values[i]);
                    chans.push(det.channels[i]);
                }
            }
        }
        done += 1;
        stages.report(STAGE_CLASSIFYING, done, windows.len() as u64);
        Ok(true)
    })?;
    let phase1_spikes = phase1.samples.len();
    Ok(result(k, samples, units, amps, chans, templates, phase1.peak_channels, phase1_spikes))
}

#[cfg(test)]
mod tests {
    use super::*;
    use dsp_core::compute::ComputeTarget;
    use dsp_io::neuro::synthetic::{SyntheticParams, SyntheticRecording};

    /// Best accuracy of each ground-truth unit against the sorted units (spikes within `tol`
    /// samples match): `matches / (n_true + n_sorted − matches)`.
    fn accuracies(rec: &SyntheticRecording, out: &Mountainsort5Result, tol: u64) -> Vec<f64> {
        let total = rec.info().samples;
        (0..rec.unit_count())
            .map(|g| {
                let truth = rec.spike_times(g, 0..total);
                (0..out.n_units)
                    .map(|u| {
                        let mine: Vec<u64> = (0..out.spike_samples.len()).filter(|&i| out.spike_units[i] == u as u32).map(|i| out.spike_samples[i]).collect();
                        let mut j = 0;
                        let mut hits = 0usize;
                        for &t in &truth {
                            while j < mine.len() && mine[j] + tol < t {
                                j += 1;
                            }
                            if j < mine.len() && mine[j] <= t + tol {
                                hits += 1;
                                j += 1;
                            }
                        }
                        hits as f64 / (truth.len() + mine.len() - hits) as f64
                    })
                    .fold(0.0, f64::max)
            })
            .collect()
    }

    /// Both schemes recover well-separated synthetic units (scheme 2 all of them).
    #[test]
    #[cfg_attr(debug_assertions, ignore = "minutes unoptimised: run with --profile validate")]
    fn sorts_synthetic_units() {
        let Ok(target) = ComputeTarget::from_env() else { return };
        let client = target.client().expect("client");
        let params = SyntheticParams { channels: 32, duration_sec: 30.0, line_noise_uv: 0.0, units: 6, drift_channels: 0.0, ..Default::default() };
        let rec = SyntheticRecording::new(params).expect("synthetic recording");
        let ids: Vec<usize> = (0..32).collect();
        let pos: Vec<[f32; 2]> = (0..32).map(|c| [0.0, 25.0 * c as f32]).collect();
        let probe = SensorLayout::from_channel_arrays("linear", &ids, &pos, &[0; 32]).expect("probe");
        for (scheme, training) in [(Scheme::One, None), (Scheme::Two, Some(10.0))] {
            let cfg = Mountainsort5Config { scheme, training_duration_sec: training, ..Default::default() };
            let start = std::time::Instant::now();
            let log = |e: &dsp_core::ProgressEvent<'_>| {
                if e.done == e.total {
                    eprintln!("  {:>7.2} s {} {}/{}", start.elapsed().as_secs_f64(), e.stage, e.done, e.total);
                }
            };
            let out = run(&client, &rec, &probe, &cfg, &log).expect("run");
            let acc = accuracies(&rec, &out, 30);
            eprintln!("{scheme:?}: {} units, {} spikes, phase 1 {}, accuracies {acc:.3?}", out.n_units, out.spike_samples.len(), out.phase1_spikes);
            // Scheme 1 splits some of these broad waveforms by detection jitter (their whitened
            // troughs are flat over ±1 sample); scheme 2's classifiers put the pieces back
            let (bar, units) = if scheme == Scheme::Two { (0.95, 6) } else { (0.9, 4) };
            assert!(acc.iter().filter(|&&a| a >= bar).count() >= units, "{scheme:?}: {acc:?}");
            assert_eq!(out.to_sorting_output(Some(probe.clone())).units.len(), out.n_units);
        }
    }

    /// Upstream's sampling: 10 s chunks, the last one shorter, spread to the end.
    #[test]
    fn training_chunks_follow_upstream() {
        let fs = 1000.0;
        assert_eq!(training_segments(100_000, fs, Some(300.0), TrainingSampling::Uniform), vec![0..100_000], "shorter than the stretch");
        assert_eq!(training_segments(100_000, fs, Some(25.0), TrainingSampling::Initial), vec![0..25_000]);
        // 25 s: chunks of 10, 10, 5 s; spacing int((100 000 − 25 000) / 2) = 37 500 samples
        assert_eq!(training_segments(100_000, fs, Some(25.0), TrainingSampling::Uniform), vec![0..10_000, 47_500..57_500, 95_000..100_000]);
        assert_eq!(training_segments(100_000, fs, None, TrainingSampling::Uniform), vec![0..100_000]);
    }
}
