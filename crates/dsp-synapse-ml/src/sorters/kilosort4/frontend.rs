//! The Kilosort4 front end over a whole recording, streamed in halo windows (dsp-core
//! `ChunkSchedule`, dsp-io `PrefetchReader`, dsp-base `PipelineWorkspace`): Kilosort4's batches are
//! the schedule's windows, so every pass sees the same batches upstream would.
//!
//! Passes (`nskip`: upstream's stride for the learning passes):
//! 1. **Whitening**: high-pass (+ common reference) every `nskip`-th window on the device, average
//!    the covariances of their interiors, fit local whitening (`whitening_range` nearest contacts).
//! 2. **Channel delays** (EMUsort): estimated on every `nskip`-th whitened window.
//! 3. **Clips** from every `nskip`-th preprocessed window, then universal templates (`wPCA`, `wTEMP`).
//! 4. **Detection** on every window (read ahead in the background), spikes kept in each window's
//!    own samples and reported with recording sample indices.
//!
//! Memory stays bounded by a few windows whatever the recording length.

use cubecl::prelude::{ComputeClient, Runtime};
use dsp_base::core::buffer;
use dsp_base::filter::{FilterBand, FilterSpec};
use dsp_base::linalg::covariance_of_host;
use dsp_base::pipeline::{Pipeline, PipelineStage, PipelineWorkspace};
use dsp_base::spatial::SpatialWhitening;
use dsp_core::{ChunkSchedule, DspError, DspResult, HaloWindow, RecordingSource};
use dsp_io::neuro::probe::SensorLayout;
use dsp_io::PrefetchReader;

use super::detect::{detect_universal, TemplateCentres, UniversalSpike};
use super::templates::{extract_clips, learn_universal_templates, ClipOptions, LearnOptions, UniversalTemplates};
use super::Kilosort4Config;
use crate::sorters::emusort::{apply_channel_delays, ChannelDelayEstimator, EmusortConfig};

/// Butterworth order of the high-pass, as upstream's `preprocessing.get_highpass_filter`
/// (third order, applied forward-backward). To verify against upstream.
pub const HIGHPASS_ORDER: usize = 3;

/// Added to covariance eigenvalues before whitening, as upstream's `whitening_from_covariance`.
/// To verify against upstream.
pub const WHITENING_EPSILON: f32 = 1e-6;

/// What the front end runs: Kilosort4's settings and the sorter-specific additions.
#[derive(Debug, Clone)]
pub struct FrontEndOptions {
    pub kilosort4: Kilosort4Config,
    pub clips: ClipOptions,
    pub learn: LearnOptions,
    /// EMUsort: remove channel delays up to this many samples.
    pub max_channel_delay: Option<usize>,
}

impl FrontEndOptions {
    /// Kilosort4 as published.
    pub fn kilosort4(config: &Kilosort4Config) -> Self {
        Self { kilosort4: config.clone(), clips: config.clip_options(), learn: config.learn_options(), max_channel_delay: None }
    }

    /// EMUsort at sample rate `fs` (Hz).
    pub fn emusort(config: &EmusortConfig, fs: f64) -> Self {
        Self {
            kilosort4: config.kilosort4.clone(),
            clips: config.clip_options(),
            learn: config.learn_options(),
            max_channel_delay: config.remove_channel_delays.then(|| config.max_delay_samples(fs)),
        }
    }
}

/// What the front end found.
#[derive(Debug, Clone)]
pub struct FrontEndResult {
    pub whitening: SpatialWhitening,
    /// `(delays, reference channel)` when channel delays were removed.
    pub channel_delays: Option<(Vec<isize>, usize)>,
    pub templates: UniversalTemplates,
    /// Detected spikes; `sample` is the recording sample.
    pub spikes: Vec<UniversalSpike>,
    /// `(left, right)` halo samples of each window.
    pub halos: (u64, u64),
    pub windows: usize,
}

fn filter_error(e: dsp_base::filter::FilterError) -> DspError {
    DspError::InvalidConfig(e.to_string())
}

/// Reads the padded samples of `window` (all channels, channel-major).
fn read_window(source: &dyn RecordingSource, window: &HaloWindow) -> DspResult<Vec<f32>> {
    let channels: Vec<usize> = (0..source.info().channel_count()).collect();
    let mut data = vec![0.0f32; channels.len() * window.read_len()];
    source.read(&channels, window.read_global.clone(), &mut data)?;
    Ok(data)
}

/// The `[channels, valid]` interior of a `[channels, read_len]` window.
fn interior(data: &[f32], window: &HaloWindow) -> Vec<f32> {
    data.chunks_exact(window.read_len()).flat_map(|row| row[window.valid_local.clone()].iter().copied()).collect()
}

/// The window with a symmetric margin of `pad` samples around its interior, and `pad` (the smaller
/// of its two halos), for host stages that assume equal padding on both sides.
fn symmetric(data: &[f32], window: &HaloWindow) -> (Vec<f32>, usize, usize) {
    let pad = window.valid_local.start.min(window.read_len() - window.valid_local.end);
    let range = window.valid_local.start - pad..window.valid_local.end + pad;
    let samples = range.len();
    let cropped = data.chunks_exact(window.read_len()).flat_map(|row| row[range.clone()].iter().copied()).collect();
    (cropped, samples, pad)
}

/// Runs the front end of `options` on all of `source`, with probe geometry `probe`.
pub fn run_front_end<R: Runtime>(client: &ComputeClient<R>, source: &dyn RecordingSource, probe: &SensorLayout, options: &FrontEndOptions) -> DspResult<FrontEndResult> {
    let info = source.info();
    let (channels, total, fs) = (info.channel_count(), info.samples, info.sample_rate_hz());
    let ks = &options.kilosort4;
    if probe.total_channels() != channels {
        return Err(DspError::InvalidConfig(format!("probe has {} sites, recording {channels} channels", probe.total_channels())));
    }

    // Preprocessing before whitening; halos cover its settling, a waveform (nt) and the delays
    let mut stages = vec![PipelineStage::Filter(FilterSpec::butterworth(HIGHPASS_ORDER, FilterBand::Highpass(ks.highpass_cutoff_hz)))];
    if ks.do_car {
        stages.push(PipelineStage::CommonAverageReference);
    }
    let filtering = Pipeline::with_stages(stages.clone());
    let (settle_left, settle_right) = filtering.settling(fs).map_err(filter_error)?;
    let margin = (ks.nt + options.max_channel_delay.unwrap_or(0)) as u64;
    let halos = (settle_left as u64 + margin, settle_right as u64 + margin);
    let schedule = ChunkSchedule::full_recording(total, ks.batch_size as u64, halos.0, halos.1);
    let max_window = schedule.max_read_samples();
    let learning: Vec<HaloWindow> = schedule.windows().iter().step_by(ks.nskip.max(1)).cloned().collect();

    // 1. Whitening from the average covariance of the learning windows' interiors
    let mut workspace = PipelineWorkspace::<R, f32>::new(client.clone(), filtering, channels, max_window, fs).map_err(filter_error)?;
    let mut covariance = vec![0.0f64; channels * channels];
    let mut counted = 0u64;
    for window in &learning {
        let filtered = workspace.process_chunk_in_vram(&read_window(source, window)?, window.read_len());
        let inner = interior(&buffer::download::<R, f32>(client, filtered)[..channels * window.read_len()], window);
        let (cov, _) = covariance_of_host::<R, f32>(client, &inner, channels, window.valid_len());
        let n = window.valid_len() as u64;
        for (acc, c) in covariance.iter_mut().zip(buffer::download::<R, f32>(client, cov)) {
            *acc += c as f64 * n as f64;
        }
        counted += n;
    }
    covariance.iter_mut().for_each(|c| *c /= counted.max(1) as f64);
    let positions: Vec<[f32; 2]> = probe.sites().iter().map(|s| [s.position.x_um, s.position.y_um]).collect();
    let k = ks.whitening_range.min(channels);
    let whitening = SpatialWhitening::local_knn_from_covariance::<R, f32>(client, &covariance, channels, &positions, k, WHITENING_EPSILON);

    stages.push(PipelineStage::SpatialWhitening(whitening.clone()));
    let mut workspace = PipelineWorkspace::<R, f32>::new(client.clone(), Pipeline::with_stages(stages), channels, max_window, fs).map_err(filter_error)?;
    // On the device; `preprocess` brings a window back for the host stages (delays, clips)
    let mut preprocess_on_device = |data: &[f32], window: &HaloWindow| workspace.process_chunk_in_vram(data, window.read_len());
    let download = |handle, window: &HaloWindow| buffer::download::<R, f32>(client, handle)[..channels * window.read_len()].to_vec();

    // 2. Channel delays (EMUsort)
    let channel_delays = match options.max_channel_delay {
        Some(max_lag) => {
            let mut estimator = ChannelDelayEstimator::new(channels, max_lag);
            for window in &learning {
                let x = download(preprocess_on_device(&read_window(source, window)?, window), window);
                let (x, samples, pad) = symmetric(&x, window);
                if pad >= max_lag && samples > 2 * pad {
                    estimator.add_batch(&x, samples, pad);
                }
            }
            Some(estimator.delays())
        }
        None => None,
    };
    let align = |x: &mut [f32], window: &HaloWindow| {
        if let Some((delays, _)) = &channel_delays {
            apply_channel_delays(x, window.read_len(), delays);
        }
    };

    // 3. Clips and universal templates
    let mut clips = Vec::new();
    for window in &learning {
        let mut x = download(preprocess_on_device(&read_window(source, window)?, window), window);
        align(&mut x, window);
        extract_clips(&x, channels, window.read_len(), &options.clips, &mut clips);
    }
    let templates = learn_universal_templates(client, &clips, ks.nt, &options.learn)?;

    // 4. Detection on every window, read ahead in the background; the window stays on the device
    //    unless delays (a host shift) must be removed
    let centres = TemplateCentres::new(probe, &ks.centres)?;
    let nt0min = ks.nt0min();
    let mut spikes = Vec::new();
    PrefetchReader::new(source, schedule.clone()).for_each_window(|window, data| {
        let mut handle = preprocess_on_device(data, window);
        if channel_delays.is_some() {
            let mut x = download(handle, window);
            align(&mut x, window);
            handle = buffer::upload(client, &x);
        }
        for mut spike in detect_universal(client, &handle, channels, window.read_len(), &centres, &templates, ks.th_universal, nt0min)? {
            if window.is_interior_local(spike.sample) {
                spike.sample = window.to_global_sample(spike.sample) as usize;
                spikes.push(spike);
            }
        }
        Ok(())
    })?;

    Ok(FrontEndResult { whitening, channel_delays, templates, spikes, halos, windows: schedule.len() })
}
