//! Kilosort4-family runner (Kilosort4, and EMUsort through [`RunPlan::emusort`]): out-of-core
//! halo-window streaming over a recording, every window processed on the device.
//!
//! Three passes, as upstream (Kilosort4 v4.0.18 / EMUsort `a06bb60`):
//! 1. **Fit** (every `nskip`-th window, the last window excluded): high-pass Butterworth + optional
//!    common average reference; on the device, the whitening second moment (`X Xᵀ / n` per window,
//!    equal weight) and, for EMUsort, the channel-delay cross-correlation. Each downloads once, at
//!    the end. Local k-NN whitening completes the preprocessing [`Pipeline`].
//! 2. **Universal templates** (every `nskip`-th window, the last included; more windows if too few
//!    clips): preprocessing and delay removal on the device, clips on the host, then `wPCA` /
//!    `wTEMP`. Or predefined templates ([`RunPlan::templates`], the hub).
//! 3. **Detection** over every window: preprocessing, delay removal and universal-template
//!    detection on the device; only spikes come back.
//!
//! Spike samples are in the delay-aligned frame (the reference channel's time), as upstream.

use cubecl::prelude::{ComputeClient, Runtime};
use dsp_base::core::buffer;
use dsp_base::filter::{FilterBand, FilterSpec};
use dsp_base::linalg::SecondMomentAccumulator;
use dsp_base::pipeline::{Pipeline, PipelineStage, PipelineWorkspace};
use dsp_base::spatial::SpatialWhitening;
use dsp_core::progress::Stages;
use dsp_core::{ChunkSchedule, DspError, DspResult, HaloWindow, ProgressSink, RecordingSource, WindowLoader};
use dsp_io::neuro::probe::SensorLayout;
use dsp_synapse::core::{SortedUnit, SortingOutput};
use dsp_synapse::QualityCriteria;

use super::detect::{TemplateCentres, UniversalDetector, UniversalSpike};
use super::templates::{extract_clips, learn_universal_templates_with_progress, LearnOptions, UniversalTemplates, MAX_CLIPS};
use super::Kilosort4Config;
use crate::sorters::emusort::kernels::{ChannelAligner, ChannelDelayEstimator};

/// Butterworth order of the high-pass filter, matching upstream Kilosort4.
pub const HIGHPASS_ORDER: usize = 3;

/// Regularization added to covariance eigenvalues before local whitening.
pub const WHITENING_EPSILON: f32 = 1e-6;

/// Fewest windows the template-learning stride samples when the recording has `nskip` windows or
/// fewer.
pub const MIN_LEARNING_WINDOWS: usize = 5;

/// Progress stages of a run (reported in this order; a run reports only the ones it has), and
/// what each counts.
pub const STAGE_FIT: &str = "Fitting preprocessing";
pub const STAGE_CLIPS: &str = "Finding clips";
pub const STAGE_TEMPLATES: &str = "Learning templates";
pub const STAGE_DETECTION: &str = "Detecting spikes";
const WINDOWS: &str = "windows";
const STEPS: &str = "steps";

/// Sorter name of a Kilosort4 run in [`SortingOutput`].
pub const KILOSORT4_SORTER: &str = "kilosort4";

fn filter_error(e: dsp_base::filter::FilterError) -> DspError {
    DspError::InvalidConfig(e.to_string())
}

/// Every `nskip`-th window; short recordings (`nskip` windows or fewer) are sampled at
/// ~[`MIN_LEARNING_WINDOWS`] windows instead of only window 0.
fn learning_stride(windows: usize, nskip: usize) -> usize {
    if windows <= nskip {
        (windows / MIN_LEARNING_WINDOWS).max(1)
    } else {
        nskip.max(1)
    }
}

/// What a run does: Kilosort4's settings plus the variant's additions.
#[derive(Debug, Clone)]
pub struct RunPlan {
    /// Sorter name in [`SortingOutput`].
    pub sorter: &'static str,
    pub config: Kilosort4Config,
    pub learn: LearnOptions,
    /// Estimate and remove channel delays up to this many samples (EMUsort).
    pub max_channel_delay: Option<usize>,
    /// Predefined universal templates, used when `config.templates_from_data` is off (e.g.
    /// [`UniversalTemplates::from_npz`]); `None` pulls Kilosort4's `wTEMP.npz` from the hub.
    pub templates: Option<UniversalTemplates>,
    /// Preprocessing fitted by an earlier run on the same recording with the same fit settings
    /// (pass 1 is skipped), e.g. to compare template choices; checked with [`FitSettings`].
    pub fitted: Option<FittedPreprocessing>,
}

impl RunPlan {
    pub fn kilosort4(config: &Kilosort4Config) -> Self {
        Self {
            sorter: KILOSORT4_SORTER,
            learn: config.learn_options(),
            config: config.clone(),
            max_channel_delay: None,
            templates: None,
            fitted: None,
        }
    }
}

/// Channel delays (EMUsort): per-channel delay (samples) and the reference channel.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChannelDelays {
    pub delays: Vec<isize>,
    pub reference: usize,
}

/// Everything pass 1 depends on: a [`FittedPreprocessing`] is reused only when these match.
#[derive(Debug, Clone, PartialEq)]
pub struct FitSettings {
    pub channels: usize,
    pub total_samples: u64,
    pub sample_rate_hz: f64,
    pub do_car: bool,
    pub highpass_cutoff_hz: f64,
    pub nt: usize,
    pub batch_size: usize,
    pub nskip: usize,
    pub whitening_range: usize,
    pub max_channel_delay: Option<usize>,
}

impl FitSettings {
    pub fn of(plan: &RunPlan, info: &dsp_core::RecordingInfo) -> Self {
        let ks = &plan.config;
        Self {
            channels: info.channel_count(),
            total_samples: info.samples,
            sample_rate_hz: info.sample_rate_hz(),
            do_car: ks.do_car,
            highpass_cutoff_hz: ks.highpass_cutoff_hz,
            nt: ks.nt,
            batch_size: ks.batch_size,
            nskip: ks.nskip,
            whitening_range: ks.whitening_range,
            max_channel_delay: plan.max_channel_delay,
        }
    }
}

/// The fitted preprocessing of a run and the schedule it streams with.
#[derive(Debug, Clone)]
pub struct FittedPreprocessing {
    /// What it was fitted with.
    pub settings: FitSettings,
    /// Filtering (optional CAR, high-pass) then local whitening.
    pub pipeline: Pipeline,
    pub whitening: SpatialWhitening,
    pub channel_delays: Option<ChannelDelays>,
    pub schedule: ChunkSchedule,
    pub halos: (u64, u64),
}

/// Results of a Kilosort4-family run.
#[derive(Debug, Clone)]
pub struct Kilosort4Result {
    pub sorter: &'static str,
    /// Pass 1: preprocessing pipeline, whitening, channel delays, schedule and halos (reusable
    /// through [`RunPlan::fitted`]).
    pub fitted: FittedPreprocessing,
    pub templates: UniversalTemplates,
    pub spikes: Vec<UniversalSpike>,
    /// `(x, y)` µm of every template centre (`UniversalSpike::centre` indexes it).
    pub centre_positions: Vec<[f32; 2]>,
    /// Recording channel nearest to every template centre.
    pub centre_channels: Vec<usize>,
    pub sample_rate_hz: f64,
    pub total_samples: u64,
}

impl Kilosort4Result {
    /// One unit per universal template that detected spikes, in [`SortingOutput`] form. Spike
    /// locations are the detecting centre's `x` and the response-weighted `y` (µm); a unit's
    /// primary channel is the channel nearest to its most frequent centre. Amplitudes are in
    /// whitened σ, not µV, so no noise floor is given (SNR is left undefined).
    pub fn to_sorting_output(&self, probe: Option<SensorLayout>) -> SortingOutput {
        let n_templates = self.templates.n_templates;
        let n_centres = self.centre_positions.len();
        let mut samples: Vec<Vec<u64>> = vec![Vec::new(); n_templates];
        let mut amps: Vec<Vec<f32>> = vec![Vec::new(); n_templates];
        let mut locs: Vec<Vec<[f32; 3]>> = vec![Vec::new(); n_templates];
        let mut centre_counts = vec![vec![0usize; n_centres]; n_templates];
        for spike in self.spikes.iter().filter(|s| s.template < n_templates && s.centre < n_centres) {
            let t = spike.template;
            samples[t].push(spike.sample as u64);
            amps[t].push(spike.amplitude);
            locs[t].push([self.centre_positions[spike.centre][0], spike.y_um, 0.0]);
            centre_counts[t][spike.centre] += 1;
        }
        let units = samples
            .into_iter()
            .zip(amps)
            .zip(locs)
            .zip(centre_counts)
            .enumerate()
            .filter(|(_, (((s, _), _), _))| !s.is_empty())
            .map(|(unit_id, (((s, a), l), counts))| {
                let primary = counts.iter().enumerate().max_by_key(|&(_, n)| *n).map(|(c, _)| self.centre_channels[c]);
                let criteria = QualityCriteria::default();
                SortedUnit::from_spikes_with(unit_id, primary, s, a, l, None, self.sample_rate_hz, self.total_samples, None, criteria)
            })
            .collect();
        SortingOutput::new(self.sorter, self.sample_rate_hz, self.total_samples, probe, units, None)
    }
}

/// Pass 1: fits the preprocessing of `plan` (filtering, whitening, channel delays) over `source`,
/// reporting [`STAGE_FIT`] to `progress`.
pub fn fit_preprocessing<R: Runtime>(
    client: &ComputeClient<R>,
    source: &dyn RecordingSource,
    probe: &SensorLayout,
    plan: &RunPlan,
    progress: &dyn ProgressSink,
) -> DspResult<FittedPreprocessing> {
    fit_with_stages(client, source, probe, plan, &Stages::new(progress, &[(STAGE_FIT, WINDOWS)]))
}

fn fit_with_stages<R: Runtime>(
    client: &ComputeClient<R>,
    source: &dyn RecordingSource,
    probe: &SensorLayout,
    plan: &RunPlan,
    progress: &Stages<'_>,
) -> DspResult<FittedPreprocessing> {
    let info = source.info();
    let (channels, total, fs) = (info.channel_count(), info.samples, info.sample_rate_hz());
    let ks = &plan.config;
    if probe.total_channels() != channels {
        return Err(DspError::InvalidConfig(format!(
            "probe has {} sites, recording {channels} channels",
            probe.total_channels()
        )));
    }

    let mut stages = Vec::new();
    if ks.do_car {
        stages.push(PipelineStage::CommonAverageReference);
    }
    stages.push(PipelineStage::Filter(FilterSpec::butterworth(HIGHPASS_ORDER, FilterBand::Highpass(ks.highpass_cutoff_hz))));
    let filtering = Pipeline::with_stages(stages.clone());
    let (settle_left, settle_right) = filtering.settling(fs).map_err(filter_error)?;
    let margin = (ks.nt + plan.max_channel_delay.unwrap_or(0)) as u64;
    let halos = (settle_left as u64 + margin, settle_right as u64 + margin);
    let schedule = ChunkSchedule::full_recording(total, ks.batch_size as u64, halos.0, halos.1);

    // Upstream fits on `range(0, n_batches - 1, nskip)`: the last (partial) window is left out
    let fit_len = if schedule.len() > 1 { schedule.len() - 1 } else { schedule.len() };
    let fit_span = &schedule.windows()[..fit_len];
    let stride = learning_stride(fit_span.len(), ks.nskip);
    let fit_windows: Vec<HaloWindow> = fit_span.iter().step_by(stride).cloned().collect();

    let mut workspace =
        PipelineWorkspace::<R, f32>::new(client.clone(), filtering, channels, schedule.max_read_samples(), fs).map_err(filter_error)?;
    let mut second_moment = SecondMomentAccumulator::<R, f32>::new(client, channels);
    let mut delays = plan.max_channel_delay.map(|max_lag| ChannelDelayEstimator::new(client, channels, max_lag));
    let fit_total = fit_windows.len() as u64;
    let mut fit_done = 0u64;
    progress.report(STAGE_FIT, 0, fit_total);
    WindowLoader::new(source).stream(&fit_windows, |window, raw| {
        let filtered = workspace.process_chunk_in_vram(raw, window.read_len());
        second_moment.add(&filtered, window.read_len(), window.valid_local.clone());
        if let Some(est) = delays.as_mut() {
            est.add(&filtered, window.read_len(), window.valid_local.clone());
        }
        fit_done += 1;
        progress.report(STAGE_FIT, fit_done, fit_total);
        Ok(())
    })?;

    let positions: Vec<[f32; 2]> = probe.sites().iter().map(|s| [s.position.x_um, s.position.y_um]).collect();
    let k = ks.whitening_range.min(channels);
    let whitening =
        SpatialWhitening::local_knn_from_covariance::<R, f32>(client, &second_moment.finish(), channels, &positions, k, WHITENING_EPSILON);
    stages.push(PipelineStage::SpatialWhitening(whitening.clone()));
    let channel_delays = delays.map(|est| {
        let (delays, reference) = est.delays();
        ChannelDelays { delays, reference }
    });
    let settings = FitSettings::of(plan, info);
    Ok(FittedPreprocessing { settings, pipeline: Pipeline::with_stages(stages), whitening, channel_delays, schedule, halos })
}

/// Fits Kilosort4 preprocessing (high-pass Butterworth, optional CAR, local whitening) over the
/// recording: the composable [`Pipeline`] and its [`SpatialWhitening`].
pub fn fit_kilosort4_preprocessing<R: Runtime>(
    client: &ComputeClient<R>,
    source: &dyn RecordingSource,
    probe: &SensorLayout,
    ks: &Kilosort4Config,
) -> DspResult<(Pipeline, SpatialWhitening)> {
    let fitted = fit_preprocessing(client, source, probe, &RunPlan::kilosort4(ks), &dsp_core::NoProgress)?;
    Ok((fitted.pipeline, fitted.whitening))
}

fn predefined_templates(plan: &RunPlan) -> DspResult<UniversalTemplates> {
    if let Some(templates) = &plan.templates {
        return Ok(templates.clone());
    }
    #[cfg(feature = "hub")]
    {
        let (_, path) = crate::runtime::pull_model(super::KILOSORT4_WTEMP_MODEL_ID)?;
        UniversalTemplates::from_npz(&path)
    }
    #[cfg(not(feature = "hub"))]
    Err(DspError::InvalidConfig(
        "templates_from_data is off: pass predefined templates (RunPlan::templates, e.g. UniversalTemplates::from_npz) or enable the `hub` feature".into(),
    ))
}

/// Runs `plan` over `source` on `client`'s device, reporting its stages ([`STAGE_FIT`],
/// [`STAGE_CLIPS`], [`STAGE_TEMPLATES`], [`STAGE_DETECTION`], those the run has) to `progress`.
pub fn run_plan<R: Runtime>(
    client: &ComputeClient<R>,
    source: &dyn RecordingSource,
    probe: &SensorLayout,
    plan: &RunPlan,
    progress: &dyn ProgressSink,
) -> DspResult<Kilosort4Result> {
    let info = source.info();
    let (channels, total, fs) = (info.channel_count(), info.samples, info.sample_rate_hz());
    let ks = &plan.config;
    let mut names = Vec::new();
    if plan.fitted.is_none() {
        names.push((STAGE_FIT, WINDOWS));
    }
    if ks.templates_from_data {
        names.extend([(STAGE_CLIPS, WINDOWS), (STAGE_TEMPLATES, STEPS)]);
    }
    names.push((STAGE_DETECTION, WINDOWS));
    let stages = Stages::new(progress, &names);
    let fitted = match &plan.fitted {
        Some(fitted) if fitted.settings == FitSettings::of(plan, info) => fitted.clone(),
        Some(fitted) => {
            return Err(DspError::InvalidConfig(format!(
                "the given preprocessing was fitted with {:?}, this run needs {:?}",
                fitted.settings,
                FitSettings::of(plan, info)
            )))
        }
        None => fit_with_stages(client, source, probe, plan, &stages)?,
    };
    let schedule = &fitted.schedule;
    let max_window = schedule.max_read_samples();
    let loader = WindowLoader::new(source);
    let mut workspace =
        PipelineWorkspace::<R, f32>::new(client.clone(), fitted.pipeline.clone(), channels, max_window, fs).map_err(filter_error)?;
    let mut aligner = fitted.channel_delays.as_ref().map(|d| ChannelAligner::new(client, d.delays.clone(), max_window));
    // Preprocessed (and delay-aligned) window, on the device
    let mut prepare = |raw: &[f32], window: &HaloWindow| {
        let handle = workspace.process_chunk_in_vram(raw, window.read_len());
        match aligner.as_mut() {
            Some(aligner) => aligner.align(&handle, window.read_len()),
            None => handle,
        }
    };

    // 2. Universal templates
    let templates = if ks.templates_from_data {
        let clip_opts = ks.clip_options();
        let stride = learning_stride(schedule.len(), ks.nskip);
        let (mut learning, mut rest) = (Vec::new(), Vec::new());
        for (i, window) in schedule.windows().iter().enumerate() {
            if i % stride == 0 { &mut learning } else { &mut rest }.push(window.clone());
        }
        let mut clips = Vec::new();
        // Clips are found on the host: the only full-window download, on learning windows only
        let mut collect = |raw: &[f32], window: &HaloWindow, clips: &mut Vec<f32>| {
            // The prepared buffer is sized for the longest window: read only this window's part
            let x = buffer::download_prefix::<R, f32>(client, prepare(raw, window), channels * window.read_len());
            extract_clips(&x, channels, window.read_len(), &clip_opts, clips);
            clips.len() / ks.nt
        };
        let mut scanned = 0u64;
        let planned = learning.len() as u64;
        // The total the last report used: the stage is closed explicitly only if it stopped short
        let mut reported_total = planned;
        stages.report(STAGE_CLIPS, 0, planned);
        loader.stream_while(&learning, |window, raw| {
            let more = collect(raw, window, &mut clips) < MAX_CLIPS;
            scanned += 1;
            stages.report(STAGE_CLIPS, scanned, planned);
            Ok(more)
        })?;
        // Too few clips at this stride: scan the other windows until there are enough
        let needed = plan.learn.n_templates.max(plan.learn.n_pcs);
        if clips.len() / ks.nt < needed {
            let extended = planned + rest.len() as u64;
            reported_total = extended;
            loader.stream_while(&rest, |window, raw| {
                let more = collect(raw, window, &mut clips) < needed;
                scanned += 1;
                stages.report(STAGE_CLIPS, scanned, extended);
                Ok(more)
            })?;
        }
        // Stopped early (enough clips): end the stage at what was scanned
        if scanned < reported_total {
            stages.report(STAGE_CLIPS, scanned, scanned);
        }
        learn_universal_templates_with_progress(client, &clips, ks.nt, &plan.learn, &mut |done, total| stages.report(STAGE_TEMPLATES, done, total))?
    } else {
        predefined_templates(plan)?
    };
    if templates.nt != ks.nt {
        return Err(DspError::InvalidConfig(format!("universal templates have nt = {}, the config nt = {}", templates.nt, ks.nt)));
    }

    // 3. Detection: only spikes leave the device
    let centres = TemplateCentres::new(probe, &ks.centres)?;
    let mut detector = UniversalDetector::new(client, channels, max_window, &centres, &templates, ks.th_universal, ks.nt0min())?;
    let mut spikes = Vec::new();
    let (mut detected, windows) = (0u64, schedule.len() as u64);
    stages.report(STAGE_DETECTION, 0, windows);
    loader.stream(schedule.windows(), |window, raw| {
        let handle = prepare(raw, window);
        for mut spike in detector.detect(&handle, window.read_len())? {
            if let Some(global) = window.remap_event(spike.sample) {
                spike.sample = global as usize;
                spikes.push(spike);
            }
        }
        detected += 1;
        stages.report(STAGE_DETECTION, detected, windows);
        Ok(())
    })?;

    Ok(Kilosort4Result {
        sorter: plan.sorter,
        fitted,
        templates,
        spikes,
        centre_channels: (0..centres.n_centres()).map(|k| centres.ic[k] as usize).collect(),
        centre_positions: centres.positions,
        sample_rate_hz: fs,
        total_samples: total,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use dsp_core::compute::{ComputeTarget, ComputeTask};
    use dsp_io::neuro::synthetic::{SyntheticParams, SyntheticRecording};

    #[test]
    fn learning_stride_samples_short_recordings() {
        assert_eq!(learning_stride(100, 25), 25);
        assert_eq!(learning_stride(20, 25), 4);
        assert_eq!(learning_stride(3, 25), 1);
    }

    #[test]
    fn kilosort4_runs_on_synthetic_recording() {
        let rec = SyntheticRecording::new(SyntheticParams { channels: 4, duration_sec: 1.0, sample_rate_hz: 30_000.0, ..Default::default() })
            .expect("synthetic recording");
        let probe = SensorLayout::from_channel_arrays("4ch", &[0, 1, 2, 3], &[[0.0, 0.0], [0.0, 25.0], [0.0, 50.0], [0.0, 75.0]], &[0, 0, 0, 0])
            .expect("probe layout");
        let config = Kilosort4Config { batch_size: 1000, nskip: 1, whitening_range: 4, th_single_ch: vec![4.0], ..Default::default() };

        struct Task<'a>(&'a dyn RecordingSource, &'a SensorLayout, &'a Kilosort4Config);
        impl ComputeTask for Task<'_> {
            type Output = DspResult<Kilosort4Result>;
            fn run<R: Runtime>(self, client: ComputeClient<R>) -> Self::Output {
                crate::sorters::Kilosort4::new(self.2.clone()).run(&client, self.0, self.1)
            }
        }

        if let Ok(target) = ComputeTarget::from_env() {
            let res = target.run(Task(&rec, &probe, &config)).expect("compute target should run").expect("runner should succeed");
            assert!(res.fitted.schedule.len() > 0);
            assert!(res.fitted.channel_delays.is_none());
            let sorting = res.to_sorting_output(Some(probe.clone()));
            assert_eq!(sorting.sorter_name, KILOSORT4_SORTER);
        }
    }
}
