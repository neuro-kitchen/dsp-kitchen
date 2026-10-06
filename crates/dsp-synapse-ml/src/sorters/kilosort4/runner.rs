//! Kilosort4 runner: prefetching, out-of-core VRAM window streaming, and spike detection.
//!
//! Workflow:
//! 1. **Preprocessing (`Pipeline`)**: High-pass Butterworth filtering (300 Hz) + Common Average Referencing
//!    (CAR) + local k-NN spatial whitening from data covariance. Returns a reusable `Pipeline` asset.
//! 2. **Universal Templates**: Isolated peak clip extraction on learning batches, SVD ($w\text{PCA}$)
//!    and k-means ($w\text{TEMP}$).
//! 3. **Universal Detection**: Halo-windowed streaming on device; preprocessed chunks remain resident
//!    in VRAM without host readback, producing detected spikes convertible to [`SortingOutput`].

use cubecl::prelude::{ComputeClient, Runtime};
use dsp_base::core::buffer;
use dsp_base::filter::{FilterBand, FilterSpec};
use dsp_base::linalg::covariance_of_host;
use dsp_base::pipeline::{Pipeline, PipelineStage, PipelineWorkspace};
use dsp_base::spatial::SpatialWhitening;
use dsp_core::{ChunkSchedule, DspError, DspResult, HaloWindow, RecordingSource};
use dsp_io::neuro::probe::SensorLayout;
use dsp_orchestrate::{remap_event, WindowLoader};
use dsp_synapse::core::{SortedUnit, SortingOutput};

use super::detect::{detect_universal, TemplateCentres, UniversalSpike};
use super::templates::{extract_clips, learn_universal_templates, UniversalTemplates, MAX_CLIPS};
use super::Kilosort4Config;

/// Butterworth order of the high-pass filter, matching upstream Kilosort4.
pub const HIGHPASS_ORDER: usize = 3;

/// Regularization added to covariance eigenvalues before local whitening.
pub const WHITENING_EPSILON: f32 = 1e-6;

fn filter_error(e: dsp_base::filter::FilterError) -> DspError {
    DspError::InvalidConfig(e.to_string())
}

/// Dynamic learning stride: ensures short recordings sample at least ~5 batches across the span
/// rather than only window 0 when total_windows < nskip.
fn learning_stride(schedule_len: usize, nskip: usize) -> usize {
    if schedule_len <= nskip {
        (schedule_len / 5).max(1)
    } else {
        nskip.max(1)
    }
}

fn load_universal_templates() -> DspResult<UniversalTemplates> {
    #[cfg(feature = "hub")]
    {
        let (_, path) = crate::runtime::pull_model(super::KILOSORT4_WTEMP_MODEL_ID)?;
        return UniversalTemplates::from_npz(&path);
    }
    #[allow(unreachable_code)]
    {
        for candidate in [
            std::path::Path::new("data/kilosort4/wTEMP.npz"),
            std::path::Path::new("../data/kilosort4/wTEMP.npz"),
            std::path::Path::new("../../data/kilosort4/wTEMP.npz"),
        ] {
            if candidate.exists() {
                return UniversalTemplates::from_npz(candidate);
            }
        }
        Err(DspError::InvalidConfig(
            "Universal templates from data is disabled, but 'hub' feature is off and data/kilosort4/wTEMP.npz was not found".into(),
        ))
    }
}

/// Fits Kilosort4 preprocessing (high-pass Butterworth, optional CAR, local whitening)
/// over the recording, returning a composable [`Pipeline`] asset and the [`SpatialWhitening`] matrix.
pub fn fit_kilosort4_preprocessing<R: Runtime>(
    client: &ComputeClient<R>,
    source: &dyn RecordingSource,
    probe: &SensorLayout,
    ks: &Kilosort4Config,
) -> DspResult<(Pipeline, SpatialWhitening)> {
    let info = source.info();
    let (channels, total, fs) = (info.channel_count(), info.samples, info.sample_rate_hz());
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
    stages.push(PipelineStage::Filter(FilterSpec::butterworth(
        HIGHPASS_ORDER,
        FilterBand::Highpass(ks.highpass_cutoff_hz),
    )));
    let filtering = Pipeline::with_stages(stages.clone());
    let (settle_left, settle_right) = filtering.settling(fs).map_err(filter_error)?;
    let margin = ks.nt as u64;
    let halos = (settle_left as u64 + margin, settle_right as u64 + margin);
    let schedule = ChunkSchedule::full_recording(total, ks.batch_size as u64, halos.0, halos.1);
    let max_window = schedule.max_read_samples();
    let stride = learning_stride(schedule.len(), ks.nskip);
    let learning: Vec<HaloWindow> = schedule.windows().iter().step_by(stride).cloned().collect();

    // 1. Whitening from average covariance of learning windows' interiors
    let loader = WindowLoader::new(source);
    let mut workspace = PipelineWorkspace::<R, f32>::new(client.clone(), filtering, channels, max_window, fs).map_err(filter_error)?;
    let mut covariance = vec![0.0f64; channels * channels];
    let mut counted = 0u64;
    loader.for_windows(&learning, |buf| {
        let filtered = workspace.process_chunk_in_vram(buf.as_slice(), buf.read_len());
        let inner = workspace.download_interior(filtered, buf.window());
        let (cov, _) = covariance_of_host::<R, f32>(client, &inner, channels, buf.valid_len());
        let n = buf.valid_len() as u64;
        for (acc, c) in covariance.iter_mut().zip(buffer::download::<R, f32>(client, cov)) {
            *acc += c as f64 * n as f64;
        }
        counted += n;
        Ok(())
    })?;
    covariance.iter_mut().for_each(|c| *c /= counted.max(1) as f64);
    let positions: Vec<[f32; 2]> = probe.sites().iter().map(|s| [s.position.x_um, s.position.y_um]).collect();
    let k = ks.whitening_range.min(channels);
    let whitening = SpatialWhitening::local_knn_from_covariance::<R, f32>(client, &covariance, channels, &positions, k, WHITENING_EPSILON);

    stages.push(PipelineStage::SpatialWhitening(whitening.clone()));
    Ok((Pipeline::with_stages(stages), whitening))
}

/// Results produced by a Kilosort4 run.
#[derive(Debug, Clone)]
pub struct Kilosort4Result {
    pub whitening: SpatialWhitening,
    pub templates: UniversalTemplates,
    pub spikes: Vec<UniversalSpike>,
    pub halos: (u64, u64),
    pub windows: usize,
    pub preprocessing: Pipeline,
}

impl Kilosort4Result {
    /// Converts detected universal spikes into canonical [`SortingOutput`] aligned with SpikeInterface / Phy standards.
    pub fn to_sorting_output(
        &self,
        probe: Option<SensorLayout>,
        sample_rate_hz: f64,
        total_samples: u64,
    ) -> SortingOutput {
        let n_templates = self.templates.n_templates;
        let mut per_template_spikes: Vec<Vec<u64>> = vec![Vec::new(); n_templates];
        let mut per_template_amps: Vec<Vec<f32>> = vec![Vec::new(); n_templates];
        let mut per_template_locs: Vec<Vec<[f32; 3]>> = vec![Vec::new(); n_templates];

        for spike in &self.spikes {
            if spike.template < n_templates {
                per_template_spikes[spike.template].push(spike.sample as u64);
                per_template_amps[spike.template].push(spike.amplitude);
                per_template_locs[spike.template].push([0.0, spike.y_um, 0.0]);
            }
        }

        let mut units = Vec::new();
        for (unit_id, ((samples, amps), locs)) in per_template_spikes
            .into_iter()
            .zip(per_template_amps)
            .zip(per_template_locs)
            .enumerate()
        {
            if samples.is_empty() {
                continue;
            }
            let unit = SortedUnit::from_spikes(
                unit_id,
                0,
                samples,
                amps,
                locs,
                None,
                sample_rate_hz,
                total_samples,
                1.0,
            );
            units.push(unit);
        }

        SortingOutput::new("kilosort4", sample_rate_hz, total_samples, probe, units, None)
    }
}

/// Kilosort4 runner orchestrating the end-to-end preprocessing, template learning, and detection pipeline.
#[derive(Debug, Clone)]
pub struct Kilosort4Runner {
    pub config: Kilosort4Config,
}

impl Kilosort4Runner {
    pub fn new(config: Kilosort4Config) -> Self {
        Self { config }
    }

    /// Runs Kilosort4 over `source` on `client`'s device runtime.
    pub fn run<R: Runtime>(
        &self,
        client: &ComputeClient<R>,
        source: &dyn RecordingSource,
        probe: &SensorLayout,
    ) -> DspResult<Kilosort4Result> {
        let info = source.info();
        let (channels, total, fs) = (info.channel_count(), info.samples, info.sample_rate_hz());
        let ks = &self.config;

        let (preprocessing, whitening) = fit_kilosort4_preprocessing(client, source, probe, ks)?;

        let (settle_left, settle_right) = preprocessing.settling(fs).map_err(filter_error)?;
        let margin = ks.nt as u64;
        let halos = (settle_left as u64 + margin, settle_right as u64 + margin);
        let schedule = ChunkSchedule::full_recording(total, ks.batch_size as u64, halos.0, halos.1);
        let max_window = schedule.max_read_samples();
        let stride = learning_stride(schedule.len(), ks.nskip);
        let learning: Vec<HaloWindow> = schedule.windows().iter().step_by(stride).cloned().collect();

        let loader = WindowLoader::new(source);
        let mut workspace = PipelineWorkspace::<R, f32>::new(client.clone(), preprocessing.clone(), channels, max_window, fs).map_err(filter_error)?;

        // Universal templates (wPCA / wTEMP)
        let min_clips_needed = ks.n_templates.max(ks.n_pcs);
        let templates = if ks.templates_from_data {
            let mut clips = Vec::new();
            loader.for_windows(&learning, |buf| {
                if clips.len() / ks.nt >= MAX_CLIPS {
                    return Ok(());
                }
                let handle = workspace.process_chunk_in_vram(buf.as_slice(), buf.read_len());
                let whitened = buffer::download::<R, f32>(client, handle)[..channels * buf.read_len()].to_vec();
                extract_clips(&whitened, channels, buf.read_len(), &ks.clip_options(), &mut clips);
                Ok(())
            })?;

            // Fallback: if stride skipped too aggressively and we didn't get enough clips, scan remaining windows
            if clips.len() / ks.nt < min_clips_needed {
                for window in schedule.windows() {
                    if learning.iter().any(|w| w.index == window.index) {
                        continue;
                    }
                    let buf = loader.load_window(window)?;
                    let handle = workspace.process_chunk_in_vram(buf.as_slice(), buf.read_len());
                    let whitened = buffer::download::<R, f32>(client, handle)[..channels * buf.read_len()].to_vec();
                    extract_clips(&whitened, channels, buf.read_len(), &ks.clip_options(), &mut clips);
                    if clips.len() / ks.nt >= min_clips_needed {
                        break;
                    }
                }
            }

            learn_universal_templates(client, &clips, ks.nt, &ks.learn_options())?
        } else {
            load_universal_templates()?
        };

        // Streaming detection with zero host readback of filtered signals
        let centres = TemplateCentres::new(probe, &ks.centres)?;
        let nt0min = ks.nt0min();
        let mut spikes = Vec::new();
        loader.stream_schedule(&schedule, |window, data| {
            let handle = workspace.process_chunk_in_vram(data, window.read_len());
            for mut spike in detect_universal(client, &handle, channels, window.read_len(), &centres, &templates, ks.th_universal, nt0min)? {
                if let Some(global_sample) = remap_event(window, spike.sample) {
                    spike.sample = global_sample as usize;
                    spikes.push(spike);
                }
            }
            Ok(())
        })?;

        Ok(Kilosort4Result {
            whitening,
            templates,
            spikes,
            halos,
            windows: schedule.len(),
            preprocessing,
        })
    }
}



#[cfg(test)]
mod tests {
    use super::*;
    use dsp_core::compute::{ComputeTarget, ComputeTask};
    use dsp_io::neuro::synthetic::{SyntheticParams, SyntheticRecording};

    #[test]
    fn kilosort4_runner_executes_on_synthetic_recording() {
        let rec = SyntheticRecording::new(SyntheticParams {
            channels: 4,
            duration_sec: 1.0,
            sample_rate_hz: 30_000.0,
            ..Default::default()
        })
        .expect("synthetic recording");
        let probe = SensorLayout::from_channel_arrays(
            "4ch",
            &[0, 1, 2, 3],
            &[[0.0, 0.0], [0.0, 25.0], [0.0, 50.0], [0.0, 75.0]],
            &[0, 0, 0, 0],
        )
        .expect("probe layout");

        let mut config = Kilosort4Config::default();
        config.batch_size = 1000;
        config.nskip = 1;
        config.whitening_range = 4;
        config.th_single_ch = vec![4.0];

        struct Task<'a>(&'a dyn RecordingSource, &'a SensorLayout, &'a Kilosort4Config);
        impl ComputeTask for Task<'_> {
            type Output = DspResult<Kilosort4Result>;
            fn run<R: Runtime>(self, client: ComputeClient<R>) -> Self::Output {
                let runner = Kilosort4Runner::new(self.2.clone());
                runner.run(&client, self.0, self.1)
            }
        }

        if let Ok(target) = ComputeTarget::from_env() {
            let res = target
                .run(Task(&rec, &probe, &config))
                .expect("compute target should run")
                .expect("runner should succeed");
            assert!(res.windows > 0);
            let sorting = res.to_sorting_output(Some(probe.clone()), rec.info().sample_rate_hz(), rec.info().samples);
            assert_eq!(sorting.sorter_name, "kilosort4");
        }
    }
}
