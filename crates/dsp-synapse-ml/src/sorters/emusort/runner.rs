//! EMUsort runner: out-of-core streaming spike sorter for high-density motor unit action potentials.
//!
//! Reuses Kilosort4 preprocessing and universal template detection, adding:
//! 1. Inter-channel propagation delay estimation via cross-correlation ([`ChannelDelayEstimator`]).
//! 2. Circular delay alignment ([`apply_channel_delays`]) to align multi-channel MUAP waveforms.
//! 3. HDBSCAN outlier rejection during template learning.

use cubecl::prelude::{ComputeClient, Runtime};
use dsp_base::core::buffer;
use dsp_base::filter::{FilterBand, FilterSpec};
use dsp_base::linalg::covariance_of_host;
use dsp_base::pipeline::{Pipeline, PipelineStage, PipelineWorkspace};
use dsp_base::spatial::SpatialWhitening;
use dsp_core::{ChunkSchedule, DspError, DspResult, HaloWindow, RecordingSource};
use dsp_io::neuro::probe::SensorLayout;
use dsp_orchestrate::{remap_event, WindowBuffer, WindowLoader};
use dsp_synapse::core::{SortedUnit, SortingOutput};

use super::{apply_channel_delays, ChannelDelayEstimator, EmusortConfig};
use crate::sorters::kilosort4::detect::{detect_universal, TemplateCentres, UniversalSpike};
use crate::sorters::kilosort4::templates::{extract_clips, learn_universal_templates, UniversalTemplates, MAX_CLIPS};

/// Butterworth order of the high-pass filter, matching upstream Kilosort4 / EMUsort.
pub const HIGHPASS_ORDER: usize = 3;

/// Regularization added to covariance eigenvalues before local whitening.
pub const WHITENING_EPSILON: f32 = 1e-6;

fn filter_error(e: dsp_base::filter::FilterError) -> DspError {
    DspError::InvalidConfig(e.to_string())
}

fn learning_stride(schedule_len: usize, nskip: usize) -> usize {
    if schedule_len <= nskip {
        (schedule_len / 5).max(1)
    } else {
        nskip.max(1)
    }
}

/// Result of an EMUsort run.
#[derive(Debug, Clone)]
pub struct EmusortResult {
    pub whitening: SpatialWhitening,
    pub channel_delays: Option<(Vec<isize>, usize)>,
    pub templates: UniversalTemplates,
    pub spikes: Vec<UniversalSpike>,
    pub halos: (u64, u64),
    pub windows: usize,
    pub preprocessing: Pipeline,
}

impl EmusortResult {
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

        SortingOutput::new("emusort", sample_rate_hz, total_samples, probe, units, None)
    }
}

/// EMUsort runner orchestrating preprocessing, channel delay estimation, template learning, and detection.
#[derive(Debug, Clone)]
pub struct EmusortRunner {
    pub config: EmusortConfig,
}

impl EmusortRunner {
    pub fn new(config: EmusortConfig) -> Self {
        Self { config }
    }

    /// Runs EMUsort over `source` on `client`'s device runtime.
    pub fn run<R: Runtime>(
        &self,
        client: &ComputeClient<R>,
        source: &dyn RecordingSource,
        probe: &SensorLayout,
    ) -> DspResult<EmusortResult> {
        let info = source.info();
        let (channels, total, fs) = (info.channel_count(), info.samples, info.sample_rate_hz());
        let ks = &self.config.kilosort4;
        if probe.total_channels() != channels {
            return Err(DspError::InvalidConfig(format!(
                "probe has {} sites, recording {channels} channels",
                probe.total_channels()
            )));
        }

        let max_channel_delay = self
            .config
            .remove_channel_delays
            .then(|| self.config.max_delay_samples(fs));

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
        let margin = (ks.nt + max_channel_delay.unwrap_or(0)) as u64;
        let halos = (settle_left as u64 + margin, settle_right as u64 + margin);
        let schedule = ChunkSchedule::full_recording(total, ks.batch_size as u64, halos.0, halos.1);
        let max_window = schedule.max_read_samples();
        let stride = learning_stride(schedule.len(), ks.nskip);
        let learning: Vec<HaloWindow> = schedule.windows().iter().step_by(stride).cloned().collect();

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
        let preprocessing = Pipeline::with_stages(stages);
        let mut workspace = PipelineWorkspace::<R, f32>::new(client.clone(), preprocessing.clone(), channels, max_window, fs).map_err(filter_error)?;
        let mut preprocess_on_device = |data: &[f32], window: &HaloWindow| workspace.process_chunk_in_vram(data, window.read_len());
        let download = |handle, window: &HaloWindow| buffer::download::<R, f32>(client, handle)[..channels * window.read_len()].to_vec();

        let channel_delays = match max_channel_delay {
            Some(max_lag) => {
                let mut estimator = ChannelDelayEstimator::new(channels, max_lag);
                loader.for_windows(&learning, |buf| {
                    let x = download(preprocess_on_device(buf.as_slice(), buf.window()), buf.window());
                    let (x, samples, pad) = WindowBuffer::new(buf.window().clone(), channels, x).symmetric_pad();
                    if pad >= max_lag && samples > 2 * pad {
                        estimator.add_batch(&x, samples, pad);
                    }
                    Ok(())
                })?;
                Some(estimator.delays())
            }
            None => None,
        };
        let align = |x: &mut [f32], window: &HaloWindow| {
            if let Some((delays, _)) = &channel_delays {
                apply_channel_delays(x, window.read_len(), delays);
            }
        };

        let clip_opts = self.config.clip_options();
        let learn_opts = self.config.learn_options();
        let mut clips = Vec::new();
        loader.for_windows(&learning, |buf| {
            if clips.len() / ks.nt >= MAX_CLIPS {
                return Ok(());
            }
            let mut x = download(preprocess_on_device(buf.as_slice(), buf.window()), buf.window());
            align(&mut x, buf.window());
            extract_clips(&x, channels, buf.read_len(), &clip_opts, &mut clips);
            Ok(())
        })?;
        let min_clips_needed = learn_opts.n_templates.max(learn_opts.n_pcs);
        if clips.len() / ks.nt < min_clips_needed {
            for window in schedule.windows() {
                if learning.iter().any(|w| w.index == window.index) {
                    continue;
                }
                let buf = loader.load_window(window)?;
                let mut x = download(preprocess_on_device(buf.as_slice(), buf.window()), buf.window());
                align(&mut x, buf.window());
                extract_clips(&x, channels, buf.read_len(), &clip_opts, &mut clips);
                if clips.len() / ks.nt >= min_clips_needed {
                    break;
                }
            }
        }
        let templates = learn_universal_templates(client, &clips, ks.nt, &learn_opts)?;

        let centres = TemplateCentres::new(probe, &ks.centres)?;
        let nt0min = ks.nt0min();
        let mut spikes = Vec::new();
        loader.stream_schedule(&schedule, |window, data| {
            let handle = preprocess_on_device(data, window);
            let aligned_handle = if let Some((delays, _)) = &channel_delays {
                let shifts: Vec<u32> = delays
                    .iter()
                    .map(|&d| d.rem_euclid(window.read_len() as isize) as u32)
                    .collect();
                let s_handle = buffer::upload(client, &shifts);
                super::kernels::execute_apply_channel_delays(client, &handle, &s_handle, channels, window.read_len())
            } else {
                handle
            };
            for mut spike in detect_universal(client, &aligned_handle, channels, window.read_len(), &centres, &templates, ks.th_universal, nt0min)? {
                if let Some(global_sample) = remap_event(window, spike.sample) {
                    spike.sample = global_sample as usize;
                    spikes.push(spike);
                }
            }
            Ok(())
        })?;

        Ok(EmusortResult {
            whitening,
            channel_delays,
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
    fn emusort_runner_executes_on_synthetic_recording() {
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

        let mut config = EmusortConfig::default();
        config.kilosort4.batch_size = 1000;
        config.kilosort4.nskip = 1;
        config.kilosort4.whitening_range = 4;
        config.kilosort4.th_single_ch = vec![4.0];

        struct Task<'a>(&'a dyn RecordingSource, &'a SensorLayout, &'a EmusortConfig);
        impl ComputeTask for Task<'_> {
            type Output = DspResult<EmusortResult>;
            fn run<R: Runtime>(self, client: ComputeClient<R>) -> Self::Output {
                let runner = EmusortRunner::new(self.2.clone());
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
            assert_eq!(sorting.sorter_name, "emusort");
            assert!(res.channel_delays.is_some());
        }
    }
}
