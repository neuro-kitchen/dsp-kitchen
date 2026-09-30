//! Out-of-core streaming spike sorting runner connecting `RecordingSource`, `PipelineWorkspace`,
//! boundary-safe `HaloWindow`s, and online Welford `TemplateAccumulator`s.

use cubecl::prelude::ComputeClient;
use cubecl::{CubeElement, Runtime};
use dsp_base::{ComputeTarget, ComputeTask};

use dsp_base::pipeline::{Pipeline, PipelineWorkspace};
use dsp_core::{ChunkSchedule, DspError, DspResult, ProbeLayout, RecordingSource};
use dsp_stream::PrefetchReader;

use crate::detection::{DeduplicatedSpike, StreamingDedup, estimate_noise_std};
use crate::extraction::extraction_margin;
use crate::kernels::{
    DetectionCarry, execute_detect_spikes_in_vram, execute_extract_sinc_in_vram,
    execute_reduce_templates_in_vram,
};
use crate::metrics::WaveformTemplate;
use crate::probe::precompute_knn_table;
use super::accumulator::TemplateAccumulator;
use super::config::StreamingSortConfig;

/// Result of running out-of-core threshold spike sorting over a recording.
#[derive(Debug, Clone)]
pub struct StreamingSortResult {
    /// Number of channels in the recording.
    pub channels: usize,
    /// Total samples processed across the recording.
    pub total_samples: u64,
    /// Sampling rate in Hz.
    pub sample_rate_hz: f64,
    /// Computed `(left_halo, right_halo)` in samples used during chunked streaming.
    pub halos: (u64, u64),
    /// Calibrated Quiroga MAD noise floor ($\sigma_n$ in $\mu\text{V}$) per channel.
    pub channel_sigmas_uv: Vec<f32>,
    /// Total raw threshold crossings across all channels before spatial deduplication.
    pub total_raw_crossings: u64,
    /// Total spatially deduplicated spikes across the recording.
    pub total_dedup_spikes: u64,
    /// Deduplicated spike count per primary channel.
    pub channel_spike_counts: Vec<u64>,
    /// Online Welford-accumulated waveform template (`mean` $\pm$ `std`) per primary channel.
    pub channel_templates: Vec<Option<WaveformTemplate>>,
    /// Deduplicated spike events with global `sample_index` across the recording.
    pub spikes: Vec<DeduplicatedSpike>,
}

/// Out-of-core streaming spike sorter.
pub struct StreamingSpikeRunner {
    config: StreamingSortConfig,
}

impl StreamingSpikeRunner {
    pub fn new(config: StreamingSortConfig) -> Self {
        Self { config }
    }

    /// Streams `source` out-of-core on the runtime selected by [`ComputeTarget::from_env`].
    pub fn run(
        &self,
        source: &dyn RecordingSource,
        pipeline: &Pipeline,
        probe: &ProbeLayout,
    ) -> DspResult<StreamingSortResult> {
        let target = ComputeTarget::from_env().map_err(|e| DspError::ComputeError(e.to_string()))?;
        self.run_with(target, source, pipeline, probe)
    }

    /// Streams `source` out-of-core on `target`.
    pub fn run_with(
        &self,
        target: ComputeTarget,
        source: &dyn RecordingSource,
        pipeline: &Pipeline,
        probe: &ProbeLayout,
    ) -> DspResult<StreamingSortResult> {
        struct Task<'a> {
            runner: &'a StreamingSpikeRunner,
            source: &'a dyn RecordingSource,
            pipeline: &'a Pipeline,
            probe: &'a ProbeLayout,
        }
        impl ComputeTask for Task<'_> {
            type Output = DspResult<StreamingSortResult>;
            fn run<R: Runtime>(self, client: ComputeClient<R>) -> Self::Output {
                self.runner.run_on(client, self.source, self.pipeline, self.probe)
            }
        }
        target
            .run(Task { runner: self, source, pipeline, probe })
            .map_err(|e| DspError::ComputeError(e.to_string()))?
    }

    /// Streams `source` out-of-core in halo-padded batches using double-buffered I/O prefetching
    /// and a persistent `PipelineWorkspace` on `client`'s runtime, returning global spikes and
    /// per-channel templates.
    pub fn run_on<R: Runtime>(
        &self,
        client: ComputeClient<R>,
        source: &dyn RecordingSource,
        pipeline: &Pipeline,
        probe: &ProbeLayout,
    ) -> DspResult<StreamingSortResult> {
        let info = source.info();
        let channels = info.channel_count();
        let total_samples = info.samples;
        let fs = info.sample_rate_hz();

        let (left_halo, right_halo) = self.config.compute_halos(fs, pipeline)?;
        let batch_samples = self.config.batch_samples(fs);
        let refrac_samples = self.config.refractory_samples(fs);
        let pre_samples = self.config.pre_samples(fs);
        let post_samples = self.config.post_samples(fs);
        let snippet_samples = pre_samples + post_samples;
        let k_neighbors = self.config.k_neighbors.clamp(1, channels.max(1));

        let schedule = ChunkSchedule::full_recording(
            total_samples,
            batch_samples,
            left_halo,
            right_halo,
        );

        let mut workspace = PipelineWorkspace::<R>::new(
            client,
            pipeline.clone(),
            channels,
            schedule.max_read_samples().max(1),
            fs,
        )
        .map_err(|e| dsp_core::DspError::InvalidConfig(e.to_string()))?;

        // Phase 1: Calibrate fixed per-channel noise floor (sigma_n) from chunks spread over the recording
        let channel_sigmas_uv =
            calibrate_noise(source, &mut workspace, &self.config, (left_halo, right_halo))?;

        // Pre-upload per-channel sigmas and K-nearest neighbor table to VRAM once
        let sigmas_handle = workspace
            .client()
            .create_from_slice(f32::as_bytes(&channel_sigmas_uv));
        let knn_table = precompute_knn_table(probe, channels, k_neighbors);
        let knn_handle = workspace
            .client()
            .create_from_slice(u32::as_bytes(&knn_table));

        let max_spikes_per_channel =
            ((batch_samples as usize) / refrac_samples.max(1)).clamp(256, 32_768);

        // Spikes are detected where a full snippet (plus sinc taps) can be cut from the recording.
        let margin = extraction_margin(self.config.apply_sinc_shift);
        let detect_range = self.config.detection_range(fs, total_samples);

        // Phase 2: Stream HaloWindows with double-buffered prefetching and zero-readback VRAM kernels
        let mut accumulators: Vec<TemplateAccumulator> = (0..channels)
            .map(|ch| {
                let ids = knn_table[ch * k_neighbors..(ch + 1) * k_neighbors].iter().map(|&c| c as usize).collect();
                TemplateAccumulator::new(ids, snippet_samples)
            })
            .collect();
        let mut channel_spike_counts = vec![0u64; channels];
        let mut total_raw_crossings = 0u64;
        let mut total_dedup_spikes = 0u64;
        let mut global_spikes = Vec::new();
        let elems_per_spike = k_neighbors * snippet_samples;
        let mut carry = DetectionCarry::new(channels);
        let mut dedup = StreamingDedup::new(probe, self.config.spatial_radius_um, refrac_samples as u64);
        debug_assert!(left_halo as usize >= refrac_samples + pre_samples + margin);

        let reader = PrefetchReader::new(source, schedule);
        reader.for_each_window(|win, raw_padded| {
            let n_read = win.read_len();
            let read_start = win.read_global.start;

            // 1. Filter padded chunk in-VRAM (reusing persistent ping-pong buffers without host readback)
            let filt_handle = workspace.process_chunk_in_vram(raw_padded, n_read);

            // 2. Detect crossings in this window's share of the detection range, continuing the
            //    refractory period from the previous window
            let det_start = win.valid_global.start.max(detect_range.start);
            let det_end = win.valid_global.end.min(detect_range.end);
            if det_start < det_end {
                let crossings = execute_detect_spikes_in_vram::<R>(
                    workspace.client(),
                    &filt_handle,
                    &sigmas_handle,
                    channels,
                    n_read,
                    (det_start - read_start) as usize,
                    (det_end - read_start) as usize,
                    read_start,
                    self.config.threshold_factor,
                    refrac_samples,
                    max_spikes_per_channel,
                    Some(&mut carry),
                );
                total_raw_crossings += crossings.len() as u64;
                dedup.push(&crossings);
            }

            // 3. Spatial deduplication of every crossing whose ±refractory neighbourhood is complete
            let finalized = if win.valid_global.end >= total_samples {
                dedup.finish()
            } else {
                dedup.finalize_before(win.valid_global.end)
            };
            total_dedup_spikes += finalized.len() as u64;
            if finalized.is_empty() {
                return Ok(());
            }
            let local: Vec<DeduplicatedSpike> = finalized
                .iter()
                .map(|d| DeduplicatedSpike { sample_index: d.sample_index - read_start, ..d.clone() })
                .collect();

            // 4. Extract Blackman-Harris sinc-realigned snippets and reduce moments directly in VRAM
            if let Some(extracted) = execute_extract_sinc_in_vram::<R>(
                workspace.client(),
                &filt_handle,
                &knn_handle,
                channels,
                n_read,
                &local,
                k_neighbors,
                pre_samples,
                post_samples,
                self.config.apply_sinc_shift,
            ) {
                debug_assert_eq!(extracted.dropped, 0, "finalized spikes lie inside the window");
                let batch_stats = execute_reduce_templates_in_vram::<R>(
                    workspace.client(),
                    &extracted.snippets,
                    &extracted.primary_channels,
                    channels,
                    extracted.num_spikes(),
                    k_neighbors,
                    snippet_samples,
                );

                // 5. Merge batch moments into online Welford/Chan accumulators in O(1) memory
                for (ch, acc) in accumulators.iter_mut().enumerate() {
                    let count = batch_stats.counts.get(ch).copied().unwrap_or(0) as u64;
                    if count > 0 {
                        let start = ch * elems_per_spike;
                        let end = start + elems_per_spike;
                        acc.merge_batch(
                            count,
                            &batch_stats.sum[start..end],
                            &batch_stats.sum_sq[start..end],
                        );
                    }
                }
            }

            // 6. Record global spike timestamps
            for d in finalized {
                if d.primary_channel < channels {
                    channel_spike_counts[d.primary_channel] += 1;
                }
                global_spikes.push(d);
            }

            Ok(())
        })?;

        let channel_templates = accumulators.iter().map(|a| a.finalize()).collect();

        Ok(StreamingSortResult {
            channels,
            total_samples,
            sample_rate_hz: fs,
            halos: (left_halo, right_halo),
            channel_sigmas_uv,
            total_raw_crossings,
            total_dedup_spikes,
            channel_spike_counts,
            channel_templates,
            spikes: global_spikes,
        })
    }
}

/// Per-channel noise floor σ (Quiroga MAD) as the median over `config.calibration_chunks` chunks
/// spread evenly across the recording (total `config.calibration_duration_sec`). Each chunk is
/// filtered with `halos` of context so its interior is settled. Independent of the batch size.
pub fn calibrate_noise<R: Runtime>(
    source: &dyn RecordingSource,
    workspace: &mut PipelineWorkspace<R>,
    config: &StreamingSortConfig,
    halos: (u64, u64),
) -> DspResult<Vec<f32>> {
    let info = source.info();
    let (channels, total, fs) = (info.channel_count(), info.samples, info.sample_rate_hz());
    let all_ch: Vec<usize> = (0..channels).collect();
    let mut per_chunk: Vec<Vec<f32>> = vec![Vec::new(); channels];

    for chunk in config.calibration_chunks(fs, total) {
        let read = chunk.start.saturating_sub(halos.0)..(chunk.end + halos.1).min(total);
        let n = (read.end - read.start) as usize;
        let mut raw = vec![0.0f32; channels * n];
        source.read(&all_ch, read.clone(), &mut raw)?;
        let mut filt = vec![0.0f32; raw.len()];
        workspace.process_chunk(&raw, n, &mut filt);
        let interior = (chunk.start - read.start) as usize..(chunk.end - read.start) as usize;
        let sigmas: Vec<f32> = std::thread::scope(|s| {
            let handles: Vec<_> = (0..channels)
                .map(|ch| {
                    let row = &filt[ch * n + interior.start..ch * n + interior.end];
                    s.spawn(move || estimate_noise_std(row))
                })
                .collect();
            handles.into_iter().map(|h| h.join().expect("noise worker")).collect()
        });
        for (acc, sigma) in per_chunk.iter_mut().zip(sigmas) {
            acc.push(sigma);
        }
    }

    Ok(per_chunk
        .into_iter()
        .map(|mut v| {
            if v.is_empty() {
                return 0.0;
            }
            v.sort_by(f32::total_cmp);
            let m = v.len() / 2;
            if v.len() % 2 == 1 { v[m] } else { 0.5 * (v[m - 1] + v[m]) }
        })
        .collect())
}
