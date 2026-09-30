//! Out-of-core streaming spike sorting runner connecting `RecordingSource`, `PipelineWorkspace`,
//! boundary-safe `HaloWindow`s, and online Welford `TemplateAccumulator`s.

use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use cubecl::{CubeElement, Runtime};

use dsp_base::pipeline::{Pipeline, PipelineWorkspace};
use dsp_core::{ChunkSchedule, DspResult, ProbeLayout, RecordingSource};
use dsp_stream::PrefetchReader;

use crate::detection::{DeduplicatedSpike, deduplicate_spikes_spatial, estimate_noise_std};
use crate::kernels::{
    execute_detect_spikes_in_vram, execute_extract_sinc_in_vram, execute_reduce_templates_in_vram,
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

    /// Streams `source` out-of-core in halo-padded batches using double-buffered I/O prefetching
    /// and a persistent WGPU `PipelineWorkspace`, returning global spikes and per-channel templates.
    pub fn run(
        &self,
        source: &dyn RecordingSource,
        pipeline: &Pipeline,
        probe: &ProbeLayout,
    ) -> DspResult<StreamingSortResult> {
        let info = source.info();
        let channels = info.channel_count();
        let total_samples = info.samples;
        let fs = info.sample_rate_hz();

        let (left_halo, right_halo) = self.config.compute_halos(fs, pipeline);
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

        let device = WgpuDevice::default();
        let client = WgpuRuntime::client(&device);
        let mut workspace = PipelineWorkspace::<WgpuRuntime>::new(
            client,
            pipeline.clone(),
            channels,
            schedule.max_read_samples().max(1),
            fs,
            false,
        );

        // Phase 1: Calibrate fixed per-channel noise floor (sigma_n) from a representative window
        let cal_samples = ((self.config.calibration_duration_sec.max(0.5) * fs).round() as u64)
            .plus_halo(left_halo)
            .min(total_samples);
        let all_ch: Vec<usize> = (0..channels).collect();
        let mut cal_raw = vec![0.0f32; channels * cal_samples as usize];
        source.read(&all_ch, 0..cal_samples, &mut cal_raw)?;
        let mut cal_filt = vec![0.0f32; cal_raw.len()];
        workspace.process_chunk(&cal_raw, cal_samples as usize, &mut cal_filt);

        let settle = (left_halo as usize).min((cal_samples as usize) / 2);
        let cal_n = cal_samples as usize;
        let mut channel_sigmas_uv = vec![0.0f32; channels];
        std::thread::scope(|s| {
            for (ch, slot) in channel_sigmas_uv.iter_mut().enumerate() {
                let row = &cal_filt[ch * cal_n..(ch + 1) * cal_n];
                s.spawn(move || {
                    *slot = estimate_noise_std(&row[settle..]);
                });
            }
        });

        // Pre-upload per-channel sigmas and K-nearest neighbor table to VRAM once
        let sigmas_handle = workspace
            .client()
            .create_from_slice(f32::as_bytes(&channel_sigmas_uv));
        let knn_table = precompute_knn_table(probe, k_neighbors);
        let knn_handle = workspace
            .client()
            .create_from_slice(u32::as_bytes(&knn_table));

        let max_spikes_per_channel =
            ((batch_samples as usize) / refrac_samples.max(1)).clamp(256, 32_768);

        // Phase 2: Stream HaloWindows with double-buffered prefetching and zero-readback VRAM kernels
        let mut accumulators: Vec<TemplateAccumulator> = (0..channels)
            .map(|_| TemplateAccumulator::new(k_neighbors, snippet_samples))
            .collect();
        let mut channel_spike_counts = vec![0u64; channels];
        let mut total_raw_crossings = 0u64;
        let mut total_dedup_spikes = 0u64;
        let mut global_spikes = Vec::new();
        let elems_per_spike = k_neighbors * snippet_samples;

        let reader = PrefetchReader::new(source, schedule);
        reader.for_each_window(|win, raw_padded| {
            let n_read = win.read_len();

            // 1. Filter padded chunk in-VRAM (reusing persistent ping-pong buffers without host readback)
            let filt_handle = workspace.process_chunk_in_vram(raw_padded, n_read);

            // Compute valid interior range with sufficient lookback/lookahead for sinc resampling
            let min_snip_idx = pre_samples + super::config::SINC_RESAMPLE_MARGIN;
            let max_snip_idx =
                n_read.saturating_sub(post_samples + super::config::SINC_RESAMPLE_MARGIN);
            let valid_start = win.valid_local.start.max(min_snip_idx);
            let valid_end = win.valid_local.end.min(max_snip_idx);
            if valid_start >= valid_end {
                return Ok(());
            }

            // 2. Detect interior threshold crossings directly in VRAM using CubeCL kernel
            let interior_crossings = execute_detect_spikes_in_vram::<WgpuRuntime>(
                workspace.client(),
                &filt_handle,
                &sigmas_handle,
                channels,
                n_read,
                valid_start,
                valid_end,
                self.config.threshold_factor,
                refrac_samples,
                max_spikes_per_channel,
                false,
            );

            total_raw_crossings += interior_crossings.len() as u64;
            if interior_crossings.is_empty() {
                return Ok(());
            }

            // 3. Spatial deduplication across ProbeLayout
            let dedup_local = deduplicate_spikes_spatial(
                &interior_crossings,
                probe,
                self.config.spatial_radius_um,
                refrac_samples as u64,
            );
            total_dedup_spikes += dedup_local.len() as u64;
            if dedup_local.is_empty() {
                return Ok(());
            }

            // 4. Extract Blackman-Harris sinc-realigned snippets and reduce moments directly in VRAM
            if let Some((snippets_handle, prim_handle)) = execute_extract_sinc_in_vram::<WgpuRuntime>(
                workspace.client(),
                &filt_handle,
                &knn_handle,
                channels,
                n_read,
                &dedup_local,
                k_neighbors,
                pre_samples,
                post_samples,
                self.config.apply_sinc_shift,
                false,
            ) {
                let batch_stats = execute_reduce_templates_in_vram::<WgpuRuntime>(
                    workspace.client(),
                    &snippets_handle,
                    &prim_handle,
                    channels,
                    dedup_local.len(),
                    k_neighbors,
                    snippet_samples,
                    false,
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
            for mut d in dedup_local {
                if d.primary_channel < channels {
                    channel_spike_counts[d.primary_channel] += 1;
                }
                d.sample_index = win.to_global_sample(d.sample_index as usize);
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

trait U64Ext {
    fn plus_halo(self, halo: u64) -> u64;
}

impl U64Ext for u64 {
    #[inline]
    fn plus_halo(self, halo: u64) -> u64 {
        self.saturating_add(halo)
    }
}
