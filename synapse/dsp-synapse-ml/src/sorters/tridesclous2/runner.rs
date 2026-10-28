//! Tridesclous 2 over a whole recording (`tridesclous2.py` `_run_from_folder`), halo windows through
//! the device as the other runners.
//!
//! 1. **Preprocessing**: Bessel band-pass 150–6000 Hz, common median reference (≥ 32 channels), local
//!    whitening; noise levels (MAD).
//! 2. **Detection**: `locally_exclusive` over every window; a uniform selection of
//!    `max(min_n_peaks, n_peaks_per_channel · channels)` peaks.
//! 3. **Clustering** (`iterative-isosplit`): local SVD features, [`split_clusters`] with
//!    SpikeInterface's isosplit ([`isosplit_si`], clusters under `isosplit_min_cluster_size` → −1),
//!    mean templates from the features, cleaning, merging by similarity, small clusters.
//! 4. **Templates for matching**: each unit's channels within `template_radius_um` of its peaks'
//!    channel barycentre; the mean waveform of its clustered peaks ([`UnitTemplateAccumulator`],
//!    `ms_before` / `ms_after`), zeros off those channels; cleaning.
//! 5. **Matching**: [`TdcPeeler`] per window (host; windows in parallel), with the fine detector.
//! 6. **Final merges** ([`auto_merge`], thresholds 0.05 … 0.35, lag `merge_similarity_lag_ms`).

use cubecl::prelude::Client;
use dsp_base::core::buffer;
use dsp_base::linalg::SecondMomentAccumulator;
use dsp_base::pipeline::{Pipeline, PipelineStage, PipelineWorkspace};
use dsp_base::spatial::SpatialWhitening;
use dsp_core::progress::Stages;
use dsp_core::{ChunkSchedule, DspError, DspResult, HaloWindow, ProgressSink, RecordingSource};
use dsp_io::neuro::probe::SensorLayout;
use dsp_synapse::core::{SortingOutput, UnitTemplateAccumulator, WaveformTemplate};
use dsp_synapse::features::{ChannelNeighbourhoods, LocalSvd};
use dsp_synapse::metrics::{CompositeScoreOptions, RefractoryOptions};
use dsp_synapse::sorting::{isosplit_si, subsample_peaks, IsosplitOptions, IsosplitVariant, SubsampleOptions};

use super::Tridesclous2Config;
use crate::sorters::components::{
    auto_merge, clean_templates, fine_prototype, merge_by_similarity, remove_small_clusters, split_clusters, templates_from_svd, AutoMergeOptions, CleanOptions,
    FineFilter, LocallyExclusiveDetector, PeakSign, SparseFeatures, SplitOptions, TdcPeeler, TdcPeelerOptions, Templates,
};
use crate::sorters::kilosort4::runner::DeviceWindows;
use crate::sorters::mountainsort5::MaskedSnippets;
use crate::sorters::result::{AmplitudeScale, SorterResult};
use crate::sorters::spykingcircus2::runner::{fits, median, spread_windows};

/// Sorter name in [`SortingOutput`].
pub const TRIDESCLOUS2_SORTER: &str = "tridesclous2";

pub const STAGE_PREPROCESSING: &str = "Fitting preprocessing";
pub const STAGE_DETECTION: &str = "Detecting spikes";
pub const STAGE_FEATURES: &str = "Computing features";
pub const STAGE_CLUSTERING: &str = "Clustering";
pub const STAGE_TEMPLATES: &str = "Estimating templates";
pub const STAGE_MATCHING: &str = "Matching templates";
pub const STAGE_MERGING: &str = "Merging units";

const MAD_TO_SIGMA: f64 = 0.674_489_750_196_081_7;

/// A Tridesclous 2 run.
#[derive(Debug, Clone)]
pub struct Tridesclous2Result {
    pub sample_rate_hz: f64,
    pub total_samples: u64,
    pub n_units: usize,
    pub spike_samples: Vec<u64>,
    pub spike_units: Vec<u32>,
    /// The peeler's amplitude (a scale of the template).
    pub spike_scalings: Vec<f32>,
    pub templates: Templates,
    pub nbefore: usize,
    pub noise_levels: Vec<f64>,
    pub detected: usize,
    pub selected: usize,
    pub final_merges: usize,
    pub positions: Vec<[f32; 2]>,
    pub pipeline: Pipeline,
}

impl Tridesclous2Result {
    /// Main channel of each unit: its template's deepest trough.
    pub fn main_channels(&self) -> Vec<usize> {
        let t = &self.templates;
        (0..t.len())
            .map(|u| {
                let tp = &t.data[u * t.width * t.channels..(u + 1) * t.width * t.channels];
                (0..t.width * t.channels).fold(0usize, |b, e| if tp[e] < tp[b] { e } else { b }) % t.channels
            })
            .collect()
    }

    /// The sorter-independent result ([`SorterResult`]).
    pub fn sorter_result(&self) -> SorterResult {
        let t = &self.templates;
        let (w, m) = (t.width, t.channels);
        let main = self.main_channels();
        let mut counts = vec![0usize; self.n_units];
        self.spike_units.iter().for_each(|&u| counts[u as usize] += 1);
        let trough: Vec<f32> = (0..self.n_units).map(|u| (0..w).map(|s| t.data[(u * w + s) * m + main[u]]).fold(f32::INFINITY, f32::min)).collect();
        let templates = (0..self.n_units)
            .map(|u| {
                let chans: Vec<usize> = (0..m).filter(|&c| t.sparsity[u * m + c]).collect();
                (!chans.is_empty()).then(|| {
                    let mean: Vec<f32> = chans.iter().flat_map(|&c| (0..w).map(move |s| t.data[(u * w + s) * m + c])).collect();
                    WaveformTemplate::with_count(chans.clone(), w, counts[u].max(1), mean, vec![0.0; chans.len() * w])
                })
            })
            .collect();
        SorterResult {
            sorter: TRIDESCLOUS2_SORTER.into(),
            sample_rate_hz: self.sample_rate_hz,
            total_samples: self.total_samples,
            n_units: self.n_units,
            spike_samples: self.spike_samples.clone(),
            spike_units: self.spike_units.clone(),
            spike_amplitudes: self.spike_units.iter().zip(&self.spike_scalings).map(|(&u, &a)| a * trough[u as usize]).collect(),
            spike_locations: self.spike_units.iter().map(|&u| { let p = self.positions[main[u as usize]]; [p[0], p[1], 0.0] }).collect(),
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

/// The filter chain of `cfg`: Bessel band-pass, common median reference on enough channels.
pub fn filter_stages(cfg: &Tridesclous2Config, channels: usize) -> Vec<PipelineStage> {
    use dsp_base::filter::{FilterBand, FilterSpec};
    let mut stages = Vec::new();
    if cfg.do_bandpass {
        let band = match cfg.bandpass_high_hz {
            Some(high) => FilterBand::Bandpass(cfg.bandpass_low_hz, high),
            None => FilterBand::Highpass(cfg.bandpass_low_hz),
        };
        stages.push(PipelineStage::Filter(FilterSpec::bessel(cfg.filter_order, band)));
    }
    if cfg.do_common_reference && channels >= cfg.common_reference_min_channels {
        stages.push(PipelineStage::CommonMedianReference);
    }
    stages
}

fn filter_error(e: dsp_base::filter::FilterError) -> DspError {
    DspError::InvalidConfig(e.to_string())
}

/// Runs Tridesclous 2 (module docs).
pub fn run(client: &Client, source: &dyn RecordingSource, probe: &SensorLayout, cfg: &Tridesclous2Config, progress: &dyn ProgressSink) -> DspResult<Tridesclous2Result> {
    let info = source.info();
    let (channels, total, fs) = (info.channel_count(), info.samples, info.sample_rate_hz());
    if probe.total_channels() != channels {
        return Err(DspError::InvalidConfig(format!("probe has {} sites, recording {channels} channels", probe.total_channels())));
    }
    let positions: Vec<[f32; 2]> = probe.sites().iter().map(|s| [s.position.x_um, s.position.y_um]).collect();
    let ms = |v: f64| (v * fs / 1000.0) as usize;
    let (cb, ca) = (ms(cfg.clustering_ms_before), ms(cfg.clustering_ms_after));
    let (nbefore, nafter) = (ms(cfg.ms_before), ms(cfg.ms_after));
    let width = nbefore + nafter;
    if total < (4 * width) as u64 {
        return Err(DspError::InvalidConfig(format!("{total} samples: too short for {width}-sample waveforms")));
    }
    let names = [
        (STAGE_PREPROCESSING, "chunks"),
        (STAGE_DETECTION, "windows"),
        (STAGE_FEATURES, "windows"),
        (STAGE_CLUSTERING, "steps"),
        (STAGE_TEMPLATES, "windows"),
        (STAGE_MATCHING, "windows"),
        (STAGE_MERGING, "steps"),
    ];
    let stages = Stages::new(progress, &names);

    // 1. Preprocessing
    let mut pipeline_stages = filter_stages(cfg, channels);
    let filtering = Pipeline::with_stages(pipeline_stages.clone());
    let (settle_l, settle_r) = filtering.settling(fs).map_err(filter_error)?;
    let chunk = ((cfg.chunk_sec * fs) as u64).max(1);
    // Peeler: 2 · max(nbefore, nafter) and the fine detector's sweep + prototype; detection sweeps
    let margin = (4 * width + ms(cfg.detection_exclude_sweep_ms) + 2) as u64;
    let halos = (settle_l as u64 + margin, settle_r as u64 + margin);
    let max_read = (chunk + halos.0 + halos.1).min(total) as usize;
    dsp_core::compute::device_elements("a window of [channels, samples]", &[channels, max_read])?;
    if cfg.do_whiten {
        let windows = spread_windows(total, cfg.whitening_chunks, ms(cfg.whitening_chunk_ms) as u64, (settle_l as u64, settle_r as u64));
        let read = windows.iter().map(HaloWindow::read_len).max().unwrap_or(1);
        let workspace = PipelineWorkspace::<f32>::new(client.clone(), filtering.clone(), channels, read, fs).map_err(filter_error)?;
        let mut device = DeviceWindows::new(source, workspace, None);
        let mut moment = SecondMomentAccumulator::<f32>::new(client, channels);
        let mut done = 0u64;
        device.for_each_while(&windows, |w, filtered| {
            moment.add(filtered, w.read_len(), w.valid_local.clone());
            done += 1;
            stages.report(STAGE_PREPROCESSING, done, windows.len() as u64);
            Ok(true)
        })?;
        let cov = moment.finish();
        let w = match cfg.whitening_radius_um {
            Some(r) => SpatialWhitening::local_radius_from_covariance(&cov, channels, &positions, r, cfg.whitening_epsilon),
            None => SpatialWhitening::zca_from_covariance::<f32>(client, &cov, channels, cfg.whitening_epsilon),
        };
        pipeline_stages.push(PipelineStage::SpatialWhitening(w));
    }
    let pipeline = Pipeline::with_stages(pipeline_stages);
    let workspace = PipelineWorkspace::<f32>::new(client.clone(), pipeline.clone(), channels, max_read, fs).map_err(filter_error)?;
    let mut device = DeviceWindows::new(source, workspace, None);
    let noise_windows = spread_windows(total, cfg.noise_chunks, ms(cfg.noise_chunk_ms) as u64, halos);
    let mut noise_sum = vec![0.0f64; channels];
    device.for_each_while(&noise_windows, |w, prepared| {
        let x = buffer::download_prefix::<f32>(client, prepared.clone(), channels * w.read_len());
        for (c, sum) in noise_sum.iter_mut().enumerate() {
            let mut row: Vec<f32> = x[c * w.read_len() + w.valid_local.start..c * w.read_len() + w.valid_local.end].to_vec();
            let med = median(&mut row) as f32;
            let mut dev: Vec<f32> = row.iter().map(|v| (v - med).abs()).collect();
            *sum += median(&mut dev) / MAD_TO_SIGMA;
        }
        Ok(true)
    })?;
    let noise_levels: Vec<f64> = noise_sum.iter().map(|s| s / noise_windows.len() as f64).collect();
    let schedule = ChunkSchedule::full_recording(total, chunk, halos.0, halos.1);
    let windows: Vec<HaloWindow> = schedule.windows().to_vec();
    let n_windows = windows.len() as u64;

    // 2. Detection over the whole recording, uniform selection
    let detector = LocallyExclusiveDetector::new(client, &positions, &noise_levels, cfg.detect_threshold, cfg.detection_radius_um, ms(cfg.detection_exclude_sweep_ms), PeakSign::Neg);
    let mut peaks = Vec::new();
    let mut done = 0u64;
    device.for_each_while(&windows, |w, prepared| {
        peaks.extend(detector.detect(client, prepared, w.read_len(), w.valid_local.clone(), w.read_global.start));
        done += 1;
        stages.report(STAGE_DETECTION, done, n_windows);
        Ok(true)
    })?;
    let detected = peaks.len();
    if detected == 0 {
        return Err(DspError::InvalidConfig("no peak reached the detection threshold".into()));
    }
    let all_samples: Vec<u64> = peaks.iter().map(|p| p.sample).collect();
    let all_channels: Vec<usize> = peaks.iter().map(|p| p.channel as usize).collect();
    let n_peaks = cfg.min_n_peaks.max(cfg.n_peaks_per_channel * channels);
    let chosen = subsample_peaks(&all_samples, &all_channels, &SubsampleOptions { n_peaks, per_channel: false, seed: cfg.seed });
    let sel_samples: Vec<u64> = chosen.iter().map(|&i| all_samples[i]).collect();
    let sel_channels: Vec<u32> = chosen.iter().map(|&i| all_channels[i] as u32).collect();
    let selected = sel_samples.len();
    let in_window = |w: &HaloWindow, list: &[u64]| -> std::ops::Range<usize> { list.partition_point(|&s| s < w.valid_global.start)..list.partition_point(|&s| s < w.valid_global.end) };

    // 3. Clustering: SVD features, splits, templates, cleaning, merges, small clusters
    let fit_idx = subsample_peaks(&sel_samples, &sel_channels.iter().map(|&c| c as usize).collect::<Vec<_>>(), &SubsampleOptions { n_peaks: cfg.svd_peaks_fit, per_channel: false, seed: cfg.seed });
    let fit_samples: Vec<u64> = fit_idx.iter().map(|&i| sel_samples[i]).collect();
    let fit_channels: Vec<u32> = fit_idx.iter().map(|&i| sel_channels[i]).collect();
    let mut fit_waves = MaskedSnippets::new(&positions, Some(0.0), cb, ca);
    device.for_each_while(&windows, |w, prepared| {
        let r = in_window(w, &fit_samples);
        let (s, c): (Vec<u32>, Vec<u32>) = r
            .map(|i| ((fit_samples[i] - w.read_global.start) as u32, fit_channels[i]))
            .filter(|&(s, _)| fits(s as usize, cb, ca, w.read_len()))
            .unzip();
        fit_waves.extract(client, prepared, w.read_len(), &s, &c);
        Ok(true)
    })?;
    let comps = cfg.n_svd_components_per_channel;
    let fit_rows: Vec<f32> = (0..fit_waves.len()).flat_map(|i| fit_waves.row(i).to_vec()).collect();
    let svd = LocalSvd::fit_rows(&fit_rows, cb + ca, cb, comps);
    drop(fit_waves);
    let neighbourhoods = ChannelNeighbourhoods::within_radius(&positions, cfg.features_radius_um);
    let nb = neighbourhoods.max_neighbours;
    let mut features = vec![0.0f32; selected * comps * nb];
    let mut done = 0u64;
    device.for_each_while(&windows, |w, prepared| {
        let r = in_window(w, &sel_samples);
        if !r.is_empty() {
            let s: Vec<u32> = r.clone().map(|i| (sel_samples[i] - w.read_global.start) as u32).collect();
            let h = svd.transform(client, prepared, channels, w.read_len(), &s, &sel_channels[r.clone()], &neighbourhoods);
            let f = buffer::download_prefix::<f32>(client, h, s.len() * comps * nb);
            features[r.start * comps * nb..r.end * comps * nb].copy_from_slice(&f);
        }
        done += 1;
        stages.report(STAGE_FEATURES, done, n_windows);
        Ok(true)
    })?;
    stages.report(STAGE_CLUSTERING, 0, 3);
    let sparse = SparseFeatures { data: &features, components: comps, mask: &neighbourhoods };
    let split = SplitOptions {
        split_radius_um: cfg.split_radius_um,
        recursive: true,
        recursive_depth: cfg.clustering_recursive_depth,
        min_size_split: cfg.min_size_split,
        n_pca_features: cfg.n_pca_features,
        minimum_overlap_ratio: 0.25,
    };
    let iso = IsosplitOptions {
        isocut_threshold: cfg.isocut_threshold,
        min_cluster_size: cfg.isosplit_min_cluster_size,
        max_iterations_per_pass: cfg.isosplit_max_iterations_per_pass,
        variant: IsosplitVariant::SpikeInterface,
        ..Default::default()
    };
    let (n_init, mcs, seed) = (cfg.isosplit_n_init, cfg.isosplit_min_cluster_size.max(1), cfg.seed);
    let mut clusterer = |x: &[f32], n: usize, d: usize| -> Vec<i32> {
        // LocalFeatureClustering's isosplit branch: n_init lowered for small sets, small clusters → −1
        let n_init = if n_init > n / mcs { (n / (2 * mcs)).max(2) } else { n_init };
        let x64: Vec<f64> = x.iter().map(|&v| v as f64).collect();
        let labels = isosplit_si(&x64, n, d, n_init, seed, &iso);
        let mut counts = std::collections::HashMap::new();
        labels.iter().for_each(|&l| *counts.entry(l).or_insert(0usize) += 1);
        labels.iter().map(|l| if counts[l] < mcs { -1 } else { *l as i32 }).collect()
    };
    let start: Vec<i64> = sel_channels.iter().map(|&c| c as i64).collect();
    let labels = split_clusters(&start, &sel_channels, &sparse, &positions, &split, &mut clusterer);
    stages.report(STAGE_CLUSTERING, 1, 3);
    let clean_opts = CleanOptions { sparsify_threshold: Some(cfg.clustering_sparsify_threshold), min_snr: Some(cfg.clustering_min_snr), max_jitter: Some(ms(cfg.template_max_jitter_ms)), remove_empty: true, mean_sd_ratio_threshold: f64::INFINITY };
    let (t0, _) = templates_from_svd(&labels, &sel_channels, &features, &neighbourhoods, &svd.components, comps, cb + ca, channels, false);
    let (cleaned, kept) = clean_templates(&t0, &noise_levels, cb, None, &clean_opts);
    let kept_ids: Vec<i64> = kept.iter().map(|&u| t0.unit_ids[u]).collect();
    let labels: Vec<i64> = labels.iter().map(|&l| if l >= 0 && !kept_ids.contains(&l) { -1 } else { l }).collect();
    let merged = merge_by_similarity(&labels, &cleaned, cfg.merge_similarity, ms(cfg.merge_similarity_lag_ms), true);
    let duration = total as f64 / fs;
    let (labels, keep_small) = remove_small_clusters(&merged.labels, duration, cfg.min_firing_rate, detected as f64 / selected as f64);
    let unit_ids: Vec<i64> = merged.templates.unit_ids.iter().copied().filter(|id| keep_small.contains(id)).collect();
    stages.report(STAGE_CLUSTERING, 3, 3);

    // 4. Templates for matching: channels around each unit's peaks' barycentre, mean waveforms
    let k0 = unit_ids.len();
    let index_of = |l: i64| unit_ids.iter().position(|&u| u == l);
    let unit_of: Vec<i32> = labels.iter().map(|&l| if l >= 0 { index_of(l).map_or(-1, |u| u as i32) } else { -1 }).collect();
    let mut sparsity = vec![false; k0 * channels];
    for u in 0..k0 {
        let mut counts: std::collections::BTreeMap<u32, f64> = std::collections::BTreeMap::new();
        for (i, &uu) in unit_of.iter().enumerate() {
            if uu == u as i32 {
                *counts.entry(sel_channels[i]).or_default() += 1.0;
            }
        }
        let total_w: f64 = counts.values().sum();
        let (mut bx, mut by) = (0.0f64, 0.0f64);
        for (&c, &n) in &counts {
            bx += positions[c as usize][0] as f64 * n / total_w;
            by += positions[c as usize][1] as f64 * n / total_w;
        }
        for c in 0..channels {
            sparsity[u * channels + c] = ((positions[c][0] as f64 - bx).hypot(positions[c][1] as f64 - by)) <= cfg.template_radius_um as f64;
        }
    }
    let mut acc = UnitTemplateAccumulator::new(client, k0.max(1), channels, nbefore, width);
    let mut done = 0u64;
    device.for_each_while(&windows, |w, prepared| {
        let r = in_window(w, &sel_samples);
        let (s, l): (Vec<u32>, Vec<i32>) = r
            .map(|i| ((sel_samples[i] - w.read_global.start) as u32, unit_of[i]))
            .filter(|&(s, l)| l >= 0 && fits(s as usize, nbefore, nafter, w.read_len()))
            .unzip();
        acc.add(prepared, w.read_len(), &s, &l);
        done += 1;
        stages.report(STAGE_TEMPLATES, done, n_windows);
        Ok(true)
    })?;
    let mean = buffer::download_prefix::<f32>(client, acc.device_mean().clone(), k0.max(1) * channels * width);
    let mut data = vec![0.0f32; k0 * width * channels];
    for u in 0..k0 {
        for c in 0..channels {
            if sparsity[u * channels + c] {
                for s in 0..width {
                    data[(u * width + s) * channels + c] = mean[(u * channels + c) * width + s];
                }
            }
        }
    }
    let t1 = Templates { unit_ids: unit_ids.clone(), width, channels, data, sparsity };
    let peel_clean = CleanOptions { sparsify_threshold: Some(cfg.template_sparsify_threshold), min_snr: Some(cfg.template_min_snr_ptp), max_jitter: Some(ms(cfg.template_max_jitter_ms)), remove_empty: true, mean_sd_ratio_threshold: f64::INFINITY };
    let (templates, _) = clean_templates(&t1, &noise_levels, nbefore, None, &peel_clean);
    let k = templates.len();

    // 5. Matching
    let opts = TdcPeelerOptions {
        exclude_sweep_ms: cfg.peeler_exclude_sweep_ms,
        detect_threshold: cfg.detect_threshold,
        detection_radius_um: cfg.peeler_detection_radius_um,
        cluster_radius_um: cfg.peeler_cluster_radius_um,
        amplitude_fitting_radius_um: cfg.peeler_amplitude_fitting_radius_um,
        sample_shift: cfg.peeler_sample_shift,
        ms_before: cfg.peeler_ms_before,
        ms_after: cfg.peeler_ms_after,
        max_peeler_loop: cfg.peeler_max_loop,
        amplitude_limits: (cfg.peeler_amplitude_min, cfg.peeler_amplitude_max),
        use_fine_detector: cfg.peeler_fine_detector,
    };
    let fine = match (cfg.peeler_fine_detector, fine_prototype(&templates, nbefore)) {
        (true, Some(proto)) => {
            let mut chunks = Vec::new();
            device.for_each_while(&spread_windows(total, cfg.fine_detector_chunks, ms(500.0) as u64, halos), |w, prepared| {
                let x = buffer::download_prefix::<f32>(client, prepared.clone(), channels * w.read_len());
                let v = w.valid_len();
                chunks.push(((0..channels).flat_map(|c| x[c * w.read_len() + w.valid_local.start..c * w.read_len() + w.valid_local.end].iter().copied()).collect::<Vec<f32>>(), v));
                Ok(true)
            })?;
            Some(FineFilter::fit(&proto, nbefore, &positions, &chunks, cfg.detect_threshold))
        }
        _ => None,
    };
    let peeler = TdcPeeler::new(templates.clone(), nbefore, &positions, &noise_levels, fs, fine, opts);
    let (mut spike_samples, mut spike_units, mut spike_scalings) = (Vec::new(), Vec::new(), Vec::new());
    if k > 0 {
        let batch = rayon::current_num_threads().max(1);
        let mut pending: Vec<(HaloWindow, Vec<f32>)> = Vec::with_capacity(batch);
        let flush = |pending: &mut Vec<(HaloWindow, Vec<f32>)>, samples: &mut Vec<u64>, units: &mut Vec<u32>, scalings: &mut Vec<f32>| {
            use rayon::prelude::*;
            let found: Vec<Vec<(u64, u32, f32)>> = pending
                .par_drain(..)
                .map(|(w, mut x)| {
                    let n = w.read_len();
                    peeler
                        .peel(&mut x, n)
                        .into_iter()
                        .filter(|s| s.sample >= 0 && w.valid_local.contains(&(s.sample as usize)))
                        .map(|s| (w.read_global.start + s.sample as u64, s.unit as u32, s.amplitude as f32))
                        .collect()
                })
                .collect();
            for (t, u, a) in found.into_iter().flatten() {
                samples.push(t);
                units.push(u);
                scalings.push(a);
            }
        };
        let mut done = 0u64;
        device.for_each_while(&windows, |w, prepared| {
            pending.push((w.clone(), buffer::download_prefix::<f32>(client, prepared.clone(), channels * w.read_len())));
            if pending.len() == batch {
                flush(&mut pending, &mut spike_samples, &mut spike_units, &mut spike_scalings);
            }
            done += 1;
            stages.report(STAGE_MATCHING, done, n_windows);
            Ok(true)
        })?;
        flush(&mut pending, &mut spike_samples, &mut spike_units, &mut spike_scalings);
    }

    // 6. Final merges
    let mut templates = templates;
    let mut final_merges = 0;
    if cfg.final_merges && k > 1 {
        stages.report(STAGE_MERGING, 0, 1);
        let mut order: Vec<usize> = (0..spike_samples.len()).collect();
        order.sort_by_key(|&i| (spike_samples[i], spike_units[i]));
        let samples: Vec<u64> = order.iter().map(|&i| spike_samples[i]).collect();
        let scalings: Vec<f32> = order.iter().map(|&i| spike_scalings[i]).collect();
        let mut units: Vec<Vec<usize>> = vec![Vec::new(); k];
        for (i, &oi) in order.iter().enumerate() {
            units[spike_units[oi] as usize].push(i);
        }
        let opts = AutoMergeOptions {
            template_diff_thresholds: (1..8).map(|i| 0.05 * i as f64).collect(),
            max_distance_um: cfg.final_merge_max_distance_um,
            censor_ms: cfg.final_merge_censor_ms,
            sparsity_overlap: cfg.final_merge_sparsity_overlap,
            num_shifts: ms(cfg.merge_similarity_lag_ms),
            ..Default::default()
        };
        let merged = auto_merge(&samples, units, templates, &positions, fs, total, &opts);
        final_merges = merged.merges;
        let mut spikes: Vec<(u64, u32, f32)> = Vec::new();
        for (u, list) in merged.spikes.iter().enumerate() {
            spikes.extend(list.iter().map(|&i| (samples[i], u as u32, scalings[i])));
        }
        spikes.sort_by_key(|s| (s.0, s.1));
        spike_samples = spikes.iter().map(|s| s.0).collect();
        spike_units = spikes.iter().map(|s| s.1).collect();
        spike_scalings = spikes.iter().map(|s| s.2).collect();
        templates = merged.templates;
        stages.report(STAGE_MERGING, 1, 1);
    }
    Ok(Tridesclous2Result {
        sample_rate_hz: fs,
        total_samples: total,
        n_units: templates.len(),
        spike_samples,
        spike_units,
        spike_scalings,
        templates,
        nbefore,
        noise_levels,
        detected,
        selected,
        final_merges,
        positions,
        pipeline,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use dsp_core::compute::ComputeTarget;
    use dsp_io::neuro::synthetic::{SyntheticParams, SyntheticRecording};

    /// Recovers well-separated synthetic units.
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
        let start = std::time::Instant::now();
        let log = |e: &dsp_core::ProgressEvent<'_>| {
            if e.done == e.total {
                eprintln!("  {:>7.2} s {} {}/{}", start.elapsed().as_secs_f64(), e.stage, e.done, e.total);
            }
        };
        let out = run(&client, &rec, &probe, &Tridesclous2Config::default(), &log).expect("run");
        let total = rec.info().samples;
        let acc: Vec<f64> = (0..rec.unit_count())
            .map(|g| {
                let truth = rec.spike_times(g, 0..total);
                (0..out.n_units)
                    .map(|u| {
                        let mine: Vec<u64> = (0..out.spike_samples.len()).filter(|&i| out.spike_units[i] == u as u32).map(|i| out.spike_samples[i]).collect();
                        let (mut j, mut hits) = (0usize, 0usize);
                        for &t in &truth {
                            while j < mine.len() && mine[j] + 30 < t {
                                j += 1;
                            }
                            if j < mine.len() && mine[j] <= t + 30 {
                                hits += 1;
                                j += 1;
                            }
                        }
                        hits as f64 / (truth.len() + mine.len() - hits) as f64
                    })
                    .fold(0.0, f64::max)
            })
            .collect();
        eprintln!("TDC2: {} units, {} spikes, detected {}, merges {}, accuracies {acc:.3?}", out.n_units, out.spike_samples.len(), out.detected, out.final_merges);
        assert!(acc.iter().filter(|&&a| a >= 0.8).count() >= 5, "{acc:?}");
    }
}
