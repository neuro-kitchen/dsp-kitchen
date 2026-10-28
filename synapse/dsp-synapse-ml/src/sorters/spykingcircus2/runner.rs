//! SpyKING CIRCUS 2 over a whole recording (`spyking_circus2.py` `_run_from_folder`), halo windows
//! through the device as the other runners.
//!
//! 1. **Preprocessing**: Bessel band-pass, common median reference (≥ 32 channels), whitening
//!    (local by radius) fitted on evenly spaced chunks (upstream: random chunks); noise levels: the
//!    mean over chunks of each channel's median absolute deviation.
//! 2. **Prototype**: `locally_exclusive` peaks (radius `radius_um / 2`, sweep `max(ms_before,
//!    ms_after)`) over windows in a seeded shuffled order until `prototype_peaks`, their own-channel
//!    waveforms, a uniform subset, [`fn@prototype`].
//! 3. **Detection**: [`MatchedFilter`] over windows in a seeded shuffled order until `max(min_n_peaks,
//!    n_peaks_per_channel · channels)` peaks (upstream stops after that many peaks over shuffled
//!    chunks the same way), sorted by sample; a uniform selection of that many.
//! 4. **Features**: the local SVD fitted on `svd_peaks_fit` of them ([`LocalSvd::fit_rows`]), every
//!    selected peak transformed in its window.
//! 5. **Clustering** (iterative HDBSCAN): [`split_clusters`] from the peak channels, templates from
//!    the features, [`clean_templates`], [`merge_by_similarity`]; then upstream's sorter does
//!    templates, cleaning and [`remove_small_clusters`] again.
//! 6. **Matching**: [`CircusOmp`] per window (`chunk_sec` + margins), the window's spikes kept; the
//!    products on the device a batch of windows at a time, their pursuits in parallel on the host.
//! 7. **Final cleaning** ([`auto_merge`], `final_merges`): units merged by template similarity and
//!    cross-contamination, merged trains censored.
//!
//! Spike amplitudes are the matching's amplitude times the template's trough on its main channel
//! (whitened units); locations that channel's position.

use std::ops::Range;

use cubecl::prelude::Client;
use cubecl::server::Handle;
use dsp_base::core::buffer;
use dsp_base::filter::{FilterBand, FilterSpec};
use dsp_base::linalg::SecondMomentAccumulator;
use dsp_base::pipeline::{Pipeline, PipelineStage, PipelineWorkspace};
use dsp_base::spatial::SpatialWhitening;
use dsp_core::progress::Stages;
use dsp_core::{ChunkSchedule, DspError, DspResult, HaloWindow, ProgressSink, RecordingSource};
use dsp_io::neuro::probe::SensorLayout;
use dsp_synapse::core::{SortingOutput, WaveformTemplate};
use dsp_synapse::features::{ChannelNeighbourhoods, LocalSvd};
use dsp_synapse::metrics::{CompositeScoreOptions, RefractoryOptions};
use dsp_synapse::sorting::{hdbscan_allow_single, subsample_peaks, SubsampleOptions};

use super::Spykingcircus2Config;
use crate::sorters::components::{
    auto_merge, clean_templates, merge_by_similarity, prototype, remove_small_clusters, split_clusters, templates_from_svd, AutoMergeOptions, CircusOmp, CleanOptions, LocallyExclusiveDetector,
    MatchedFilter, MatchedFilterOptions, OmpOptions, PeakSign, SparseFeatures, SplitOptions, Templates,
};
use crate::sorters::kilosort4::runner::DeviceWindows;
use crate::sorters::mountainsort5::MaskedSnippets;
use crate::sorters::result::{AmplitudeScale, SorterResult};

/// Sorter name in [`SortingOutput`].
pub const SPYKINGCIRCUS2_SORTER: &str = "spykingcircus2";

pub const STAGE_PREPROCESSING: &str = "Fitting preprocessing";
pub const STAGE_PROTOTYPE: &str = "Learning the prototype";
pub const STAGE_DETECTION: &str = "Detecting spikes";
pub const STAGE_FEATURES: &str = "Computing features";
pub const STAGE_CLUSTERING: &str = "Clustering";
pub const STAGE_MATCHING: &str = "Matching templates";
pub const STAGE_MERGING: &str = "Merging units";

/// `0.6744897501960817`: the MAD of a standard normal.
const MAD_TO_SIGMA: f64 = 0.674_489_750_196_081_7;

/// A SpyKING CIRCUS 2 run.
#[derive(Debug, Clone)]
pub struct Spykingcircus2Result {
    pub sample_rate_hz: f64,
    pub total_samples: u64,
    pub n_units: usize,
    pub spike_samples: Vec<u64>,
    pub spike_units: Vec<u32>,
    /// The matching's amplitude (a scale of the template).
    pub spike_scalings: Vec<f32>,
    pub templates: Templates,
    pub nbefore: usize,
    pub noise_levels: Vec<f64>,
    pub prototype: Vec<f32>,
    /// Units merged by the final cleaning.
    pub final_merges: usize,
    /// Peaks detected / clustered.
    pub detected: usize,
    pub selected: usize,
    pub positions: Vec<[f32; 2]>,
    pub pipeline: Pipeline,
}

impl Spykingcircus2Result {
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
            sorter: SPYKINGCIRCUS2_SORTER.into(),
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

/// The filter chain of `cfg` (Bessel band-pass, common median reference on enough channels).
pub fn filter_stages(cfg: &Spykingcircus2Config, channels: usize) -> Vec<PipelineStage> {
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

/// `count` evenly spaced windows of `len` samples, read with `halos`.
pub(crate) fn spread_windows(total: u64, count: usize, len: u64, halos: (u64, u64)) -> Vec<HaloWindow> {
    let len = len.clamp(1, total);
    let n = count.max(1) as u64;
    (0..n)
        .map(|i| {
            let start = if n > 1 { i * (total - len) / (n - 1) } else { 0 };
            HaloWindow::around(i as usize, start..start + len, halos.0, halos.1, total)
        })
        .collect()
}

/// A seeded shuffle of `0..n` (Fisher–Yates on splitmix64).
pub(crate) fn shuffled(n: usize, seed: u64) -> Vec<usize> {
    let mut v: Vec<usize> = (0..n).collect();
    let mut state = seed;
    for i in (1..n).rev() {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        let j = ((z ^ (z >> 31)) % (i as u64 + 1)) as usize;
        v.swap(i, j);
    }
    v
}

/// NumPy's median (`v` reordered).
pub(crate) fn median(v: &mut [f32]) -> f64 {
    let n = v.len();
    if n == 0 {
        return 0.0;
    }
    let (lower, mid, _) = v.select_nth_unstable_by(n / 2, f32::total_cmp);
    let mid = *mid as f64;
    if n % 2 == 1 { mid } else { 0.5 * (lower.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64 + mid) }
}

/// Local samples of a window's own range whose snippet `[s − before, s + after)` fits the buffer.
pub(crate) fn fits(s: usize, before: usize, after: usize, len: usize) -> bool {
    s >= before && s + after <= len
}

/// Runs SpyKING CIRCUS 2 (module docs).
pub fn run(client: &Client, source: &dyn RecordingSource, probe: &SensorLayout, cfg: &Spykingcircus2Config, progress: &dyn ProgressSink) -> DspResult<Spykingcircus2Result> {
    let info = source.info();
    let (channels, total, fs) = (info.channel_count(), info.samples, info.sample_rate_hz());
    if probe.total_channels() != channels {
        return Err(DspError::InvalidConfig(format!("probe has {} sites, recording {channels} channels", probe.total_channels())));
    }
    let positions: Vec<[f32; 2]> = probe.sites().iter().map(|s| [s.position.x_um, s.position.y_um]).collect();
    let ms = |v: f64| (v * fs / 1000.0) as usize;
    let (nbefore, nafter) = (ms(cfg.ms_before), ms(cfg.ms_after));
    let width = nbefore + nafter;
    if total < (4 * width) as u64 {
        return Err(DspError::InvalidConfig(format!("{total} samples: too short for {width}-sample waveforms")));
    }
    let sweep = ms(cfg.ms_before.max(cfg.ms_after));
    let names = [
        (STAGE_PREPROCESSING, "chunks"),
        (STAGE_PROTOTYPE, "windows"),
        (STAGE_DETECTION, "windows"),
        (STAGE_FEATURES, "windows"),
        (STAGE_CLUSTERING, "steps"),
        (STAGE_MATCHING, "windows"),
        (STAGE_MERGING, "steps"),
    ];
    let stages = Stages::new(progress, &names);

    // 1. Preprocessing
    let mut pipeline_stages = filter_stages(cfg, channels);
    let filtering = Pipeline::with_stages(pipeline_stages.clone());
    let (settle_l, settle_r) = filtering.settling(fs).map_err(filter_error)?;
    let chunk = ((cfg.chunk_sec * fs) as u64).max(1);
    let margin = (3 * width + sweep + 2) as u64;
    let halos = (settle_l as u64 + margin, settle_r as u64 + margin);
    let max_read = (chunk + halos.0 + halos.1).min(total) as usize;
    dsp_core::compute::device_elements("a window of [channels, samples]", &[channels, max_read])?;
    stages.report(STAGE_PREPROCESSING, 0, cfg.whitening_chunks as u64);
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

    // Noise levels: per chunk each channel's MAD, averaged over chunks
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
    let order: Vec<HaloWindow> = shuffled(windows.len(), cfg.seed).into_iter().map(|i| windows[i].clone()).collect();
    let n_windows = windows.len() as u64;

    // 2. Prototype
    let detector = LocallyExclusiveDetector::new(client, &positions, &noise_levels, cfg.detect_threshold, cfg.radius_um / 2.0, sweep, PeakSign::Neg);
    let mut waves = MaskedSnippets::new(&positions, Some(0.0), nbefore, nafter);
    let mut scanned = 0u64;
    device.for_each_while(&order, |w, prepared| {
        let peaks = detector.detect(client, prepared, w.read_len(), w.valid_local.clone(), w.read_global.start);
        let (s, c): (Vec<u32>, Vec<u32>) = peaks
            .iter()
            .map(|p| ((p.sample - w.read_global.start) as u32, p.channel))
            .filter(|&(s, _)| fits(s as usize, nbefore, nafter, w.read_len()))
            .unzip();
        waves.extract(client, prepared, w.read_len(), &s, &c);
        scanned += 1;
        stages.report(STAGE_PROTOTYPE, scanned, n_windows);
        Ok(waves.len() < cfg.prototype_peaks)
    })?;
    stages.report(STAGE_PROTOTYPE, n_windows, n_windows);
    if waves.is_empty() {
        return Err(DspError::InvalidConfig("no peak reached the detection threshold".into()));
    }
    let rows: Vec<f32> = (0..waves.len()).flat_map(|i| waves.row(i).to_vec()).collect();
    let chosen = subsample_peaks(&vec![0u64; waves.len()], &vec![0usize; waves.len()], &SubsampleOptions { n_peaks: cfg.prototype_peaks, per_channel: false, seed: cfg.seed });
    let chosen_rows: Vec<f32> = chosen.iter().flat_map(|&i| rows[i * width..(i + 1) * width].iter().copied()).collect();
    let proto = prototype(&chosen_rows, width, nbefore);
    drop(waves);

    // 3. Detection
    let mf_opts = MatchedFilterOptions { detect_threshold: cfg.detect_threshold, exclude_sweep: sweep, radius_um: cfg.radius_um / 2.0, ..Default::default() };
    let mut mf = MatchedFilter::new(client, &positions, &proto, nbefore, mf_opts);
    let mut threshold_chunks: Vec<(Handle, usize)> = Vec::new();
    let mf_windows = spread_windows(total, cfg.matched_filter_chunks, ms(500.0) as u64, halos);
    device.for_each_while(&mf_windows, |w, prepared| {
        // The valid part only, copied out of the reused window buffer
        let x = buffer::download_prefix::<f32>(client, prepared.clone(), channels * w.read_len());
        let v = w.valid_len();
        let part: Vec<f32> = (0..channels).flat_map(|c| x[c * w.read_len() + w.valid_local.start..c * w.read_len() + w.valid_local.end].iter().copied()).collect();
        threshold_chunks.push((buffer::upload(client, &part), v));
        Ok(true)
    })?;
    mf.fit_thresholds(client, &threshold_chunks);
    drop(threshold_chunks);
    let n_peaks = cfg.min_n_peaks.max(cfg.n_peaks_per_channel * channels);
    let mut peaks = Vec::new();
    let mut scanned = 0u64;
    device.for_each_while(&order, |w, prepared| {
        peaks.extend(mf.detect(client, prepared, w.read_len(), w.valid_local.clone(), w.read_global.start));
        scanned += 1;
        stages.report(STAGE_DETECTION, scanned, n_windows);
        Ok(peaks.len() < n_peaks)
    })?;
    stages.report(STAGE_DETECTION, n_windows, n_windows);
    peaks.sort_by_key(|p| (p.sample, p.channel));
    let detected = peaks.len();
    let all_samples: Vec<u64> = peaks.iter().map(|p| p.sample).collect();
    let all_channels: Vec<usize> = peaks.iter().map(|p| p.channel as usize).collect();
    let selected_idx = subsample_peaks(&all_samples, &all_channels, &SubsampleOptions { n_peaks, per_channel: false, seed: cfg.seed });
    let sel_samples: Vec<u64> = selected_idx.iter().map(|&i| all_samples[i]).collect();
    let sel_channels: Vec<u32> = selected_idx.iter().map(|&i| all_channels[i] as u32).collect();
    let selected = sel_samples.len();
    if selected == 0 {
        return Err(DspError::InvalidConfig("no peak detected".into()));
    }

    // 4. Features: fit on a subset's own-channel waveforms, then every selected peak
    let fit_idx = subsample_peaks(&sel_samples, &sel_channels.iter().map(|&c| c as usize).collect::<Vec<_>>(), &SubsampleOptions { n_peaks: cfg.svd_peaks_fit, per_channel: false, seed: cfg.seed });
    let mut fit_waves = MaskedSnippets::new(&positions, Some(0.0), nbefore, nafter);
    let in_window = |w: &HaloWindow, list: &[u64]| -> Range<usize> {
        let a = list.partition_point(|&s| s < w.valid_global.start);
        let b = list.partition_point(|&s| s < w.valid_global.end);
        a..b
    };
    let fit_samples: Vec<u64> = fit_idx.iter().map(|&i| sel_samples[i]).collect();
    let fit_channels: Vec<u32> = fit_idx.iter().map(|&i| sel_channels[i]).collect();
    let neighbourhoods = ChannelNeighbourhoods::within_radius(&positions, cfg.radius_um);
    let nb = neighbourhoods.max_neighbours;
    let comps = cfg.svd_components;
    let mut features = vec![0.0f32; selected * comps * nb];
    let mut done = 0u64;
    // Pass A: the fit waveforms
    device.for_each_while(&windows, |w, prepared| {
        let r = in_window(w, &fit_samples);
        let (s, c): (Vec<u32>, Vec<u32>) = r
            .clone()
            .map(|i| ((fit_samples[i] - w.read_global.start) as u32, fit_channels[i]))
            .filter(|&(s, _)| fits(s as usize, nbefore, nafter, w.read_len()))
            .unzip();
        fit_waves.extract(client, prepared, w.read_len(), &s, &c);
        Ok(true)
    })?;
    let fit_rows: Vec<f32> = (0..fit_waves.len()).flat_map(|i| fit_waves.row(i).to_vec()).collect();
    let svd = LocalSvd::fit_rows(&fit_rows, width, nbefore, comps);
    drop(fit_waves);
    // Pass B: every selected peak's features
    device.for_each_while(&windows, |w, prepared| {
        let r = in_window(w, &sel_samples);
        if !r.is_empty() {
            let s: Vec<u32> = r.clone().map(|i| (sel_samples[i] - w.read_global.start) as u32).collect();
            let c: Vec<u32> = sel_channels[r.clone()].to_vec();
            let h = svd.transform(client, prepared, channels, w.read_len(), &s, &c, &neighbourhoods);
            let f = buffer::download_prefix::<f32>(client, h, s.len() * comps * nb);
            features[r.start * comps * nb..r.end * comps * nb].copy_from_slice(&f);
        }
        done += 1;
        stages.report(STAGE_FEATURES, done, n_windows);
        Ok(true)
    })?;

    // 5. Clustering
    stages.report(STAGE_CLUSTERING, 0, 4);
    let sparse = SparseFeatures { data: &features, components: comps, mask: &neighbourhoods };
    let split = SplitOptions { split_radius_um: cfg.split_radius_um, recursive: true, recursive_depth: cfg.split_depth, min_size_split: 2 * cfg.min_cluster_size, n_pca_features: cfg.split_pca_features, minimum_overlap_ratio: 0.25 };
    let start: Vec<i64> = sel_channels.iter().map(|&c| c as i64).collect();
    let mcs = cfg.min_cluster_size;
    let mut clusterer = |x: &[f32], n: usize, d: usize| hdbscan_allow_single(client, x, n, d, mcs);
    let labels = split_clusters(&start, &sel_channels, &sparse, &positions, &split, &mut clusterer);
    stages.report(STAGE_CLUSTERING, 1, 4);
    let clean_opts = CleanOptions {
        sparsify_threshold: Some(cfg.sparsify_threshold),
        min_snr: Some(cfg.min_snr),
        max_jitter: Some(ms(cfg.max_jitter_ms)),
        remove_empty: true,
        mean_sd_ratio_threshold: cfg.mean_sd_ratio_threshold,
    };
    let templates_of = |labels: &[i64]| templates_from_svd(labels, &sel_channels, &features, &neighbourhoods, &svd.components, comps, width, channels, true);
    // Inside the clustering method: templates, cleaning, merges
    let (t0, max_std) = templates_of(&labels);
    let (cleaned, kept) = clean_templates(&t0, &noise_levels, nbefore, Some(&max_std), &clean_opts);
    let kept_ids: Vec<i64> = kept.iter().map(|&u| t0.unit_ids[u]).collect();
    let labels: Vec<i64> = labels.iter().map(|&l| if l >= 0 && !kept_ids.contains(&l) { -1 } else { l }).collect();
    let merged = merge_by_similarity(&labels, &cleaned, cfg.merge_similarity, cfg.merge_num_shifts, true);
    stages.report(STAGE_CLUSTERING, 2, 4);
    // The sorter: templates and cleaning again, small clusters
    let (t1, max_std) = templates_of(&merged.labels);
    let (cleaned, kept) = clean_templates(&t1, &noise_levels, nbefore, Some(&max_std), &clean_opts);
    let kept_ids: Vec<i64> = kept.iter().map(|&u| t1.unit_ids[u]).collect();
    let labels: Vec<i64> = merged.labels.iter().map(|&l| if l >= 0 && !kept_ids.contains(&l) { -1 } else { l }).collect();
    let duration = total as f64 / fs;
    let (_, keep_small) = remove_small_clusters(&labels, duration, cfg.min_firing_rate, detected as f64 / selected as f64);
    let final_units: Vec<usize> = (0..cleaned.len()).filter(|&u| keep_small.contains(&cleaned.unit_ids[u])).collect();
    let templates = cleaned.select(&final_units);
    stages.report(STAGE_CLUSTERING, 4, 4);

    // 6. Matching
    let k = templates.len();
    let omp = CircusOmp::new(
        client,
        &templates.data,
        &templates.sparsity,
        k,
        width,
        channels,
        nbefore,
        OmpOptions { amplitudes: (cfg.omp_min_amplitude, f32::INFINITY), max_failures: cfg.omp_max_failures, rank: cfg.omp_rank, vicinity: cfg.omp_vicinity },
    );
    let (mut spike_samples, mut spike_units, mut spike_scalings) = (Vec::new(), Vec::new(), Vec::new());
    let mut done = 0u64;
    if k > 0 {
        // The products of a batch of windows on the device, then their pursuits in parallel (the
        // windows are independent), the spikes in window order
        let batch = rayon::current_num_threads().max(1);
        let mut pending: Vec<(HaloWindow, Vec<f32>)> = Vec::with_capacity(batch);
        let flush = |pending: &mut Vec<(HaloWindow, Vec<f32>)>, samples: &mut Vec<u64>, units: &mut Vec<u32>, scalings: &mut Vec<f32>| {
            use rayon::prelude::*;
            let found: Vec<Vec<(u64, u32, f32)>> = pending
                .par_drain(..)
                .map(|(w, sp)| {
                    omp.match_scalar_products(sp)
                        .into_iter()
                        .filter(|s| w.valid_local.contains(&s.sample))
                        .map(|s| (w.read_global.start + s.sample as u64, s.template, s.amplitude))
                        .collect()
                })
                .collect();
            for (t, u, a) in found.into_iter().flatten() {
                samples.push(t);
                units.push(u);
                scalings.push(a);
            }
        };
        device.for_each_while(&windows, |w, prepared| {
            pending.push((w.clone(), omp.scalar_products(client, prepared, w.read_len())));
            if pending.len() == batch {
                flush(&mut pending, &mut spike_samples, &mut spike_units, &mut spike_scalings);
            }
            done += 1;
            stages.report(STAGE_MATCHING, done, n_windows);
            Ok(true)
        })?;
        flush(&mut pending, &mut spike_samples, &mut spike_units, &mut spike_scalings);
    }
    // 7. Final cleaning
    let mut templates = templates;
    let mut final_merges = 0;
    if cfg.final_merges && k > 1 {
        stages.report(STAGE_MERGING, 0, 1);
        let mut order: Vec<usize> = (0..spike_samples.len()).collect();
        order.sort_by_key(|&i| (spike_samples[i], spike_units[i]));
        let samples: Vec<u64> = order.iter().map(|&i| spike_samples[i]).collect();
        let units_of: Vec<u32> = order.iter().map(|&i| spike_units[i]).collect();
        let scalings: Vec<f32> = order.iter().map(|&i| spike_scalings[i]).collect();
        let mut units: Vec<Vec<usize>> = vec![Vec::new(); k];
        for (i, &u) in units_of.iter().enumerate() {
            units[u as usize].push(i);
        }
        let opts = AutoMergeOptions {
            max_distance_um: cfg.final_merge_max_distance_um,
            censor_ms: cfg.final_merge_censor_ms,
            sparsity_overlap: cfg.final_merge_sparsity_overlap,
            num_shifts: (cfg.final_merge_max_lag_ms * fs / 1000.0) as usize,
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
    let k = templates.len();
    Ok(Spykingcircus2Result {
        sample_rate_hz: fs,
        total_samples: total,
        n_units: k,
        final_merges,
        spike_samples,
        spike_units,
        spike_scalings,
        templates,
        nbefore,
        noise_levels,
        prototype: proto,
        detected,
        selected,
        positions,
        pipeline,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use dsp_core::compute::ComputeTarget;
    use dsp_io::neuro::synthetic::{SyntheticParams, SyntheticRecording};

    #[test]
    fn shuffle_is_a_seeded_permutation() {
        let a = shuffled(50, 42);
        let mut sorted = a.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..50).collect::<Vec<_>>());
        assert_eq!(a, shuffled(50, 42));
        assert_ne!(a, shuffled(50, 43));
    }

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
        let out = run(&client, &rec, &probe, &Spykingcircus2Config::default(), &log).expect("run");
        let total = rec.info().samples;
        let acc: Vec<f64> = (0..rec.unit_count())
            .map(|g| {
                let truth = rec.spike_times(g, 0..total);
                (0..out.n_units)
                    .map(|u| {
                        let mine: Vec<u64> = (0..out.spike_samples.len()).filter(|&i| out.spike_units[i] == u as u32).map(|i| out.spike_samples[i]).collect::<Vec<_>>();
                        let mut mine = mine;
                        mine.sort_unstable();
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
        eprintln!("SC2: {} units, {} spikes, detected {}, accuracies {acc:.3?}", out.n_units, out.spike_samples.len(), out.detected);
        assert!(acc.iter().filter(|&&a| a >= 0.8).count() >= 5, "{acc:?}");
        assert_eq!(out.to_sorting_output(Some(probe)).units.len() <= out.n_units, true);
    }
}
