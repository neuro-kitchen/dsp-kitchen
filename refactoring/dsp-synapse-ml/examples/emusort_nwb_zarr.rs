//! Run the EMUsort HD-EMG spike sorting pipeline on an NWB Zarr recording (`/acquisition/HDEMG`)
//! and save the resulting motor units as an embedded `/units` Zarr v3 table inside the NWB store.
//!
//! Usage:
//!   cargo run --release -p dsp-synapse-ml --example emusort_nwb_zarr -- [nwb_path] [duration_sec]

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::Instant;

use cubecl::prelude::*;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use dsp_base::filter::{FilterMode, FilterSpec};
use dsp_base::{Pipeline, PipelineStage, SpatialWhitening};
use dsp_synapse::core::{
    RecordingMeta, SortedUnit, SortingOutput, UnitQualityLabel, WaveformTemplate,
};
use dsp_synapse::spatial::center_of_mass::{localize_spike_center_of_mass, waveform_peak_to_peak};
use dsp_synapse::storage::save_nwb_units;
use dsp_synapse::{
    DeduplicatedSpike, SpikeDetector, deduplicate_spikes_spatial, extract_snippets_multichannel,
    hdemg_4x8, hdemg_8x8, hdemg_grid,
};
use dsp_synapse_ml::{EmusortDetector, EmusortLatencyAligner};

fn median_f32(values: &mut [f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(f32::total_cmp);
    let mid = values.len() / 2;
    if values.len() % 2 == 0 {
        0.5 * (values[mid - 1] + values[mid])
    } else {
        values[mid]
    }
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let nwb_path = args
        .get(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("data/nwb/15-25-33_meps.nwb.zarr"));
    let duration_sec: f64 = args
        .get(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(60.0);
    let series = "/acquisition/HDEMG";
    let threshold_sigma: f32 = 6.5;
    let refractory_ms: f64 = 2.5;
    let ied_mm: f32 = 4.0;

    let t0 = Instant::now();
    let rec = dsp_io::open_source(&nwb_path, series)?;
    let info = rec.info();
    let fs = info.sample_rate_hz();
    let n_ch = info.channel_count();
    let total_rec_samples = info.samples;
    let actual_samples = ((duration_sec * fs).round() as u64).min(total_rec_samples) as usize;

    println!("================================================================================");
    println!("  EMUSORT HD-EMG SPIKE SORTING -> EMBEDDED NWB /units ZARR EXPORTER");
    println!("================================================================================");
    println!("  Input/Output NWB Store:   {}", nwb_path.display());
    println!("  Electrical Series:        {series}");
    println!(
        "  Stream Info:              {n_ch} ch @ {fs:.4} Hz ({:.1} s total, sorting first {:.1} s / {actual_samples} samples)",
        total_rec_samples as f64 / fs,
        actual_samples as f64 / fs
    );

    // 1. Read channel-major slice in µV
    let t_read = Instant::now();
    let all_channels: Vec<usize> = (0..n_ch).collect();
    let mut raw = vec![0.0f32; n_ch * actual_samples];
    rec.read(&all_channels, 0..actual_samples as u64, &mut raw)?;
    println!("  [1/6] Read {actual_samples} samples x {n_ch} ch in {:.2?}", t_read.elapsed());

    // 2. Configure HD-EMG Probe Layout
    let probe = if n_ch == 32 {
        hdemg_4x8(ied_mm)
    } else if n_ch == 64 {
        hdemg_8x8(ied_mm)
    } else {
        hdemg_grid("HDEMG", 1, n_ch, ied_mm)
    };
    let positions_xy: Vec<[f32; 2]> = probe
        .contacts
        .iter()
        .map(|s| [s.position.x_um, s.position.y_um])
        .collect();

    // 3. Sequential Preprocessing: CAR -> Bandpass (100-2000 Hz) -> Local Spatial Whitening
    let t_prep = Instant::now();
    let device = WgpuDevice::default();
    let client = WgpuRuntime::client(&device);

    let mut car_bp_pipe = Pipeline::new();
    car_bp_pipe
        .add(PipelineStage::CommonAverageReference)
        .add(PipelineStage::Filter(
            FilterSpec::bandpass(100.0, 2000.0)
                .with_order(4)
                .with_mode(FilterMode::ForwardBackward),
        ));
    let in_handle = client.create_from_slice(f32::as_bytes(&raw));
    let bp_handle = car_bp_pipe.execute::<WgpuRuntime>(&client, &in_handle, n_ch, actual_samples, fs)?;
    let bp_bytes = client.read_one_unchecked(bp_handle);
    let bp_data = f32::from_bytes(&bp_bytes);

    let whiten_samples = actual_samples.min((2.0 * fs) as usize);
    let mut whiten_slice = vec![0.0f32; n_ch * whiten_samples];
    for ch in 0..n_ch {
        whiten_slice[ch * whiten_samples..(ch + 1) * whiten_samples]
            .copy_from_slice(&bp_data[ch * actual_samples..ch * actual_samples + whiten_samples]);
    }
    let whitener = SpatialWhitening::fit_local_knn(
        &whiten_slice,
        n_ch,
        whiten_samples,
        &positions_xy,
        n_ch.min(8),
        1e-5,
    );
    let preprocessed = whitener.apply_cpu(bp_data, n_ch, actual_samples);
    println!("  [2/6] Preprocessed (CAR -> Bandpass -> ZCA Whiten) in {:.2?}", t_prep.elapsed());

    // 4. Universal MUAP Matched-Filter Detection + Spatial Deduplication
    let t_det = Instant::now();
    let refractory_samples = ((refractory_ms * 1e-3 * fs) as usize).max(10);
    let detector = EmusortDetector::from_canonical(threshold_sigma, refractory_samples, None)?;
    let raw_events = detector.detect(&preprocessed, n_ch, actual_samples, fs)?;
    let spatial_radius_um = ied_mm * 1000.0;
    let dedup_window_samples = ((refractory_ms * 1e-3 * fs * 0.6) as u64).max(12);
    let dedup_events: Vec<DeduplicatedSpike> =
        deduplicate_spikes_spatial(&raw_events, &probe, spatial_radius_um, dedup_window_samples);
    println!(
        "  [3/6] Detected {} raw crossings -> {} deduplicated MUAPs in {:.2?}",
        raw_events.len(),
        dedup_events.len(),
        t_det.elapsed()
    );

    // 5. Extract 150-sample multi-channel snippets & Conduction Latency Alignment
    let t_snip = Instant::now();
    let k_neighbors = n_ch.min(8);
    let pre_samples = 50usize;
    let post_samples = 100usize;
    let window_len = pre_samples + post_samples;
    let mut snippets = extract_snippets_multichannel(
        &preprocessed,
        n_ch,
        actual_samples,
        &dedup_events,
        &probe,
        k_neighbors,
        pre_samples,
        post_samples,
        true,
    );
    let aligner = EmusortLatencyAligner::new(25);
    for snip in &mut snippets {
        let lags = aligner.estimate_channel_lags(&snip.waveform, k_neighbors, window_len, 0);
        snip.waveform = aligner.align_snippet(&snip.waveform, k_neighbors, window_len, &lags);
    }
    println!(
        "  [4/6] Extracted & latency-aligned {} snippets in {:.2?}",
        snippets.len(),
        t_snip.elapsed()
    );

    // 6. Contact-Localized Motor Unit Formation & Full 32-Channel Templates
    let t_unit = Instant::now();
    let mut spikes_by_ch: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (i, s) in snippets.iter().enumerate() {
        spikes_by_ch.entry(s.primary_channel).or_default().push(i);
    }

    let mut sorted_units: Vec<SortedUnit> = Vec::new();
    let mut next_unit_id = 0usize;

    // Precompute baseline noise MAD per channel in bandpass µV for SNR calculation
    let noise_window = actual_samples.min((5.0 * fs) as usize);
    let mut ch_noise_uv = vec![1.0f32; n_ch];
    for ch in 0..n_ch {
        let mut slice = bp_data[ch * actual_samples..ch * actual_samples + noise_window].to_vec();
        let med = median_f32(&mut slice);
        let mut abs_dev: Vec<f32> = slice.iter().map(|&v| (v - med).abs()).collect();
        ch_noise_uv[ch] = (1.4826 * median_f32(&mut abs_dev)).max(1e-3);
    }

    let mut groups: Vec<(usize, Vec<usize>)> = Vec::new();
    for ch in 0..n_ch {
        let Some(ch_indices) = spikes_by_ch.get(&ch) else {
            continue;
        };
        if ch_indices.len() < 15 {
            continue;
        }
        let ch_amps: Vec<f32> = ch_indices
            .iter()
            .map(|&idx| {
                snippets[idx].waveform[..window_len]
                    .iter()
                    .map(|v| v.abs())
                    .fold(0.0f32, f32::max)
            })
            .collect();
        let mut amps_copy = ch_amps.clone();
        let med_amp = median_f32(&mut amps_copy);
        let high_count = ch_amps.iter().filter(|&&a| a > med_amp * 1.5).count();
        let low_count = ch_indices.len() - high_count;

        if high_count >= 20 && low_count >= 20 {
            let mut low_idx = Vec::with_capacity(low_count);
            let mut high_idx = Vec::with_capacity(high_count);
            for (local_i, &idx) in ch_indices.iter().enumerate() {
                if ch_amps[local_i] > med_amp * 1.5 {
                    high_idx.push(idx);
                } else {
                    low_idx.push(idx);
                }
            }
            groups.push((ch, low_idx));
            groups.push((ch, high_idx));
        } else {
            groups.push((ch, ch_indices.clone()));
        }
    }

    for (pri_ch, indices) in groups {
        let u_id = next_unit_id;
        next_unit_id += 1;

        let mut spike_samples: Vec<u64> = indices.iter().map(|&i| snippets[i].center_sample).collect();
        let mut order: Vec<usize> = (0..indices.len()).collect();
        order.sort_by_key(|&k| spike_samples[k]);
        spike_samples = order.iter().map(|&k| spike_samples[k]).collect();

        let mut amplitudes_uv = Vec::with_capacity(indices.len());
        let mut locations_um = Vec::with_capacity(indices.len());

        // Accumulate full 32-channel template in µV (CAR+bandpass scale)
        let max_tpl_spikes = indices.len().min(150);
        let mut tpl_mean = vec![0.0f32; n_ch * window_len];
        let mut tpl_m2 = vec![0.0f32; n_ch * window_len];
        let mut tpl_count = 0usize;

        for &ord_i in &order {
            let snip = &snippets[indices[ord_i]];
            let c_samp = snip.center_sample as usize;
            let amp_uv = if c_samp < actual_samples {
                bp_data[pri_ch * actual_samples + c_samp].abs()
            } else {
                snip.waveform[..window_len]
                    .iter()
                    .map(|v| v.abs())
                    .fold(0.0f32, f32::max)
            };
            amplitudes_uv.push(amp_uv);

            let ptp: Vec<f32> = (0..snip.channel_ids.len())
                .map(|r| waveform_peak_to_peak(&snip.waveform[r * window_len..(r + 1) * window_len]))
                .collect();
            let loc = localize_spike_center_of_mass(&snip.channel_ids, &ptp, &probe, 2.0);
            locations_um.push(loc);

            if tpl_count < max_tpl_spikes && c_samp >= pre_samples && c_samp + post_samples <= actual_samples {
                tpl_count += 1;
                let n_f = tpl_count as f32;
                for ch in 0..n_ch {
                    let row_in = &bp_data[ch * actual_samples + (c_samp - pre_samples)..ch * actual_samples + (c_samp + post_samples)];
                    let row_mean = &mut tpl_mean[ch * window_len..(ch + 1) * window_len];
                    let row_m2 = &mut tpl_m2[ch * window_len..(ch + 1) * window_len];
                    for t in 0..window_len {
                        let x = row_in[t];
                        let delta = x - row_mean[t];
                        row_mean[t] += delta / n_f;
                        let delta2 = x - row_mean[t];
                        row_m2[t] += delta * delta2;
                    }
                }
            }
        }

        let denom = (tpl_count.saturating_sub(1)).max(1) as f32;
        let sqrt_n = (tpl_count.max(1) as f32).sqrt();
        let tpl_std: Vec<f32> = tpl_m2.iter().map(|&m2| (m2 / denom).sqrt()).collect();
        let tpl_se: Vec<f32> = tpl_std.iter().map(|&s| s / sqrt_n).collect();
        let trough_index = tpl_mean[pri_ch * window_len..(pri_ch + 1) * window_len]
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(idx, _)| idx)
            .unwrap_or(pre_samples);

        let template = WaveformTemplate {
            channel_ids: (0..n_ch).collect(),
            num_channels: n_ch,
            num_samples: window_len,
            count: tpl_count.max(1),
            trough_index,
            mean: tpl_mean,
            std: tpl_std,
            se: tpl_se,
        };

        let mut unit = SortedUnit::from_spikes(
            u_id,
            pri_ch,
            spike_samples,
            amplitudes_uv,
            locations_um,
            Some(template),
            fs,
            actual_samples as u64,
            ch_noise_uv[pri_ch],
        );
        if unit.snr >= 2.5 && unit.num_spikes() >= 30 {
            unit.quality_label = UnitQualityLabel::Good;
        } else {
            unit.quality_label = UnitQualityLabel::Mua;
        }
        sorted_units.push(unit);
    }

    let num_good = sorted_units
        .iter()
        .filter(|u| u.quality_label == UnitQualityLabel::Good)
        .count();
    let num_mua = sorted_units.len() - num_good;
    let total_spikes: usize = sorted_units.iter().map(|u| u.spike_samples.len()).sum();
    println!(
        "  [5/6] Resolved {} motor units ({num_good} good, {num_mua} mua, {total_spikes} total spikes) in {:.2?}",
        sorted_units.len(),
        t_unit.elapsed()
    );

    let dat_str = nwb_path
        .canonicalize()
        .unwrap_or_else(|_| nwb_path.clone())
        .to_string_lossy()
        .into_owned();
    let sorting = SortingOutput::new(
        "emusort",
        fs,
        total_rec_samples,
        Some(probe),
        sorted_units,
        None,
    )
    .with_recording_meta(RecordingMeta {
        dat_path: Some(dat_str),
        n_channels_dat: Some(n_ch),
        dtype: Some("float32".to_string()),
        offset: 0,
        hp_filtered: false,
    });

    // Save embedded /units Zarr v3 table inside the NWB Zarr store
    let t_save = Instant::now();
    save_nwb_units(&sorting, &nwb_path)?;
    println!(
        "  [6/6] Saved embedded /units table to {} in {:.2?} (total {:.2?})",
        nwb_path.display(),
        t_save.elapsed(),
        t0.elapsed()
    );
    println!("================================================================================");

    Ok(())
}
