pub mod template_reduce;

pub use template_reduce::{
    BatchTemplateStats, execute_reduce_templates_in_vram, reduce_channel_templates_kernel,
};

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
    use cubecl::{CubeElement, Runtime};
    use dsp_io::neuro::probe::{precompute_knn_table, Position3D, SensorLayout, SensorSite};
    use crate::detection::{
        SpikePolarity, SpikeSpacing, deduplicate_spikes_spatial, detect_spikes_with_sigma,
        detection_heights, execute_detect_spikes_in_vram,
    };
    use crate::extraction::{execute_extract_sinc_in_vram, extract_snippets_multichannel};
    use crate::streaming::TemplateAccumulator;

    #[test]
    fn test_cubecl_vram_kernels_match_cpu_reference() {
        let channels = 4usize;
        let samples = 2000usize;
        let k_neighbors = 2usize;
        let pre_samples = 15usize;
        let post_samples = 30usize;
        let snippet_len = pre_samples + post_samples;

        let contacts: Vec<SensorSite> = (0..channels)
            .map(|c| SensorSite {
                channel_id: c,
                device_index: c,
                group_id: 0,
                shank_id: 0,
                position: Position3D::new((c as f32) * 100.0, 0.0, 0.0),
                enabled: true,
            })
            .collect();
        let probe = SensorLayout::new("test-4ch", contacts);

        let mut trace = vec![0.0f32; channels * samples];
        for ch in 0..channels {
            for t in 0..samples {
                trace[ch * samples + t] = (((t + ch * 7) % 11) as f32 - 5.0) * 1.5;
            }
        }

        for &center in &[300usize, 800, 1400] {
            for dt in -10isize..=20 {
                let s = (center as isize + dt) as usize;
                let x = (dt as f32 - 0.25) / 3.0;
                let wave = (-0.5 * x * x).exp() * -95.0;
                trace[s] += wave;
                trace[samples + s] += wave * 0.35;
            }
        }
        for &center in &[550usize, 1150] {
            for dt in -10isize..=20 {
                let s = (center as isize + dt) as usize;
                let x = (dt as f32 + 0.2) / 3.0;
                let wave = (-0.5 * x * x).exp() * -110.0;
                trace[2 * samples + s] += wave;
                trace[3 * samples + s] += wave * 0.4;
            }
        }

        let sigmas = vec![5.0f32; channels];
        let threshold_factor = 5.0f32;
        let refrac = 25usize;

        let spacing = SpikeSpacing::new(refrac);
        let cpu_spikes = detect_spikes_with_sigma(
            &trace,
            channels,
            samples,
            &sigmas,
            threshold_factor,
            SpikePolarity::Negative,
            spacing,
        );
        let cpu_dedup = deduplicate_spikes_spatial(&cpu_spikes, &probe, 150.0, refrac as u64);
        let cpu_snips = extract_snippets_multichannel(
            &trace,
            channels,
            samples,
            &cpu_dedup,
            &probe,
            k_neighbors,
            pre_samples,
            post_samples,
            true,
        );
        let knn_table = precompute_knn_table(&probe, channels, k_neighbors);
        let mut cpu_accs: Vec<TemplateAccumulator> = (0..channels)
            .map(|ch| {
                TemplateAccumulator::new(
                    knn_table[ch * k_neighbors..(ch + 1) * k_neighbors]
                        .iter()
                        .map(|&c| c as usize)
                        .collect(),
                    snippet_len,
                )
            })
            .collect();
        for snip in &cpu_snips {
            cpu_accs[snip.primary_channel].update(snip);
        }

        let device = WgpuDevice::default();
        let client = WgpuRuntime::client(&device);
        let trace_handle = client.create_from_slice(f32::as_bytes(&trace));
        let heights_handle = client.create_from_slice(f32::as_bytes(&detection_heights(&sigmas, threshold_factor)));
        let knn_handle = client.create_from_slice(u32::as_bytes(&knn_table));

        let gpu_spikes = execute_detect_spikes_in_vram::<WgpuRuntime>(
            &client,
            &trace_handle,
            &heights_handle,
            channels,
            samples,
            0..samples,
            0,
            SpikePolarity::Negative,
            spacing,
        );
        assert_eq!(gpu_spikes, cpu_spikes);

        let gpu_dedup = deduplicate_spikes_spatial(&gpu_spikes, &probe, 150.0, refrac as u64);
        let extracted = execute_extract_sinc_in_vram::<WgpuRuntime, f32>(
            &client,
            &trace_handle,
            &knn_handle,
            channels,
            samples,
            &gpu_dedup,
            k_neighbors,
            pre_samples,
            post_samples,
            true,
        )
        .unwrap();

        assert_eq!(extracted.dropped, 0);
        let batch_stats = execute_reduce_templates_in_vram::<WgpuRuntime>(
            &client,
            &extracted.snippets,
            &extracted.primaries,
            channels,
            k_neighbors,
            snippet_len,
        );

        let elems_per_spike = k_neighbors * snippet_len;
        let mut gpu_accs: Vec<TemplateAccumulator> = (0..channels)
            .map(|ch| {
                TemplateAccumulator::new(
                    knn_table[ch * k_neighbors..(ch + 1) * k_neighbors]
                        .iter()
                        .map(|&c| c as usize)
                        .collect(),
                    snippet_len,
                )
            })
            .collect();
        for ch in 0..channels {
            let cnt = batch_stats.counts[ch] as u64;
            if cnt > 0 {
                let s = ch * elems_per_spike;
                let e = s + elems_per_spike;
                gpu_accs[ch].merge_batch(cnt, &batch_stats.mean[s..e], &batch_stats.m2[s..e]);
            }
        }

        for ch in 0..channels {
            assert_eq!(gpu_accs[ch].count(), cpu_accs[ch].count());
            if let (Some(g), Some(c)) = (gpu_accs[ch].finalize(), cpu_accs[ch].finalize()) {
                for (gv, cv) in g.mean.iter().zip(c.mean.iter()) {
                    assert!((gv - cv).abs() < 1e-2, "mean diff on ch {ch}: {gv} vs {cv}");
                }
                for (gv, cv) in g.std.iter().zip(c.std.iter()) {
                    assert!((gv - cv).abs() < 1e-2, "std diff on ch {ch}: {gv} vs {cv}");
                }
            }
        }
    }

    #[test]
    fn dense_crossings_are_all_found_across_blocks_and_windows() {
        // Troughs every 40 samples, plus a smaller one 7 samples before a regular trough on each
        // channel: the larger one wins, whatever the window split
        let (channels, samples, refrac) = (3usize, 3_001usize, 10usize);
        let mut trace = vec![0.0f32; channels * samples];
        for ch in 0..channels {
            for t in (50 + ch..samples - 50).step_by(40) {
                trace[ch * samples + t] = -100.0;
            }
            trace[ch * samples + 1_003] = -90.0;
        }
        let client = WgpuRuntime::client(&WgpuDevice::default());
        let trace_h = client.create_from_slice(f32::as_bytes(&trace));
        let heights_h = client.create_from_slice(f32::as_bytes(&detection_heights(&[5.0f32; 3], 5.0)));
        let detect = |start: usize, end: usize| {
            execute_detect_spikes_in_vram::<WgpuRuntime>(
                &client, &trace_h, &heights_h, channels, samples, start..end, 0, SpikePolarity::Negative, SpikeSpacing::new(refrac),
            )
        };
        let full = detect(0, samples);
        let expected: Vec<(usize, u64)> = {
            let mut v: Vec<(usize, u64)> = (0..channels)
                .flat_map(|ch| {
                    (50 + ch..samples - 50)
                        .step_by(40)
                        .map(move |t| (ch, t as u64))
                })
                .collect();
            v.sort_by_key(|&(ch, t)| (t, ch));
            v
        };
        assert_eq!(
            full.iter()
                .map(|e| (e.channel_id, e.sample_index))
                .collect::<Vec<_>>(),
            expected
        );

        let mut split = Vec::new();
        for w in [0usize, 700, 1_005, 1_900, samples].windows(2) {
            split.extend(detect(w[0], w[1]));
        }
        split.sort_by_key(|s| (s.sample_index, s.channel_id));
        assert_eq!(split, full);
    }

    #[test]
    fn template_reduction_is_segmented_and_free_of_f32_cancellation() {
        let (channels, k, len) = (2usize, 1usize, 3usize);
        let primaries = [1u32, 0, 1, 7, 0, 1];
        let snippets: Vec<f32> = primaries
            .iter()
            .enumerate()
            .flat_map(|(i, _)| (0..len).map(move |t| 10_000.0 + (i as f32) + t as f32 * 0.5))
            .collect();
        let client = WgpuRuntime::client(&WgpuDevice::default());
        let snip_h = client.create_from_slice(f32::as_bytes(&snippets));
        let stats = execute_reduce_templates_in_vram::<WgpuRuntime>(
            &client, &snip_h, &primaries, channels, k, len,
        );
        assert_eq!(stats.counts, vec![2, 3]);
        for ch in 0..channels {
            let members: Vec<usize> = (0..primaries.len())
                .filter(|&i| primaries[i] == ch as u32)
                .collect();
            for t in 0..len {
                let vals: Vec<f64> = members
                    .iter()
                    .map(|&i| snippets[i * len + t] as f64)
                    .collect();
                let mean = vals.iter().sum::<f64>() / vals.len() as f64;
                let m2: f64 = vals.iter().map(|v| (v - mean).powi(2)).sum();
                assert!(
                    (stats.mean[ch * len + t] as f64 - mean).abs() < 1e-3,
                    "mean ch {ch} t {t}"
                );
                assert!(
                    (stats.m2[ch * len + t] as f64 - m2).abs() < 1e-2,
                    "m2 ch {ch} t {t}: {} vs {m2}",
                    stats.m2[ch * len + t]
                );
            }
        }
    }
}
