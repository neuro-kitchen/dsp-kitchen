//! CubeCL GPU/SIMT kernels for in-VRAM threshold spike detection, Blackman-Harris sinc snippet
//! extraction, and per-channel template moment reduction.

pub mod threshold;
pub mod extract_sinc;
pub mod template_reduce;

pub use threshold::{detect_channel_troughs_kernel, execute_detect_spikes_in_vram};
pub use extract_sinc::{extract_sinc_snippets_kernel, execute_extract_sinc_in_vram};
pub use template_reduce::{
    BatchTemplateStats, reduce_channel_templates_kernel, execute_reduce_templates_in_vram,
};

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
    use cubecl::{CubeElement, Runtime};
    use dsp_core::layout::{Position3D, SensorLayout, SensorSite};
    use crate::detection::{deduplicate_spikes_spatial, detect_spikes_with_sigma};
    use crate::extraction::extract_snippets_multichannel;
    use crate::probe::precompute_knn_table;
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

        // Inject synthetic spikes on ch 0 and ch 2
        for &center in &[300usize, 800, 1400] {
            for dt in -10isize..=20 {
                let s = (center as isize + dt) as usize;
                let x = (dt as f32 - 0.25) / 3.0;
                let wave = (-0.5 * x * x).exp() * -95.0;
                trace[center.min(0) * samples + s] += wave;
                trace[1 * samples + s] += wave * 0.35;
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

        // CPU Reference
        let cpu_spikes = detect_spikes_with_sigma(
            &trace,
            channels,
            samples,
            &sigmas,
            threshold_factor,
            refrac,
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
        let mut cpu_accs: Vec<TemplateAccumulator> = (0..channels)
            .map(|_| TemplateAccumulator::new(k_neighbors, snippet_len))
            .collect();
        for snip in &cpu_snips {
            cpu_accs[snip.primary_channel].update(snip);
        }

        // GPU / CubeCL VRAM execution
        let device = WgpuDevice::default();
        let client = WgpuRuntime::client(&device);
        let trace_handle = client.create_from_slice(f32::as_bytes(&trace));
        let sigmas_handle = client.create_from_slice(f32::as_bytes(&sigmas));
        let knn_table = precompute_knn_table(&probe, k_neighbors);
        let knn_handle = client.create_from_slice(u32::as_bytes(&knn_table));

        let gpu_spikes = execute_detect_spikes_in_vram::<WgpuRuntime>(
            &client,
            &trace_handle,
            &sigmas_handle,
            channels,
            samples,
            1,
            samples - 1,
            threshold_factor,
            refrac,
            256,
            false,
        );
        assert_eq!(gpu_spikes, cpu_spikes);

        let gpu_dedup = deduplicate_spikes_spatial(&gpu_spikes, &probe, 150.0, refrac as u64);
        let (snips_handle, prim_handle) = execute_extract_sinc_in_vram::<WgpuRuntime>(
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
            false,
        )
        .unwrap();

        let batch_stats = execute_reduce_templates_in_vram::<WgpuRuntime>(
            &client,
            &snips_handle,
            &prim_handle,
            channels,
            gpu_dedup.len(),
            k_neighbors,
            snippet_len,
            false,
        );

        let elems_per_spike = k_neighbors * snippet_len;
        let mut gpu_accs: Vec<TemplateAccumulator> = (0..channels)
            .map(|_| TemplateAccumulator::new(k_neighbors, snippet_len))
            .collect();
        for ch in 0..channels {
            let cnt = batch_stats.counts[ch] as u64;
            if cnt > 0 {
                let s = ch * elems_per_spike;
                let e = s + elems_per_spike;
                gpu_accs[ch].merge_batch(cnt, &batch_stats.sum[s..e], &batch_stats.sum_sq[s..e]);
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
}
