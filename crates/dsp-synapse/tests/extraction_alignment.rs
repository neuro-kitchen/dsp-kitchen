//! Snippet realignment: troughs at fractional positions land on the snippet centre, CPU and GPU,
//! with full sinc taps and safe handling of spikes at the buffer edges.

use cubecl::prelude::*;
use cubecl::device::WgpuDevice;
use dsp_synapse::detection::DeduplicatedSpike;
use dsp_synapse::extraction::{
    extract_snippet_batch_multichannel, extract_snippets_multichannel, snippet_fits,
};
use dsp_base::math::parabolic_vertex_offset;
use dsp_synapse::extraction::execute_extract_sinc_in_vram;
use dsp_io::neuro::probe::{precompute_knn_table, tetrode};

const CHANNELS: usize = 4;
const SAMPLES: usize = 4_000;
const PRE: usize = 20;
const POST: usize = 40;
const K: usize = 4;
const SIGMA: f32 = 3.0;

fn trough(t: f32) -> f32 {
    -100.0 * (-0.5 * (t / SIGMA).powi(2)).exp()
}

/// Troughs at `centers[i] + frac[i]` on every channel (scaled per channel).
fn trace_with_troughs(positions: &[f32]) -> Vec<f32> {
    let mut trace = vec![0.0f32; CHANNELS * SAMPLES];
    for ch in 0..CHANNELS {
        let gain = 1.0 / (1.0 + ch as f32);
        for &p in positions {
            for t in 0..SAMPLES {
                trace[ch * SAMPLES + t] += gain * trough(t as f32 - p);
            }
        }
    }
    trace
}

fn spike(center: usize) -> DeduplicatedSpike {
    DeduplicatedSpike {
        primary_channel: 0,
        sample_index: center as u64,
        peak_amplitude_uv: -100.0,
        participating_channels: vec![0],
    }
}

/// Sub-sample trough position in a snippet row relative to the centre sample `PRE`.
fn residual(row: &[f32]) -> f32 {
    let (i, _) = row.iter().enumerate().min_by(|a, b| a.1.total_cmp(b.1)).unwrap();
    i as f32 + parabolic_vertex_offset(row[i - 1], row[i], row[i + 1]) - PRE as f32
}

fn fractions() -> Vec<f32> {
    (-3..=3).map(|k| k as f32 * 0.15).collect()
}

fn setup() -> (Vec<f32>, Vec<DeduplicatedSpike>, Vec<f32>) {
    let fracs = fractions();
    let positions: Vec<f32> = fracs.iter().enumerate().map(|(i, f)| 300.0 + 400.0 * i as f32 + f).collect();
    let spikes = positions.iter().map(|p| spike(p.round() as usize)).collect();
    (trace_with_troughs(&positions), spikes, fracs)
}

#[test]
fn cpu_extractors_put_trough_on_centre() {
    let (trace, spikes, fracs) = setup();
    let layout = tetrode();
    let snippets = extract_snippets_multichannel(&trace, CHANNELS, SAMPLES, &spikes, &layout, K, PRE, POST, true);
    let batch = extract_snippet_batch_multichannel(&trace, CHANNELS, SAMPLES, &spikes, &layout, K, PRE, POST, true);
    assert_eq!(snippets.len(), fracs.len());
    assert_eq!(batch.num_spikes, fracs.len());
    for (i, (snip, f)) in snippets.iter().zip(&fracs).enumerate() {
        // Row 0 is the primary channel (nearest neighbour of itself).
        let r = residual(&snip.waveform[..PRE + POST]);
        assert!(r.abs() < 0.02, "trough at +{f}: residual {r} after realignment");
        assert_eq!(&batch.snippet_slice(i)[..], &snip.waveform[..]);
    }
    // Without realignment the residual is the fractional offset itself.
    let raw = extract_snippets_multichannel(&trace, CHANNELS, SAMPLES, &spikes, &layout, K, PRE, POST, false);
    let worst = raw.iter().map(|s| residual(&s.waveform[..PRE + POST]).abs()).fold(0.0, f32::max);
    assert!(worst > 0.4, "unaligned residual {worst}");
}

fn gpu_extract(trace: &[f32], spikes: &[DeduplicatedSpike]) -> Option<(Vec<f32>, Vec<usize>, usize)> {
    let client = dsp_core::compute::open_device(cubecl::Device::Wgpu(WgpuDevice::default()));
    let trace_h = client.create_from_slice(f32::as_bytes(trace));
    let knn_h = client.create_from_slice(u32::as_bytes(&precompute_knn_table(&tetrode(), CHANNELS, K)));
    let out = execute_extract_sinc_in_vram::<f32>(
        &client, &trace_h, &knn_h, CHANNELS, SAMPLES, spikes, K, PRE, POST, true,
    )?;
    let data = f32::from_bytes(&client.read_one_unchecked(out.snippets.clone())).to_vec();
    Some((data, out.kept, out.dropped))
}

#[test]
fn gpu_kernel_matches_cpu_and_centres_trough() {
    let (trace, spikes, fracs) = setup();
    let cpu = extract_snippets_multichannel(&trace, CHANNELS, SAMPLES, &spikes, &tetrode(), K, PRE, POST, true);
    let (gpu, kept, dropped) = gpu_extract(&trace, &spikes).unwrap();
    assert_eq!((kept.len(), dropped), (fracs.len(), 0));
    let stride = K * (PRE + POST);
    for (i, snip) in cpu.iter().enumerate() {
        let g = &gpu[i * stride..(i + 1) * stride];
        let err = g.iter().zip(&snip.waveform).fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
        assert!(err < 1e-3, "spike {i}: GPU vs CPU max error {err}");
        assert!(residual(&g[..PRE + POST]).abs() < 0.02);
    }
}

#[test]
fn edge_spikes_are_skipped_and_counted() {
    let trace = trace_with_troughs(&[1_000.0]);
    let centers = [0, 3, PRE, PRE + 5, 1_000, SAMPLES - POST - 5, SAMPLES - POST, SAMPLES - 2, SAMPLES - 1];
    let spikes: Vec<_> = centers.iter().map(|&c| spike(c)).collect();
    let expected: Vec<usize> = centers
        .iter()
        .enumerate()
        .filter(|(_, c)| snippet_fits(**c, PRE, POST, SAMPLES, true))
        .map(|(i, _)| i)
        .collect();
    assert_eq!(expected, vec![3, 4, 5]);

    let cpu = extract_snippets_multichannel(&trace, CHANNELS, SAMPLES, &spikes, &tetrode(), K, PRE, POST, true);
    let cpu_centers: Vec<usize> = cpu.iter().map(|s| s.center_sample as usize).collect();
    assert_eq!(cpu_centers, expected.iter().map(|&i| centers[i]).collect::<Vec<_>>());

    let (_, kept, dropped) = gpu_extract(&trace, &spikes).unwrap();
    assert_eq!(kept, expected);
    assert_eq!(dropped, centers.len() - expected.len());
}

#[test]
fn every_snippet_sample_uses_full_sinc_taps() {
    // A band-limited trace: realigned snippets must equal the signal at the shifted times, including
    // the first and last samples (a truncated kernel there is off by percent-level errors).
    let freq = 1_000.0f32 / 30_000.0;
    let signal = |t: f32| 50.0 * (2.0 * std::f32::consts::PI * freq * t).sin() + 20.0 * (2.0 * std::f32::consts::PI * 2.3 * freq * t + 0.4).cos();
    let mut trace = vec![0.0f32; CHANNELS * SAMPLES];
    for ch in 0..CHANNELS {
        for t in 0..SAMPLES {
            trace[ch * SAMPLES + t] = signal(t as f32);
        }
    }
    // Force a known sub-sample offset on the primary channel's parabola around the centre.
    let center = 2_000;
    let (a, b, c) = (-90.0f32, -100.0, -80.0);
    trace[center - 1] = a;
    trace[center] = b;
    trace[center + 1] = c;
    let delta = parabolic_vertex_offset(a, b, c);

    let snip = &extract_snippets_multichannel(&trace, CHANNELS, SAMPLES, &[spike(center)], &tetrode(), K, PRE, POST, true)[0];
    // Rows 1.. are other channels (untouched by the forced parabola).
    let row = &snip.waveform[(PRE + POST)..2 * (PRE + POST)];
    let err = row
        .iter()
        .enumerate()
        .map(|(i, v)| (v - signal((center - PRE + i) as f32 + delta)).abs())
        .fold(0.0f32, f32::max);
    assert!(err < 0.05, "max interpolation error {err} µV (signal amplitude 70 µV)");
}
