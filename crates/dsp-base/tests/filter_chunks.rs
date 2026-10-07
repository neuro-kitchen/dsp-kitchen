//! Chunk invariance: whole recording vs halo windows vs stateful chunks, and the settling model
//! vs measured impulse-response decay.

// Runs on the WGPU runtime (default feature); every-runtime coverage lives in filter_runtimes,
// filter_blocks and kernel_runtimes.
#![cfg(feature = "wgpu")]

mod common;

use common::*;
use cubecl::device::WgpuDevice;
use dsp_base::filter::{FilterMode, FilterSpec, FilterStart};
use dsp_base::pipeline::{Pipeline, PipelineStage, PipelineWorkspace};

const FS: f64 = 30_000.0;
const CHANNELS: usize = 4;

fn recording(n: usize) -> Vec<f32> {
    let base = test_signal(n, FS);
    (0..CHANNELS)
        .flat_map(|c| base.iter().enumerate().map(move |(i, v)| (*v * (1.0 + 0.25 * c as f64) + ((i * 7919 + c * 104_729) % 97) as f64 * 0.5) as f32))
        .collect()
}

fn workspace(pipeline: &Pipeline, samples: usize, stateful: bool) -> PipelineWorkspace<f32> {
    let client = cubecl::Device::Wgpu(WgpuDevice::default()).client();
    if stateful {
        PipelineWorkspace::new_stateful(client, pipeline.clone(), CHANNELS, samples, FS).unwrap()
    } else {
        PipelineWorkspace::new(client, pipeline.clone(), CHANNELS, samples, FS).unwrap()
    }
}

fn whole(pipeline: &Pipeline, x: &[f32], n: usize) -> Vec<f32> {
    let mut out = vec![0.0; x.len()];
    workspace(pipeline, n, false).process_chunk(x, n, &mut out);
    out
}

fn slice(x: &[f32], n: usize, range: std::ops::Range<usize>) -> Vec<f32> {
    (0..CHANNELS).flat_map(|c| x[c * n + range.start..c * n + range.end].iter().copied()).collect()
}

/// Independent halo windows of `batch` samples, halos from `Pipeline::settling`.
fn halo_windows(pipeline: &Pipeline, x: &[f32], n: usize, batch: usize) -> Vec<f32> {
    let (left, right) = pipeline.settling(FS).unwrap();
    let mut ws = workspace(pipeline, batch + left + right, false);
    let mut out = vec![0.0; x.len()];
    let mut start = 0;
    while start < n {
        let end = (start + batch).min(n);
        let (rs, re) = (start.saturating_sub(left), (end + right).min(n));
        let len = re - rs;
        let chunk = slice(x, n, rs..re);
        let mut y = vec![0.0; chunk.len()];
        ws.process_chunk(&chunk, len, &mut y);
        for c in 0..CHANNELS {
            let src = &y[c * len + (start - rs)..c * len + (end - rs)];
            out[c * n + start..c * n + end].copy_from_slice(src);
        }
        start = end;
    }
    out
}

fn stateful_chunks(pipeline: &Pipeline, x: &[f32], n: usize, batch: usize) -> Vec<f32> {
    let mut ws = workspace(pipeline, batch, true);
    let mut out = vec![0.0; x.len()];
    let mut start = 0;
    while start < n {
        let end = (start + batch).min(n);
        let chunk = slice(x, n, start..end);
        let mut y = vec![0.0; chunk.len()];
        ws.process_chunk(&chunk, end - start, &mut y);
        for c in 0..CHANNELS {
            out[c * n + start..c * n + end].copy_from_slice(&y[c * (end - start)..(c + 1) * (end - start)]);
        }
        start = end;
    }
    out
}

fn max_err(a: &[f32], b: &[f32]) -> f64 {
    a.iter().zip(b).fold(0.0f64, |m, (x, y)| m.max((x - y).abs() as f64))
}

fn amplitude(x: &[f32]) -> f64 {
    x.iter().fold(0.0f64, |m, v| m.max(v.abs() as f64))
}

fn filters() -> Vec<(&'static str, FilterSpec, usize)> {
    vec![
        ("bandpass 300-6000 order 5", FilterSpec::bandpass(300.0, 6000.0), 60_000),
        ("highpass 300 order 5", FilterSpec::highpass(300.0), 60_000),
        ("notch 60 Hz Q30", FilterSpec::notch(60.0, 30.0), 240_000),
    ]
}

#[test]
fn halo_windows_match_whole_recording() {
    for (name, spec, n) in filters() {
        // Forward windows start from the steady state: from rest each window would also absorb the
        // step to its DC level, which the settling model (impulse tail) does not bound tightly.
        for mode in [FilterMode::Forward, FilterMode::ForwardBackward] {
            let spec = spec.clone().with_mode(mode).with_start(FilterStart::SteadyState);
            let pipeline = Pipeline::with_stages(vec![PipelineStage::Filter(spec)]);
            let x = recording(n);
            let reference = whole(&pipeline, &x, n);
            let amp = amplitude(&x);
            for batch in [4_096, 7_919, 30_000] {
                let err = max_err(&halo_windows(&pipeline, &x, n, batch), &reference);
                assert!(err < 5e-4 * amp, "{name} {mode:?} batch {batch}: max error {err} (amplitude {amp})");
            }
        }
    }
}

#[test]
fn stateful_chunks_are_exact_for_forward_filters() {
    for (name, spec, n) in filters() {
        let pipeline = Pipeline::with_stages(vec![PipelineStage::Filter(spec.with_mode(FilterMode::Forward))]);
        let x = recording(n);
        let reference = whole(&pipeline, &x, n);
        let amp = amplitude(&x);
        for batch in [1_000, 7_919, 30_000] {
            let err = max_err(&stateful_chunks(&pipeline, &x, n, batch), &reference);
            assert!(err < 1e-5 * amp, "{name} batch {batch}: max error {err}");
        }
    }
}

#[test]
fn stateful_workspace_rejects_forward_backward() {
    let pipeline = Pipeline::with_stages(vec![PipelineStage::bandpass(300.0, 6000.0)]);
    let client = cubecl::Device::Wgpu(WgpuDevice::default()).client();
    assert!(PipelineWorkspace::<f32>::new_stateful(client, pipeline, 2, 100, FS).is_err());
}

#[test]
fn dc_offset_gives_no_step_transient() {
    // A constant −500 µV input through a high-pass must come out as ~0 from the first sample when
    // passes start from the steady state (forward-backward always does).
    let n = 3_000;
    let x = vec![-500.0f32; CHANNELS * n];
    for mode in [FilterMode::Forward, FilterMode::ForwardBackward] {
        let spec = FilterSpec::highpass(300.0).with_mode(mode).with_start(FilterStart::SteadyState);
        let pipeline = Pipeline::with_stages(vec![PipelineStage::Filter(spec)]);
        let y = whole(&pipeline, &x, n);
        assert!(amplitude(&y) < 1e-2, "{mode:?}: residual {}", amplitude(&y));
    }
}

#[test]
fn settling_is_the_impulse_tail_criterion() {
    let tol = 1e-3;
    for (name, spec, _) in filters() {
        let sos = spec.design(FS).unwrap();
        let t = sos.settling_samples(tol);
        let bound = sos.pole_settling_bound(tol);
        assert!(t > 0 && t <= bound, "{name}: settling {t}, pole bound {bound}");
        let mut impulse = vec![0.0; 2 * bound + 64];
        impulse[0] = 1.0;
        let h: Vec<f64> = sos.filter(&impulse, false).iter().map(|v| v.abs()).collect();
        let total: f64 = h.iter().sum();
        let tail = |from: usize| h[from..].iter().sum::<f64>();
        assert!(tail(t) <= tol * total, "{name}: tail after {t} too large");
        assert!(tail(t - 1) > tol * total, "{name}: {t} is not the first settled sample");
    }
}

#[test]
fn forward_from_rest_has_the_sosfilt_step_transient() {
    // scipy sosfilt without zi: a constant input starts from zero output and a DC-blocking filter
    // shows the step, decaying to ~0.
    let n = 3_000;
    let x = vec![-500.0f32; CHANNELS * n];
    let spec = FilterSpec::highpass(300.0).with_mode(FilterMode::Forward);
    let sos = spec.design(FS).unwrap();
    let pipeline = Pipeline::with_stages(vec![PipelineStage::Filter(spec)]);
    let y = whole(&pipeline, &x, n);
    let expected = sos.filter(&vec![-500.0f64; n], false);
    for c in 0..CHANNELS {
        let err = y[c * n..(c + 1) * n].iter().zip(&expected).fold(0.0f64, |m, (a, b)| m.max((*a as f64 - b).abs()));
        assert!(err < 1e-2, "channel {c}: max error {err}");
    }
    assert!(y[0].abs() > 100.0, "the step must show at the first sample");
}
