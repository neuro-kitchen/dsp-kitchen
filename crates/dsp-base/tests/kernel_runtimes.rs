//! Every dsp-base dispatcher gives the host-reference result on every compiled-in CubeCL runtime
//! (one launch geometry, no device-specific paths). Run with `--features cpu` to include the CPU
//! runtime next to WGPU.

use cubecl::prelude::*;
use dsp_base::filter::{execute_fir, execute_median_9p, execute_teager_kaiser};
use dsp_base::math::{execute_baseline_subtract, execute_clamp, execute_scaling, execute_unpack_stored, stored_words};
use dsp_core::SampleFormat;
use dsp_base::pipeline::{Pipeline, PipelineStage};
use dsp_base::spatial::execute_direct_car;
use dsp_core::compute::{ComputeTarget, ComputeTask};

const CHANNELS: usize = 37;
const SAMPLES: usize = 5_003;

fn signal() -> Vec<f32> {
    (0..CHANNELS * SAMPLES)
        .map(|i| {
            let (c, t) = ((i / SAMPLES) as f32, (i % SAMPLES) as f32);
            (t * 0.37 + c).sin() * 50.0 + ((i * 7919) % 101) as f32 - 50.0 + c * 3.0
        })
        .collect()
}

fn max_err(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).fold(0.0f32, |m, (x, y)| m.max((x - y).abs()))
}

fn median9(w: &[f32]) -> f32 {
    let mut v = w.to_vec();
    v.sort_by(f32::total_cmp);
    v[4]
}

struct Check;

impl ComputeTask for Check {
    type Output = ();

    fn run<R: Runtime>(self, client: ComputeClient<R>) {
        let name = R::name(&client).to_string();
        let x = signal();
        let n = x.len();
        let input = client.create_from_slice(f32::as_bytes(&x));
        let output = client.empty(n * 4);
        let read = |h: &cubecl::server::Handle| f32::from_bytes(&client.read_one_unchecked(h.clone())).to_vec();

        execute_scaling::<R>(&client, &input, &output, n, 0.5, -3.0);
        let expected: Vec<f32> = x.iter().map(|v| v * 0.5 - 3.0).collect();
        assert!(max_err(&read(&output), &expected) < 1e-4, "{name}: scaling");

        execute_clamp::<R>(&client, &input, &output, n, -20.0, 30.0);
        let expected: Vec<f32> = x.iter().map(|v| v.clamp(-20.0, 30.0)).collect();
        assert!(max_err(&read(&output), &expected) == 0.0, "{name}: clamp");

        let baselines: Vec<f32> = (0..CHANNELS).map(|c| c as f32 * 3.0).collect();
        let base_h = client.create_from_slice(f32::as_bytes(&baselines));
        execute_baseline_subtract::<R>(&client, &input, &output, &base_h, CHANNELS, SAMPLES);
        let expected: Vec<f32> = x.iter().enumerate().map(|(i, v)| v - baselines[i / SAMPLES]).collect();
        assert!(max_err(&read(&output), &expected) < 1e-4, "{name}: baseline");

        execute_direct_car::<R>(&client, &input, &output, CHANNELS, SAMPLES);
        let means: Vec<f32> = (0..SAMPLES)
            .map(|t| (0..CHANNELS).map(|c| x[c * SAMPLES + t]).sum::<f32>() / CHANNELS as f32)
            .collect();
        let expected: Vec<f32> = x.iter().enumerate().map(|(i, v)| v - means[i % SAMPLES]).collect();
        assert!(max_err(&read(&output), &expected) < 1e-3, "{name}: CAR");

        execute_median_9p::<R>(&client, &input, &output, CHANNELS, SAMPLES);
        let got = read(&output);
        for c in [0, CHANNELS / 2, CHANNELS - 1] {
            let row = &x[c * SAMPLES..(c + 1) * SAMPLES];
            for t in [0, 3, 4, 2_000, SAMPLES - 5, SAMPLES - 1] {
                let want = if (4..SAMPLES - 4).contains(&t) { median9(&row[t - 4..t + 5]) } else { row[t] };
                assert_eq!(got[c * SAMPLES + t], want, "{name}: median ch {c} t {t}");
            }
        }

        execute_teager_kaiser::<R>(&client, &input, &output, CHANNELS, SAMPLES);
        let got = read(&output);
        for c in [0, CHANNELS - 1] {
            let row = &x[c * SAMPLES..(c + 1) * SAMPLES];
            for t in [1, 100, SAMPLES - 2] {
                let want = (row[t] * row[t] - row[t - 1] * row[t + 1]).max(0.0);
                assert!((got[c * SAMPLES + t] - want).abs() <= 1e-3 * want.abs().max(1.0), "{name}: TKEO");
            }
        }

        let taps = [0.25f32, 0.5, 0.25];
        let taps_h = client.create_from_slice(f32::as_bytes(&taps));
        execute_fir::<R>(&client, &input, &output, &taps_h, CHANNELS, SAMPLES, taps.len());
        let got = read(&output);
        for c in [0, CHANNELS - 1] {
            let row = &x[c * SAMPLES..(c + 1) * SAMPLES];
            for t in [0, 1, 2, 4_000] {
                let want: f32 = (0..3).filter(|k| t >= *k).map(|k| taps[k] * row[t - k]).sum();
                assert!((got[c * SAMPLES + t] - want).abs() < 1e-3, "{name}: FIR");
            }
        }

        // Elementwise geometry beyond one cube-count dimension (> 65 535 × 256 elements).
        let big = 20_000_000usize;
        let ones = client.create_from_slice(f32::as_bytes(&vec![1.0f32; big]));
        let big_out = client.empty(big * 4);
        execute_scaling::<R>(&client, &ones, &big_out, big, 2.0, 1.0);
        let got = read(&big_out);
        assert!(got.iter().all(|v| *v == 3.0), "{name}: large elementwise launch left elements unwritten");

        // Stored samples → µV: every device format, extremes, an odd count (partial last word)
        let (ch, n) = (3usize, 7usize);
        let gains = [0.5f32, 2.0, 1.0];
        let offsets = [1.0f32, -3.0, 0.0];
        let (gains_h, offsets_h) = (client.create_from_slice(f32::as_bytes(&gains)), client.create_from_slice(f32::as_bytes(&offsets)));
        let ints: [i64; 7] = [0, 1, -1, 100, -100, i64::MIN, i64::MAX];
        for format in [SampleFormat::I8, SampleFormat::I16, SampleFormat::U16, SampleFormat::I32, SampleFormat::F32] {
            let values: Vec<f64> = (0..ch * n)
                .map(|i| {
                    let v = ints[i % n];
                    match format {
                        SampleFormat::I8 => v.clamp(i8::MIN as i64, i8::MAX as i64) as f64,
                        SampleFormat::I16 => v.clamp(i16::MIN as i64, i16::MAX as i64) as f64,
                        SampleFormat::U16 => v.clamp(0, u16::MAX as i64) as f64,
                        SampleFormat::I32 => v.clamp(i32::MIN as i64, i32::MAX as i64) as f64,
                        _ => v.clamp(-1_000_000, 1_000_000) as f64 + 0.25,
                    }
                })
                .collect();
            let bytes: Vec<u8> = values
                .iter()
                .flat_map(|&v| match format {
                    SampleFormat::I8 => (v as i8).to_le_bytes().to_vec(),
                    SampleFormat::I16 => (v as i16).to_le_bytes().to_vec(),
                    SampleFormat::U16 => (v as u16).to_le_bytes().to_vec(),
                    SampleFormat::I32 => (v as i32).to_le_bytes().to_vec(),
                    _ => (v as f32).to_le_bytes().to_vec(),
                })
                .collect();
            let words = client.create_from_slice(u32::as_bytes(&stored_words(&bytes)));
            let out = client.empty(ch * n * 4);
            execute_unpack_stored::<R>(&client, &words, format, &gains_h, &offsets_h, &out, ch, n).unwrap();
            let got = read(&out);
            for (i, v) in values.iter().enumerate() {
                let want = (*v as f32) * gains[i / n] + offsets[i / n];
                assert_eq!(got[i], want, "{name}: {format:?} value {i} ({v})");
            }
        }
        assert!(execute_unpack_stored::<R>(&client, &input, SampleFormat::F64, &gains_h, &offsets_h, &output, ch, n).is_err());

        // Pipeline with filter and stencil stages.
        let pipeline = Pipeline::with_stages(vec![
            PipelineStage::Scale { alpha: 1.0, beta: 0.0 },
            PipelineStage::bandpass(300.0, 6000.0),
            PipelineStage::CommonAverageReference,
            PipelineStage::Median9p,
        ]);
        let out = pipeline.execute::<R>(&client, &input, CHANNELS, SAMPLES, 30_000.0).unwrap();
        assert!(read(&out).iter().all(|v| v.is_finite()), "{name}: pipeline");
    }
}

#[test]
fn dispatchers_match_host_reference_on_every_runtime() {
    let targets = ComputeTarget::available();
    assert!(!targets.is_empty());
    for target in targets {
        target.run(Check).unwrap();
    }
}
