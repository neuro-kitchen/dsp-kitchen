//! A workspace's fused steps give what its stages give one by one, and a captured pipeline's
//! replay gives what a run does, on every compiled-in runtime.

use cubecl::prelude::*;
use dspk_core::compute::{ComputeTarget, ComputeTask};
use dspk_ops::core::buffer;
use dspk_ops::math::Elementwise;
use dspk_ops::pipeline::PipelineWorkspace;
use dspk_ops::{Pipeline, PipelineStage};

const FS: f64 = 30_000.0;

/// One test on the device at a time: a capture records everything the shared client runs.
static DEVICE: std::sync::Mutex<()> = std::sync::Mutex::new(());
const CHANNELS: usize = 3;
const SAMPLES: usize = 12_000;

fn signal(seed: usize) -> Vec<f32> {
    (0..CHANNELS * SAMPLES).map(|i| ((i % SAMPLES) as f32 * 0.021 + seed as f32).sin() * 80.0 + ((i * 7919 + seed) % 1013) as f32 * 0.03).collect()
}

fn chain() -> Vec<PipelineStage> {
    vec![
        PipelineStage::bandpass(300.0, 6000.0),
        // A run of four pointwise stages: one kernel in the workspace
        PipelineStage::Scale { alpha: 1.5, beta: 0.25 },
        PipelineStage::Elementwise(Elementwise::Abs),
        PipelineStage::SubtractBaseline { baseline: 2.0 },
        PipelineStage::Elementwise(Elementwise::Sqrt),
        PipelineStage::teager_kaiser(),
        // A lone pointwise stage: runs as it is
        PipelineStage::Scale { alpha: 0.5, beta: 0.0 },
    ]
}

fn run(client: &Client, stages: Vec<PipelineStage>, x: &[f32]) -> Vec<f32> {
    let mut ws = PipelineWorkspace::<f32>::new(client.clone(), Pipeline::with_stages(stages), CHANNELS, SAMPLES, FS).unwrap();
    let out = ws.process_handle(&buffer::upload(client, x), SAMPLES);
    buffer::download_prefix(client, out, CHANNELS * SAMPLES)
}

struct FusedEqualsOneByOne;

impl ComputeTask for FusedEqualsOneByOne {
    type Output = ();
    fn run(self, client: Client) {
        let x = signal(0);
        let fused = run(&client, chain(), &x);
        // Each stage in a workspace of its own: nothing to fuse
        let mut one_by_one = x.clone();
        for stage in chain() {
            one_by_one = run(&client, vec![stage], &one_by_one);
        }
        let scale = one_by_one.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1.0);
        for (i, (f, o)) in fused.iter().zip(&one_by_one).enumerate() {
            // The same operations; `x · a + b` may be contracted differently in one kernel than in
            // another: within a few roundings of the signal's scale
            assert!((f - o).abs() <= 8.0 * f32::EPSILON * scale, "{} sample {i}: fused {f}, one by one {o}", client.name());
        }
        let ws = PipelineWorkspace::<f32>::new(client.clone(), Pipeline::with_stages(chain()), CHANNELS, SAMPLES, FS).unwrap();
        assert_eq!(ws.launches(), 4, "band-pass, one fused run, Teager-Kaiser, the lone scale");
    }
}

#[test]
fn fused_pointwise_runs_equal_their_stages_one_by_one() {
    let _device = DEVICE.lock().unwrap_or_else(|e| e.into_inner());
    for target in ComputeTarget::for_tests() {
        target.run(FusedEqualsOneByOne).expect("compiled-in runtime");
    }
}

struct ReplayEqualsRun;

impl ComputeTask for ReplayEqualsRun {
    type Output = ();
    fn run(self, client: Client) {
        let mut ws = PipelineWorkspace::<f32>::new(client.clone(), Pipeline::with_stages(chain()), CHANNELS, SAMPLES, FS).unwrap();
        let input = buffer::upload(&client, &signal(1));
        let captured = match ws.capture(&input, SAMPLES) {
            Ok(c) => c,
            // No graphs on this runtime (the CPU runtime): nothing to replay
            Err(e) => {
                assert!(client.name() == "cpu" || e.contains("graph"), "{}: {e}", client.name());
                return;
            }
        };
        // A new window written into the recorded input, then replayed
        let next = signal(2);
        buffer::write_owned(&client, &input, next.clone());
        // SAFETY: `input` and `ws` live until the read below has completed
        unsafe { captured.replay() }.unwrap();
        let replayed: Vec<f32> = buffer::download_prefix(&client, captured.output().clone(), CHANNELS * SAMPLES);
        let direct = run(&client, chain(), &next);
        assert_eq!(replayed, direct, "{}: a replay is the same work on the same input", client.name());
    }
}

#[test]
fn a_captured_pipeline_replays_what_a_run_computes() {
    let _device = DEVICE.lock().unwrap_or_else(|e| e.into_inner());
    for target in ComputeTarget::for_tests() {
        target.run(ReplayEqualsRun).expect("compiled-in runtime");
    }
}
