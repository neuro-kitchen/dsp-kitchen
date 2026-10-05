//! The time-block-parallel SOS pass matches the host f64 reference for any block length (one
//! block, single-step blocks, lengths that do not divide the signal), for forward, zero-phase and
//! stateful streaming passes, from rest and from the steady state, in channel-major and time-major
//! memory order, on every compiled-in runtime.

mod common;

use common::*;
use cubecl::prelude::*;
use dsp_base::filter::{DeviceFilter, FilterBand, FilterMode, FilterSpec, FilterStart, PassLayout};
use dsp_core::compute::{ComputeTarget, ComputeTask};

const FS: f64 = 30_000.0;
const CHANNELS: usize = 3;
const BLOCKS: [usize; 6] = [usize::MAX, 1, 7, 64, 1_000, 4_096];

fn specs() -> Vec<FilterSpec> {
    vec![
        FilterSpec::butterworth(5, FilterBand::Bandpass(300.0, 6000.0)),
        // Poles next to z = 1: block start states carry most of the signal
        FilterSpec::butterworth(3, FilterBand::Highpass(0.5)),
        FilterSpec::notch(60.0, 30.0),
    ]
}

fn upload<R: Runtime>(client: &ComputeClient<R>, x: &[f64]) -> cubecl::server::Handle {
    let data: Vec<f32> = (0..CHANNELS).flat_map(|_| x.iter().map(|v| *v as f32)).collect();
    client.create_from_slice(f32::as_bytes(&data))
}

fn read<R: Runtime>(client: &ComputeClient<R>, h: cubecl::server::Handle) -> Vec<f64> {
    f32::from_bytes(&client.read_one_unchecked(h)).iter().map(|v| *v as f64).collect()
}

fn assert_close(name: &str, got: &[f64], expected: &[f64], scale: f64) {
    let n = expected.len();
    for c in 0..CHANNELS {
        let err = max_abs_diff(&got[c * n..(c + 1) * n], expected);
        assert!(err < 2e-5 * scale, "{name} channel {c}: max error {err}");
    }
}

struct Check;

impl ComputeTask for Check {
    type Output = ();

    fn run<R: Runtime>(self, client: ComputeClient<R>) {
        let rt = R::name(&client).to_string();
        let x = test_signal(20_003, FS);
        let n = x.len();
        let scale = max_abs(&x);
        let input = upload(&client, &x);

        for (spec, layout) in specs().into_iter().flat_map(|s| [PassLayout::ChannelMajor, PassLayout::TimeMajor].map(|l| (s.clone(), l))) {
            for block in BLOCKS {
                for (mode, start) in [
                    (FilterMode::Forward, FilterStart::Rest),
                    (FilterMode::Forward, FilterStart::SteadyState),
                    (FilterMode::ForwardBackward, FilterStart::Rest),
                ] {
                    let spec = spec.clone().with_mode(mode).with_start(start);
                    let sos = spec.design(FS).unwrap();
                    let filter = DeviceFilter::<f32>::new(&client, &spec, FS).unwrap().with_block_len(block).with_layout(layout);
                    let output = client.empty(CHANNELS * n * 4);
                    let scratch = client.empty((filter.scratch_len(CHANNELS, n) * 4).max(4));
                    let state = client.empty(CHANNELS * filter.state_len() * 4);
                    filter.apply(&client, &input, &output, &scratch, &state, CHANNELS, n);
                    let expected = match mode {
                        FilterMode::Forward => sos.filter(&x, start == FilterStart::SteadyState),
                        FilterMode::ForwardBackward => sos.filtfilt(&x, sos.settling_samples(1e-3).min(n - 1)),
                    };
                    assert_close(&format!("{rt} {spec:?} {layout:?} block {block}"), &read(&client, output), &expected, scale);
                }

                // Stateful chunks continue the carried state across block-parallel passes
                for start in [FilterStart::Rest, FilterStart::SteadyState] {
                    let spec = spec.clone().with_mode(FilterMode::Forward).with_start(start);
                    let sos = spec.design(FS).unwrap();
                    let filter = DeviceFilter::<f32>::new(&client, &spec, FS).unwrap().with_block_len(block).with_layout(layout);
                    let state = client.empty(CHANNELS * filter.state_len() * 4);
                    let mut got = vec![0.0f64; CHANNELS * n];
                    for (i, w) in [0usize, 3_001, 7_777, n].windows(2).enumerate() {
                        let len = w[1] - w[0];
                        let chunk: Vec<f64> = x[w[0]..w[1]].to_vec();
                        let output = client.empty(CHANNELS * len * 4);
                        filter.apply_stateful(&client, &upload(&client, &chunk), &output, &state, CHANNELS, len, i == 0).unwrap();
                        let part = read(&client, output);
                        for c in 0..CHANNELS {
                            got[c * n + w[0]..c * n + w[1]].copy_from_slice(&part[c * len..(c + 1) * len]);
                        }
                    }
                    let expected = sos.filter(&x, start == FilterStart::SteadyState);
                    assert_close(&format!("{rt} stateful {spec:?} {layout:?} block {block}"), &got, &expected, scale);
                }
            }
        }

        // Autotuned block length gives the same result
        let spec = specs().remove(0);
        let sos = spec.design(FS).unwrap();
        let filter = DeviceFilter::<f32>::new(&client, &spec, FS).unwrap();
        let output = client.empty(CHANNELS * n * 4);
        let scratch = client.empty((filter.scratch_len(CHANNELS, n) * 4).max(4));
        let state = client.empty(CHANNELS * filter.state_len() * 4);
        filter.apply(&client, &input, &output, &scratch, &state, CHANNELS, n);
        assert_close(&format!("{rt} autotuned"), &read(&client, output), &sos.filtfilt(&x, sos.settling_samples(1e-3).min(n - 1)), scale);
    }
}

#[test]
fn block_parallel_pass_matches_reference_on_every_runtime() {
    for target in ComputeTarget::available() {
        target.run(Check).unwrap();
    }
}
