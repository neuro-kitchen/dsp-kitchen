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

/// Relative error limit (of the signal's peak) for every pass except the one below.
const TOLERANCE: f64 = 2e-5;

/// Limit for a forward pass **from rest** through the 0.5 Hz high-pass. Starting at rest, the
/// signal's −500 DC level enters as a step whose slow response depends on the exact pole positions,
/// and rounding the coefficients to `f32` moves poles this close to `z = 1` slightly: the `f64`
/// state-variable filter with the same rounded coefficients is already 1.08e-4 off scipy's direct
/// form (measured), while the kernel matches its own arithmetic to ~6e-6. Steady-state starts never
/// excite that transient and keep [`TOLERANCE`].
const REST_LOW_CUTOFF_TOLERANCE: f64 = 2e-4;

fn specs() -> Vec<FilterSpec> {
    vec![
        FilterSpec::butterworth(5, FilterBand::Bandpass(300.0, 6000.0)),
        // Poles next to z = 1: block start states carry most of the signal
        FilterSpec::butterworth(3, FilterBand::Highpass(0.5)),
        FilterSpec::notch(60.0, 30.0),
    ]
}

/// The limit for `spec` started at `start`.
fn tolerance(spec: &FilterSpec, start: FilterStart) -> f64 {
    let low_cutoff = matches!(spec.design, dsp_base::filter::FilterDesign::Butterworth { band: FilterBand::Highpass(f), .. } if f < 1.0);
    if low_cutoff && start == FilterStart::Rest { REST_LOW_CUTOFF_TOLERANCE } else { TOLERANCE }
}

fn upload(client: &Client, x: &[f64]) -> cubecl::server::Handle {
    let data: Vec<f32> = (0..CHANNELS).flat_map(|_| x.iter().map(|v| *v as f32)).collect();
    client.create_from_slice(f32::as_bytes(&data))
}

fn read(client: &Client, h: cubecl::server::Handle) -> Vec<f64> {
    f32::from_bytes(&client.read_one_unchecked(h)).iter().map(|v| *v as f64).collect()
}

fn assert_close(name: &str, got: &[f64], expected: &[f64], scale: f64, tolerance: f64) {
    let n = expected.len();
    for c in 0..CHANNELS {
        let err = max_abs_diff(&got[c * n..(c + 1) * n], expected) / scale;
        assert!(err < tolerance, "{name} channel {c}: max error {err:.2e} of the peak (limit {tolerance:.0e})");
    }
}

struct Check;

impl ComputeTask for Check {
    type Output = ();

    fn run(self, client: Client) {
        let rt = client.name().to_string();
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
                    assert_close(&format!("{rt} {spec:?} {layout:?} block {block}"), &read(&client, output), &expected, scale, tolerance(&spec, start));
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
                    assert_close(&format!("{rt} stateful {spec:?} {layout:?} block {block}"), &got, &expected, scale, tolerance(&spec, start));
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
        assert_close(&format!("{rt} autotuned"), &read(&client, output), &sos.filtfilt(&x, sos.settling_samples(1e-3).min(n - 1)), scale, TOLERANCE);
    }
}

#[test]
fn block_parallel_pass_matches_reference_on_every_runtime() {
    for target in ComputeTarget::available() {
        target.run(Check).unwrap();
    }
}

/// Under `pin_tuned_choices`, an untuned filter runs the documented fixed split
/// (`PINNED_BLOCK_COUNT` blocks, channel-major): bit-identical to setting it explicitly.
#[test]
fn pinned_choices_match_the_fixed_split() {
    use dsp_base::filter::iir::sos::PINNED_BLOCK_COUNT;
    struct Pinned;
    impl ComputeTask for Pinned {
        type Output = ();
        fn run(self, client: Client) {
            let x = test_signal(20_003, FS);
            let n = x.len();
            let input = upload(&client, &x);
            let spec = FilterSpec::butterworth(3, FilterBand::Highpass(0.5)).with_mode(FilterMode::ForwardBackward);
            let run = |filter: DeviceFilter<f32>| {
                let output = client.empty(CHANNELS * n * 4);
                let scratch = client.empty((filter.scratch_len(CHANNELS, n) * 4).max(4));
                let state = client.empty(CHANNELS * filter.state_len() * 4);
                filter.apply(&client, &input, &output, &scratch, &state, CHANNELS, n);
                read(&client, output)
            };
            let pinned = {
                let _pin = dsp_core::compute::pin_tuned_choices();
                run(DeviceFilter::<f32>::new(&client, &spec, FS).unwrap())
            };
            // Each forward-backward pass covers the padded signal; the split is per pass length
            let filter = DeviceFilter::<f32>::new(&client, &spec, FS).unwrap();
            let steps = n + 2 * filter.settling_samples().min(n - 1);
            let explicit = run(filter.with_block_len(steps.div_ceil(PINNED_BLOCK_COUNT)).with_layout(PassLayout::ChannelMajor));
            assert_eq!(pinned, explicit, "{}", client.name());
        }
    }
    for target in ComputeTarget::available() {
        target.run(Pinned).expect("compiled-in runtime");
    }
}
