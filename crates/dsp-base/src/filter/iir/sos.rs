//! Host dispatch of the SOS cascade kernel: forward, forward-backward (zero phase) and stateful
//! streaming passes over `[channels, samples]` device buffers.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use cubecl::prelude::*;
use cubecl::server::Handle;
use cubecl::tune::{LocalTuner, Tunable, TunableSet, local_tuner};
use dsp_core::compute::LaunchGeometry;
use dsp_core::compute::tune::{size_class, tune_id};

use super::kernels::sos::{SVF_COEFFS, sos_block_kernel, sos_block_scan_kernel};
use crate::filter::design::{FilterError, FilterMode, FilterSpec, Sos};

/// Numbers of time blocks the autotuner tries per pass. One block is a single sequential walk per
/// channel; more blocks give more parallel units at the cost of a second pass over the data. The
/// fastest depends on the device and problem size, so it is measured (see
/// [`dsp_core::compute::tune`]).
const BLOCK_COUNT_CANDIDATES: [usize; 5] = [1, 4, 16, 64, 256];

/// A designed filter uploaded to the device, ready to run on any chunk.
#[derive(Debug, Clone)]
pub struct DeviceFilter {
    sos: Sos,
    mode: FilterMode,
    settling: usize,
    coeffs: Handle,
    /// Uploaded `Aᴸ` per block length `L`.
    transitions: Arc<Mutex<HashMap<usize, Handle>>>,
    /// Fixed block length instead of the autotuned one (`usize::MAX` = one block).
    block_len: Option<usize>,
}

impl DeviceFilter {
    /// Designs `spec` for `sample_rate` Hz and uploads its coefficients.
    pub fn new<R: Runtime>(
        client: &ComputeClient<R>,
        spec: &FilterSpec,
        sample_rate: f64,
    ) -> Result<Self, FilterError> {
        let sos = spec.design(sample_rate)?;
        Ok(Self::from_sos(client, sos, spec.mode))
    }

    /// Uploads already designed sections.
    pub fn from_sos<R: Runtime>(client: &ComputeClient<R>, sos: Sos, mode: FilterMode) -> Self {
        let settling = sos.settling_samples(crate::filter::design::DEFAULT_SETTLING_TOLERANCE);
        let coeffs = client.create_from_slice(f32::as_bytes(&sos.svf_coeffs_f32()));
        Self { sos, mode, settling, coeffs, transitions: Arc::default(), block_len: None }
    }

    /// Runs every pass with time blocks of `block_len` steps (`usize::MAX` for one block) instead
    /// of the autotuned length. Results are the same up to f32 rounding; for tests and benchmarks.
    pub fn with_block_len(mut self, block_len: usize) -> Self {
        self.block_len = Some(block_len.max(1));
        self
    }

    pub fn sos(&self) -> &Sos {
        &self.sos
    }

    pub fn mode(&self) -> FilterMode {
        self.mode
    }

    pub fn n_sections(&self) -> usize {
        self.sos.len()
    }

    /// Settling of one pass in samples; forward-backward needs it on both sides.
    pub fn settling_samples(&self) -> usize {
        self.settling
    }

    /// Floats of state per channel (`n_sections · 2` integrator states plus the offset).
    pub fn state_len(&self) -> usize {
        self.n_sections() * 2 + 1
    }

    /// Floats of scratch needed by [`Self::apply`] for `channels × samples`
    /// (forward-backward keeps the odd-padded forward pass).
    pub fn scratch_len(&self, channels: usize, samples: usize) -> usize {
        match self.mode {
            FilterMode::Forward => 0,
            FilterMode::ForwardBackward => channels * (samples + 2 * self.edge_pad(samples)),
        }
    }

    /// Odd-reflection padding used at the chunk edges: the settling length, limited by the chunk.
    fn edge_pad(&self, samples: usize) -> usize {
        self.settling.min(samples.saturating_sub(1))
    }

    /// Filters an independent chunk (no carried state): each pass starts from the steady state of
    /// its first sample; forward-backward odd-pads both edges (`sosfiltfilt`). `state` must hold
    /// `channels · state_len()` floats, `scratch` at least `scratch_len()` floats.
    #[allow(clippy::too_many_arguments)]
    pub fn apply<R: Runtime>(
        &self,
        client: &ComputeClient<R>,
        input: &Handle,
        output: &Handle,
        scratch: &Handle,
        state: &Handle,
        channels: usize,
        samples: usize,
    ) {
        if channels == 0 || samples == 0 {
            return;
        }
        match self.mode {
            FilterMode::Forward => {
                self.pass(client, input, output, state, channels, samples, 0, 0, samples, false, false);
            }
            FilterMode::ForwardBackward => {
                let pad = self.edge_pad(samples);
                let ext = samples + 2 * pad;
                self.pass(client, input, scratch, state, channels, samples, pad, 0, ext, false, false);
                self.pass(client, scratch, output, state, channels, ext, 0, pad, samples, true, false);
            }
        }
    }

    /// Filters the next chunk of a continuous stream, continuing from `state` (forward only).
    /// `first` starts from the steady state of the first sample instead.
    #[allow(clippy::too_many_arguments)]
    pub fn apply_stateful<R: Runtime>(
        &self,
        client: &ComputeClient<R>,
        input: &Handle,
        output: &Handle,
        state: &Handle,
        channels: usize,
        samples: usize,
        first: bool,
    ) -> Result<(), FilterError> {
        if self.mode != FilterMode::Forward {
            return Err(FilterError::ForwardBackwardOnLiveStream);
        }
        if channels > 0 && samples > 0 {
            self.pass(client, input, output, state, channels, samples, 0, 0, samples, false, !first);
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn pass<R: Runtime>(
        &self,
        client: &ComputeClient<R>,
        input: &Handle,
        output: &Handle,
        state: &Handle,
        channels: usize,
        in_len: usize,
        pad: usize,
        out_start: usize,
        out_len: usize,
        reverse: bool,
        carry: bool,
    ) {
        let p = PassInputs {
            client: client.clone(),
            filter: self.clone(),
            input: input.clone(),
            output: output.clone(),
            state: state.clone(),
            channels,
            in_len,
            pad,
            out_start,
            out_len,
            reverse,
            carry,
        };
        match self.block_len {
            Some(len) => p.run(len),
            None => tuned_pass(p),
        }
    }

    /// `Aᴸ − I` of the cascade (zero-input state transition over `block_len` steps) on the device.
    fn transition<R: Runtime>(&self, client: &ComputeClient<R>, block_len: usize) -> Handle {
        let mut cache = self.transitions.lock().expect("transition cache");
        cache
            .entry(block_len)
            .or_insert_with(|| {
                let power = transition_power(&self.sos.svf_coeffs_f32(), self.n_sections(), block_len);
                client.create_from_slice(f32::as_bytes(&power))
            })
            .clone()
    }
}

/// Row-major `Aᴸ − I` (`[2n][2n]`) of the state update the kernel performs with zero input, from
/// the same f32 coefficients, composed in f64.
fn transition_power(coeffs: &[f32], n_sections: usize, power: usize) -> Vec<f32> {
    let dim = 2 * n_sections;
    // Column k of A: one zero-input step from basis state e_k
    let mut a = vec![0.0f64; dim * dim];
    for k in 0..dim {
        let mut z = vec![0.0f64; dim];
        z[k] = 1.0;
        let mut v = 0.0f64;
        for s in 0..n_sections {
            let c: Vec<f64> = coeffs[s * SVF_COEFFS..(s + 1) * SVF_COEFFS].iter().map(|&x| x as f64).collect();
            let (ic1, ic2) = (z[2 * s], z[2 * s + 1]);
            let v3 = v - ic2;
            let v1 = c[0] * ic1 + c[1] * v3;
            let v2 = ic2 + c[1] * ic1 + c[2] * v3;
            z[2 * s] = 2.0 * v1 - ic1;
            z[2 * s + 1] = 2.0 * v2 - ic2;
            v = c[3] * v + c[4] * v1 + c[5] * v2;
        }
        for r in 0..dim {
            a[r * dim + k] = z[r];
        }
    }
    let mul = |x: &[f64], y: &[f64]| {
        let mut out = vec![0.0f64; dim * dim];
        for r in 0..dim {
            for c in 0..dim {
                out[r * dim + c] = (0..dim).map(|k| x[r * dim + k] * y[k * dim + c]).sum();
            }
        }
        out
    };
    let mut result: Vec<f64> = (0..dim * dim).map(|i| if i % (dim + 1) == 0 { 1.0 } else { 0.0 }).collect();
    let (mut base, mut e) = (a, power);
    while e > 0 {
        if e & 1 == 1 {
            result = mul(&result, &base);
        }
        base = mul(&base, &base);
        e >>= 1;
    }
    for d in 0..dim {
        result[d * dim + d] -= 1.0;
    }
    result.into_iter().map(|x| x as f32).collect()
}

/// One pass of a [`DeviceFilter`] (cloned per autotune candidate).
#[derive(Clone)]
struct PassInputs<R: Runtime> {
    client: ComputeClient<R>,
    filter: DeviceFilter,
    input: Handle,
    output: Handle,
    state: Handle,
    channels: usize,
    in_len: usize,
    pad: usize,
    out_start: usize,
    out_len: usize,
    reverse: bool,
    carry: bool,
}

impl<R: Runtime> PassInputs<R> {
    fn steps(&self) -> usize {
        self.in_len + 2 * self.pad
    }

    /// Runs the pass with time blocks of `block_len` steps.
    fn run(&self, block_len: usize) {
        let (client, f, channels) = (&self.client, &self.filter, self.channels);
        let n = f.n_sections();
        let steps = self.steps();
        let block_len = block_len.clamp(1, steps.max(1));
        let blocks = steps.div_ceil(block_len).max(1);
        let slots = client.empty((channels * blocks * f.state_len() * 4).max(4));
        let launch = |phase: u32, active: usize| unsafe {
            // `phase` is a compile-time kernel parameter: each phase is its own specialised kernel
            let geom = LaunchGeometry::elementwise(client, channels * active);
            sos_block_kernel::launch::<R>(
                client,
                geom.cube_count,
                geom.cube_dim,
                ArrayArg::from_raw_parts(self.input.clone(), channels * self.in_len),
                ArrayArg::from_raw_parts(self.output.clone(), channels * self.out_len),
                ArrayArg::from_raw_parts(f.coeffs.clone(), n * SVF_COEFFS + 1),
                ArrayArg::from_raw_parts(self.state.clone(), channels * f.state_len()),
                ArrayArg::from_raw_parts(slots.clone(), (channels * blocks * f.state_len()).max(1)),
                channels as u32,
                self.in_len as u32,
                self.pad as u32,
                self.out_start as u32,
                self.out_len as u32,
                self.reverse as u32,
                self.carry as u32,
                block_len as u32,
                blocks as u32,
                phase,
                n,
            );
        };
        if blocks > 1 {
            launch(0, blocks - 1);
            let transition = f.transition(client, block_len);
            let geom = LaunchGeometry::per_channel(client, channels);
            unsafe {
                sos_block_scan_kernel::launch::<R>(
                    client,
                    geom.cube_count,
                    geom.cube_dim,
                    ArrayArg::from_raw_parts(slots.clone(), channels * blocks * f.state_len()),
                    ArrayArg::from_raw_parts(transition, 4 * n * n),
                    channels as u32,
                    blocks as u32,
                    n,
                );
            }
        }
        launch(1, blocks);
    }
}

/// [`PassInputs::run`] with the block length CubeCL's autotuner found fastest for this device,
/// filter order and problem size. Benchmarks write to scratch output and state buffers.
fn tuned_pass<R: Runtime>(inputs: PassInputs<R>) {
    static TUNER: LocalTuner<String, String> = local_tuner!("sos-blocks");
    let set = TUNER.init(|| {
        let key = |p: &PassInputs<R>| format!("n{}-c{}-t{}", p.filter.n_sections(), size_class(p.channels), size_class(p.steps()));
        let scratch = |_: &String, p: &PassInputs<R>| PassInputs {
            output: p.client.empty((p.channels * p.out_len * 4).max(4)),
            state: p.client.create_from_slice(f32::as_bytes(&vec![0.0f32; p.channels * p.filter.state_len()])),
            ..p.clone()
        };
        let set: TunableSet<String, PassInputs<R>, ()> = TunableSet::new(key, scratch);
        BLOCK_COUNT_CANDIDATES.iter().fold(set, |set, &blocks| {
            set.with(Tunable::new(&format!("blocks{blocks}"), move |p: PassInputs<R>| {
                Ok::<_, String>(p.run(p.steps().div_ceil(blocks)))
            }))
        })
    });
    let client = inputs.client.clone();
    TUNER.execute(&tune_id(&client), &client, set, inputs)
}

/// One-shot filtering of a `[channels, samples]` buffer (independent chunk semantics).
pub fn execute_filter<R: Runtime>(
    client: &ComputeClient<R>,
    spec: &FilterSpec,
    sample_rate: f64,
    input: &Handle,
    output: &Handle,
    channels: usize,
    samples: usize,
) -> Result<(), FilterError> {
    let filter = DeviceFilter::new(client, spec, sample_rate)?;
    let scratch = client.empty((filter.scratch_len(channels, samples) * 4).max(4));
    let state = client.empty((channels * filter.state_len() * 4).max(4));
    filter.apply(client, input, output, &scratch, &state, channels, samples);
    Ok(())
}
