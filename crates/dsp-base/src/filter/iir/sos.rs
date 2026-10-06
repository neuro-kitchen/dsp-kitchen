//! Host dispatch of the SOS cascade kernel: forward, forward-backward (zero phase) and stateful
//! streaming passes over `[channels, samples]` device buffers.

use std::collections::HashMap;
use std::marker::PhantomData;
use std::sync::{Arc, Mutex};

use cubecl::prelude::*;
use cubecl::server::Handle;
use cubecl::tune::{LocalTuner, Tunable, TunableSet, local_tuner};
use dsp_core::compute::LaunchGeometry;
use dsp_core::compute::tune::{size_class, tune_id};

use super::kernels::sos::{SVF_COEFFS, coeffs_len, sos_block_kernel, sos_block_scan_kernel};
use crate::core::{buffer, cast, cast_all, layout, to_f64, DspFloat, Scratch};
use crate::filter::design::{FilterError, FilterMode, FilterSpec, FilterStart, Sos};

/// Numbers of time blocks the autotuner tries per pass. One block is a single sequential walk per
/// channel; more blocks give more parallel units at the cost of a second pass over the data. The
/// fastest depends on the device and problem size, so it is measured (see
/// [`dsp_core::compute::tune`]).
const BLOCK_COUNT_CANDIDATES: [usize; 5] = [1, 4, 16, 64, 256];

/// Memory order a pass runs in. Channel-major rows are the buffers' own order; time-major runs on a
/// transposed copy so the units of a plane (consecutive channels) read and write consecutive
/// addresses, at the cost of two transposes. Which is faster depends on the device; the autotuner
/// tries both.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PassLayout {
    ChannelMajor,
    TimeMajor,
}

/// Device buffers a filter reuses across passes: block start states and time-major staging. Kept per
/// thread (CubeCL gives each thread its own stream), so clones running on several threads never share
/// a buffer.
#[derive(Debug, Default)]
struct PassWorkspace {
    slots: Scratch,
    input_t: Scratch,
    output_t: Scratch,
}

/// A designed filter uploaded to the device in element type `F`, ready to run on any chunk.
///
/// Passes follow scipy: [`FilterMode::Forward`] starts per [`FilterStart`] (default at rest, as
/// `sosfilt`), [`FilterMode::ForwardBackward`] odd-pads both edges and starts each pass from the
/// steady state of its first sample (`sosfiltfilt`).
#[derive(Debug, Clone)]
pub struct DeviceFilter<F: DspFloat = f32> {
    sos: Sos,
    mode: FilterMode,
    start: FilterStart,
    settling: usize,
    /// Kernel coefficients (see [`coeffs_len`]) after rounding to `F`, kept in f64 for the host-side
    /// transition matrices.
    coeffs_host: Vec<f64>,
    coeffs: Handle,
    /// Uploaded `Aᴸ − I` per block length `L`.
    transitions: Arc<Mutex<HashMap<usize, Handle>>>,
    /// Fixed block length instead of the autotuned one (`usize::MAX` = one block).
    block_len: Option<usize>,
    /// Fixed memory order instead of the autotuned one.
    layout: Option<PassLayout>,
    workspace: Arc<Mutex<HashMap<std::thread::ThreadId, PassWorkspace>>>,
    _float: PhantomData<F>,
}

impl<F: DspFloat> DeviceFilter<F> {
    /// Designs `spec` for `sample_rate` Hz and uploads its coefficients.
    pub fn new<R: Runtime>(client: &ComputeClient<R>, spec: &FilterSpec, sample_rate: f64) -> Result<Self, FilterError> {
        let sos = spec.design(sample_rate)?;
        Ok(Self::from_sos(client, sos, spec.mode).with_start(spec.start))
    }

    /// Uploads already designed sections (forward passes start at rest; see [`Self::with_start`]).
    pub fn from_sos<R: Runtime>(client: &ComputeClient<R>, sos: Sos, mode: FilterMode) -> Self {
        let settling = sos.settling_samples(crate::filter::design::DEFAULT_SETTLING_TOLERANCE);
        let coeffs_host = kernel_coeffs::<F>(&sos);
        let coeffs = buffer::upload(client, &cast_all::<F>(&coeffs_host));
        Self {
            sos,
            mode,
            start: FilterStart::default(),
            settling,
            coeffs_host,
            coeffs,
            transitions: Arc::default(),
            block_len: None,
            layout: None,
            workspace: Arc::default(),
            _float: PhantomData,
        }
    }

    /// Where forward passes start.
    pub fn with_start(mut self, start: FilterStart) -> Self {
        self.start = start;
        self
    }

    /// Runs every pass with time blocks of `block_len` steps (`usize::MAX` for one block) instead
    /// of the autotuned length. Results are the same up to rounding; for tests and benchmarks.
    pub fn with_block_len(mut self, block_len: usize) -> Self {
        self.block_len = Some(block_len.max(1));
        self
    }

    /// Runs every pass in `layout` instead of the autotuned one (for tests and benchmarks).
    pub fn with_layout(mut self, layout: PassLayout) -> Self {
        self.layout = Some(layout);
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

    /// Values of state per channel (`n_sections · 2` integrator states plus the offset).
    pub fn state_len(&self) -> usize {
        self.n_sections() * 2 + 1
    }

    /// Values of scratch needed by [`Self::apply`] for `channels × samples`
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

    /// Filters an independent chunk (no carried state). Forward passes start per [`FilterStart`];
    /// forward-backward odd-pads both edges and starts each pass from the steady state of its first
    /// sample. `state` must hold `channels · state_len()` values, `scratch` at least `scratch_len()`.
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
        let pass = |input: &Handle, output: &Handle, in_len, pad, out_start, out_len, reverse, rest| Pass {
            input: input.clone(),
            output: output.clone(),
            state: state.clone(),
            channels,
            in_len,
            pad,
            out_start,
            out_len,
            reverse,
            carry: false,
            rest,
        };
        let rest = self.start == FilterStart::Rest;
        match self.mode {
            FilterMode::Forward => self.run(client, pass(input, output, samples, 0, 0, samples, false, rest)),
            FilterMode::ForwardBackward => {
                let pad = self.edge_pad(samples);
                let ext = samples + 2 * pad;
                self.run(client, pass(input, scratch, samples, pad, 0, ext, false, false));
                self.run(client, pass(scratch, output, ext, 0, pad, samples, true, false));
            }
        }
    }

    /// Filters the next chunk of a continuous stream, continuing from `state` (forward only).
    /// `first` starts the stream per [`FilterStart`] instead.
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
            let pass = Pass {
                input: input.clone(),
                output: output.clone(),
                state: state.clone(),
                channels,
                in_len: samples,
                pad: 0,
                out_start: 0,
                out_len: samples,
                reverse: false,
                carry: !first,
                rest: first && self.start == FilterStart::Rest,
            };
            self.run(client, pass);
        }
        Ok(())
    }

    fn run<R: Runtime>(&self, client: &ComputeClient<R>, pass: Pass) {
        let inputs = PassInputs { client: client.clone(), filter: self.clone(), pass };
        match (self.block_len, self.layout) {
            (None, None) => tuned_pass(inputs),
            (len, layout) => inputs.run(len.unwrap_or(usize::MAX), layout.unwrap_or(PassLayout::ChannelMajor)),
        }
    }

    /// `Aᴸ − I` of the cascade (zero-input state transition over `block_len` steps) on the device.
    fn transition<R: Runtime>(&self, client: &ComputeClient<R>, block_len: usize) -> Handle {
        let mut cache = self.transitions.lock().expect("transition cache");
        cache
            .entry(block_len)
            .or_insert_with(|| {
                let power = transition_power(&self.coeffs_host, self.n_sections(), block_len);
                buffer::upload(client, &cast_all::<F>(&power))
            })
            .clone()
    }
}

/// Kernel coefficients of `sos` (layout of [`coeffs_len`]) rounded to `F`: the SVF coefficients and
/// DC gain, then the at-rest states computed from those rounded coefficients, so the device starts
/// exactly where its own arithmetic settles.
fn kernel_coeffs<F: DspFloat>(sos: &Sos) -> Vec<f64> {
    let round = |v: f64| to_f64(cast::<F>(v));
    let n = sos.len();
    let mut c: Vec<f64> = sos.sections.iter().flat_map(|s| s.svf()).map(round).collect();
    c.push(round(sos.dc_gain()));
    let rest = unit_step_states(&c, n);
    c.extend(rest.into_iter().map(round));
    debug_assert_eq!(c.len(), coeffs_len(n));
    c
}

/// One step of the SVF cascade the kernel runs (`coeffs` as laid out by [`kernel_coeffs`]): updates
/// the `2n` states `z` with input `x` and returns the output.
fn svf_step(coeffs: &[f64], n_sections: usize, z: &mut [f64], x: f64) -> f64 {
    let mut v = x;
    for s in 0..n_sections {
        let c = &coeffs[s * SVF_COEFFS..(s + 1) * SVF_COEFFS];
        let (ic1, ic2) = (z[2 * s], z[2 * s + 1]);
        let v3 = v - ic2;
        let v1 = c[0] * ic1 + c[1] * v3;
        let v2 = ic2 + c[1] * ic1 + c[2] * v3;
        z[2 * s] = 2.0 * v1 - ic1;
        z[2 * s + 1] = 2.0 * v2 - ic2;
        v = c[3] * v + c[4] * v1 + c[5] * v2;
    }
    v
}

/// Row-major zero-input state transition `A` (`[2n][2n]`) of one [`svf_step`].
fn transition_matrix(coeffs: &[f64], n_sections: usize) -> Vec<f64> {
    let dim = 2 * n_sections;
    let mut a = vec![0.0f64; dim * dim];
    for k in 0..dim {
        let mut z = vec![0.0f64; dim];
        z[k] = 1.0;
        svf_step(coeffs, n_sections, &mut z, 0.0);
        for r in 0..dim {
            a[r * dim + k] = z[r];
        }
    }
    a
}

/// States the cascade settles to under a constant unit input: the solution of `z = A·z + b`, where
/// `b` is the state one step from zero with input 1. `I − A` is invertible for a stable cascade.
fn unit_step_states(coeffs: &[f64], n_sections: usize) -> Vec<f64> {
    let dim = 2 * n_sections;
    let a = transition_matrix(coeffs, n_sections);
    let mut b = vec![0.0f64; dim];
    svf_step(coeffs, n_sections, &mut b, 1.0);
    let i_minus_a: Vec<f64> = (0..dim * dim).map(|i| if i % (dim + 1) == 0 { 1.0 } else { 0.0 } - a[i]).collect();
    solve(i_minus_a, b, dim)
}

/// Solves `m · x = rhs` (`m` row-major `[dim][dim]`) by Gaussian elimination with partial pivoting.
fn solve(mut m: Vec<f64>, mut rhs: Vec<f64>, dim: usize) -> Vec<f64> {
    for col in 0..dim {
        let pivot = (col..dim).max_by(|&a, &b| m[a * dim + col].abs().total_cmp(&m[b * dim + col].abs())).unwrap_or(col);
        if pivot != col {
            for k in 0..dim {
                m.swap(col * dim + k, pivot * dim + k);
            }
            rhs.swap(col, pivot);
        }
        let p = m[col * dim + col];
        for row in col + 1..dim {
            let f = m[row * dim + col] / p;
            for k in col..dim {
                m[row * dim + k] -= f * m[col * dim + k];
            }
            rhs[row] -= f * rhs[col];
        }
    }
    let mut x = vec![0.0f64; dim];
    for row in (0..dim).rev() {
        let s: f64 = (row + 1..dim).map(|k| m[row * dim + k] * x[k]).sum();
        x[row] = (rhs[row] - s) / m[row * dim + row];
    }
    x
}

/// Row-major `Aᴸ − I` (`[2n][2n]`) of the zero-input state update, composed in f64.
fn transition_power(coeffs: &[f64], n_sections: usize, power: usize) -> Vec<f64> {
    let dim = 2 * n_sections;
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
    let (mut base, mut e) = (transition_matrix(coeffs, n_sections), power);
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
    result
}

/// Geometry and flags of one pass.
#[derive(Clone)]
struct Pass {
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
    rest: bool,
}

/// One pass of a [`DeviceFilter`] (cloned per autotune candidate).
#[derive(Clone)]
struct PassInputs<R: Runtime, F: DspFloat> {
    client: ComputeClient<R>,
    filter: DeviceFilter<F>,
    pass: Pass,
}

impl<R: Runtime, F: DspFloat> PassInputs<R, F> {
    fn steps(&self) -> usize {
        self.pass.in_len + 2 * self.pass.pad
    }

    /// Runs the pass with time blocks of `block_len` steps in `layout`.
    fn run(&self, block_len: usize, layout: PassLayout) {
        let (client, f, p) = (&self.client, &self.filter, &self.pass);
        let n = f.n_sections();
        let steps = self.steps();
        let block_len = block_len.clamp(1, steps.max(1));
        let blocks = steps.div_ceil(block_len).max(1);
        let mut workspaces = f.workspace.lock().expect("filter workspace");
        let ws = workspaces.entry(std::thread::current().id()).or_default();
        let slots = ws.slots.get::<R, F>(client, p.channels * blocks * f.state_len());
        // Buffers the kernel reads and writes, and their (channel, time) strides
        let (input, output, in_strides, out_strides) = match layout {
            PassLayout::ChannelMajor => (p.input.clone(), p.output.clone(), (p.in_len, 1), (p.out_len, 1)),
            PassLayout::TimeMajor => {
                let input_t = ws.input_t.get::<R, F>(client, p.channels * p.in_len);
                layout::transpose::<R, F>(client, &p.input, &input_t, p.channels, p.in_len);
                let output_t = ws.output_t.get::<R, F>(client, p.channels * p.out_len);
                (input_t, output_t, (1, p.channels), (1, p.channels))
            }
        };
        let launch = |phase: u32, active: usize| unsafe {
            // `phase` is a compile-time kernel parameter: each phase is its own specialised kernel
            let geom = LaunchGeometry::elementwise(client, p.channels * active);
            sos_block_kernel::launch::<F, R>(
                client,
                geom.cube_count,
                geom.cube_dim,
                ArrayArg::from_raw_parts(input.clone(), p.channels * p.in_len),
                ArrayArg::from_raw_parts(output.clone(), p.channels * p.out_len),
                ArrayArg::from_raw_parts(f.coeffs.clone(), coeffs_len(n)),
                ArrayArg::from_raw_parts(p.state.clone(), p.channels * f.state_len()),
                ArrayArg::from_raw_parts(slots.clone(), (p.channels * blocks * f.state_len()).max(1)),
                p.channels as u32,
                p.in_len as u32,
                p.pad as u32,
                p.out_start as u32,
                p.out_len as u32,
                in_strides.0 as u32,
                in_strides.1 as u32,
                out_strides.0 as u32,
                out_strides.1 as u32,
                p.reverse as u32,
                p.carry as u32,
                p.rest as u32,
                block_len as u32,
                blocks as u32,
                phase,
                n,
            );
        };
        if blocks > 1 {
            launch(0, blocks - 1);
            let transition = f.transition(client, block_len);
            let geom = LaunchGeometry::per_channel(client, p.channels);
            unsafe {
                sos_block_scan_kernel::launch::<F, R>(
                    client,
                    geom.cube_count,
                    geom.cube_dim,
                    ArrayArg::from_raw_parts(slots.clone(), p.channels * blocks * f.state_len()),
                    ArrayArg::from_raw_parts(transition, (2 * n) * (2 * n)),
                    p.channels as u32,
                    blocks as u32,
                    n,
                );
            }
        }
        launch(1, blocks);
        if layout == PassLayout::TimeMajor {
            layout::transpose::<R, F>(client, &output, &p.output, p.out_len, p.channels);
        }
    }
}

/// [`PassInputs::run`] with the block length and memory order CubeCL's autotuner found fastest for
/// this device, element type, filter order and problem size. Benchmarks write to scratch output and
/// state buffers.
fn tuned_pass<R: Runtime, F: DspFloat>(inputs: PassInputs<R, F>) {
    // cubecl-runtime 0.10's `local_tuner!` expands with a trailing semicolon (rust-lang/rust#79813)
    #[allow(semicolon_in_expressions_from_non_local_macros)]
    static TUNER: LocalTuner<String, String> = local_tuner!("sos-blocks");
    let set = TUNER.init(|| {
        let key = |p: &PassInputs<R, F>| {
            format!("{}-n{}-c{}-t{}", F::type_name(), p.filter.n_sections(), size_class(p.pass.channels), size_class(p.steps()))
        };
        let scratch = |_: &String, p: &PassInputs<R, F>| {
            let mut pass = p.pass.clone();
            pass.output = buffer::empty::<R, F>(&p.client, pass.channels * pass.out_len);
            pass.state = buffer::zeros::<R, F>(&p.client, pass.channels * p.filter.state_len());
            PassInputs { client: p.client.clone(), filter: p.filter.clone(), pass }
        };
        let set: TunableSet<String, PassInputs<R, F>, ()> = TunableSet::new(key, scratch);
        let candidates = [PassLayout::ChannelMajor, PassLayout::TimeMajor]
            .into_iter()
            .flat_map(|layout| BLOCK_COUNT_CANDIDATES.into_iter().map(move |blocks| (layout, blocks)));
        candidates.fold(set, |set, (layout, blocks)| {
            set.with(Tunable::new(&format!("{layout:?}-blocks{blocks}"), move |p: PassInputs<R, F>| {
                p.run(p.steps().div_ceil(blocks), layout);
                Ok::<_, String>(())
            }))
        })
    });
    let client = inputs.client.clone();
    TUNER.execute(&tune_id(&client), &client, set, inputs)
}

/// One-shot filtering of a `[channels, samples]` buffer (independent chunk semantics).
#[allow(clippy::too_many_arguments)]
pub fn execute_filter<R: Runtime, F: DspFloat>(
    client: &ComputeClient<R>,
    spec: &FilterSpec,
    sample_rate: f64,
    input: &Handle,
    output: &Handle,
    channels: usize,
    samples: usize,
) -> Result<(), FilterError> {
    let filter = DeviceFilter::<F>::new(client, spec, sample_rate)?;
    let scratch = buffer::empty::<R, F>(client, filter.scratch_len(channels, samples));
    let state = buffer::empty::<R, F>(client, channels * filter.state_len());
    filter.apply(client, input, output, &scratch, &state, channels, samples);
    Ok(())
}
