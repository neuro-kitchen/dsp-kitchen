//! Host dispatch of the SOS cascade kernel: forward, forward-backward (zero phase) and stateful
//! streaming passes over `[channels, samples]` device buffers.

use cubecl::prelude::*;
use cubecl::server::Handle;

use super::kernels::sos::{SVF_COEFFS, sos_cascade_kernel};
use crate::filter::design::{FilterError, FilterMode, FilterSpec, Sos};

const CHANNELS_PER_CUBE: u32 = 64;

/// A designed filter uploaded to the device, ready to run on any chunk.
#[derive(Debug, Clone)]
pub struct DeviceFilter {
    sos: Sos,
    mode: FilterMode,
    settling: usize,
    coeffs: Handle,
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
        Self { sos, mode, settling, coeffs }
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
        let n = self.n_sections();
        let cubes = (channels as u32).div_ceil(CHANNELS_PER_CUBE);
        unsafe {
            sos_cascade_kernel::launch::<R>(
                client,
                CubeCount::Static(cubes, 1, 1),
                CubeDim::new_1d(CHANNELS_PER_CUBE),
                ArrayArg::from_raw_parts(input.clone(), channels * in_len),
                ArrayArg::from_raw_parts(output.clone(), channels * out_len),
                ArrayArg::from_raw_parts(self.coeffs.clone(), n * SVF_COEFFS + 1),
                ArrayArg::from_raw_parts(state.clone(), channels * self.state_len()),
                channels as u32,
                in_len as u32,
                pad as u32,
                out_start as u32,
                out_len as u32,
                reverse as u32,
                carry as u32,
                n,
            );
        }
    }
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
