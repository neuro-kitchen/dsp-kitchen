//! Hardware-agnostic CubeCL (`#[cube]`) kernels for temporal basis projection,
//! waveform reconstruction, and multi-template matched filtering.
//!
//! All launch geometries are derived dynamically at runtime from the `ComputeClient<R>`
//! via [`dsp_core::compute::LaunchGeometry`] — zero hardcoded workgroup constants or
//! `is_cpu` branches.

use cubecl::prelude::*;
use dsp_core::compute::{
    ComputeTask, LaunchGeometry, channel_position, sample_position,
};
use dsp_core::{ComputeTarget, DspError, DspResult};

/// Projects `[num_spikes, num_channels, window_len]` snippets onto a `[num_components, window_len]`
/// temporal basis to produce `[num_spikes, num_channels, num_components]`.
///
/// One unit per output coefficient `(spike, channel, comp)` indexed via `ABSOLUTE_POS`.
#[cube(launch)]
pub fn temporal_basis_project_kernel(
    snippets: &Array<f32>,
    basis: &Array<f32>,
    out_coeffs: &mut Array<f32>,
    total_outputs: u32,
    num_components: u32,
    window_len: u32,
) {
    let idx = ABSOLUTE_POS as u32;
    if idx < total_outputs {
        let spike_ch = idx / num_components;
        let comp = idx - spike_ch * num_components;
        let snip_base = spike_ch * window_len;
        let basis_base = comp * window_len;

        let mut dot = 0.0f32;
        let mut t = 0u32;
        while t < window_len {
            dot += snippets[(snip_base + t) as usize] * basis[(basis_base + t) as usize];
            t += 1u32;
        }
        out_coeffs[idx as usize] = dot;
    }
}

/// Reconstructs `[num_spikes, num_channels, window_len]` snippets from
/// `[num_spikes, num_channels, num_components]` basis coefficients and a
/// `[num_components, window_len]` orthonormal temporal basis.
#[cube(launch)]
pub fn temporal_basis_reconstruct_kernel(
    coeffs: &Array<f32>,
    basis: &Array<f32>,
    out_snippets: &mut Array<f32>,
    total_elements: u32,
    num_components: u32,
    window_len: u32,
) {
    let idx = ABSOLUTE_POS as u32;
    if idx < total_elements {
        let spike_ch = idx / window_len;
        let t = idx - spike_ch * window_len;
        let coeff_base = spike_ch * num_components;

        let mut sum = 0.0f32;
        let mut c = 0u32;
        while c < num_components {
            sum += coeffs[(coeff_base + c) as usize] * basis[(c * window_len + t) as usize];
            c += 1u32;
        }
        out_snippets[idx as usize] = sum;
    }
}

/// Convolves a `[channels, samples]` signal chunk with a bank of `[num_templates, window_len]`
/// L2-normalized universal temporal templates centered at `center_offset`.
///
/// Uses [`LaunchGeometry::channels_samples`] so neighbouring units along `x` walk contiguous
/// time samples of the same channel. Interior samples `[center_offset, samples - post_Span)`
/// run without boundary clamping.
#[cube(launch)]
pub fn universal_template_filter_kernel(
    trace: &Array<f32>,
    templates: &Array<f32>,
    out_energy: &mut Array<f32>,
    channels: u32,
    samples: u32,
    num_templates: u32,
    window_len: u32,
    center_offset: u32,
) {
    let ch = channel_position();
    let s = sample_position();

    if ch < channels && s < samples {
        let post_span = window_len - center_offset;
        let is_interior = s >= center_offset && s + post_span <= samples;

        let mut energy_sum = 0.0f32;
        if is_interior {
            let trace_start = ch * samples + (s - center_offset);
            let mut m = 0u32;
            while m < num_templates {
                let tpl_base = m * window_len;
                let mut dot = 0.0f32;
                let mut t = 0u32;
                while t < window_len {
                    dot += trace[(trace_start + t) as usize] * templates[(tpl_base + t) as usize];
                    t += 1u32;
                }
                // Templates are oriented with negative troughs, so a negative-going
                // extracellular spike yields a positive inner product (dot > 0).
                let proj = f32::max(0.0f32, dot);
                energy_sum += proj * proj;
                m += 1u32;
            }
        }
        out_energy[(ch * samples + s) as usize] = f32::sqrt(energy_sum);
    }
}

/// Executes temporal basis projection (`snippets @ basis^T`) on a CubeCL `ComputeClient<R>`.
pub fn execute_temporal_basis_project<R: Runtime>(
    client: &ComputeClient<R>,
    snippets: &[f32],
    num_spikes: usize,
    num_channels: usize,
    window_len: usize,
    basis: &[f32],
    num_components: usize,
) -> Vec<f32> {
    let total_outputs = num_spikes * num_channels * num_components;
    if total_outputs == 0 || window_len == 0 {
        return Vec::new();
    }

    let snip_handle = client.create_from_slice(f32::as_bytes(snippets));
    let basis_handle = client.create_from_slice(f32::as_bytes(basis));
    let out_handle = client.empty(total_outputs * std::mem::size_of::<f32>());

    let geom = LaunchGeometry::elementwise(client, total_outputs);
    unsafe {
        temporal_basis_project_kernel::launch::<R>(
            client,
            geom.cube_count,
            geom.cube_dim,
            ArrayArg::from_raw_parts(snip_handle, snippets.len()),
            ArrayArg::from_raw_parts(basis_handle, basis.len()),
            ArrayArg::from_raw_parts(out_handle.clone(), total_outputs),
            total_outputs as u32,
            num_components as u32,
            window_len as u32,
        );
    }

    let bytes = client.read_one(out_handle).expect("VRAM read basis coeffs");
    f32::from_bytes(&bytes)[..total_outputs].to_vec()
}

/// Executes waveform reconstruction (`coeffs @ basis`) on a CubeCL `ComputeClient<R>`.
pub fn execute_temporal_basis_reconstruct<R: Runtime>(
    client: &ComputeClient<R>,
    coeffs: &[f32],
    num_spikes: usize,
    num_channels: usize,
    num_components: usize,
    basis: &[f32],
    window_len: usize,
) -> Vec<f32> {
    let total_elements = num_spikes * num_channels * window_len;
    if total_elements == 0 || num_components == 0 {
        return vec![0.0; total_elements];
    }

    let coeff_handle = client.create_from_slice(f32::as_bytes(coeffs));
    let basis_handle = client.create_from_slice(f32::as_bytes(basis));
    let out_handle = client.empty(total_elements * std::mem::size_of::<f32>());

    let geom = LaunchGeometry::elementwise(client, total_elements);
    unsafe {
        temporal_basis_reconstruct_kernel::launch::<R>(
            client,
            geom.cube_count,
            geom.cube_dim,
            ArrayArg::from_raw_parts(coeff_handle, coeffs.len()),
            ArrayArg::from_raw_parts(basis_handle, basis.len()),
            ArrayArg::from_raw_parts(out_handle.clone(), total_elements),
            total_elements as u32,
            num_components as u32,
            window_len as u32,
        );
    }

    let bytes = client.read_one(out_handle).expect("VRAM read reconstructed snippets");
    f32::from_bytes(&bytes)[..total_elements].to_vec()
}

/// Executes universal template matched filtering across `[channels, samples]` on a CubeCL `ComputeClient<R>`.
pub fn execute_universal_template_filter<R: Runtime>(
    client: &ComputeClient<R>,
    trace: &[f32],
    channels: usize,
    samples: usize,
    templates: &[f32],
    num_templates: usize,
    window_len: usize,
    center_offset: usize,
) -> Vec<f32> {
    let total = channels * samples;
    if total == 0 || num_templates == 0 || window_len == 0 || samples < window_len {
        return vec![0.0; total];
    }

    let trace_handle = client.create_from_slice(f32::as_bytes(trace));
    let tpl_handle = client.create_from_slice(f32::as_bytes(templates));
    let out_handle = client.empty(total * std::mem::size_of::<f32>());

    let geom = LaunchGeometry::channels_samples(client, channels, samples);
    unsafe {
        universal_template_filter_kernel::launch::<R>(
            client,
            geom.cube_count,
            geom.cube_dim,
            ArrayArg::from_raw_parts(trace_handle, trace.len()),
            ArrayArg::from_raw_parts(tpl_handle, templates.len()),
            ArrayArg::from_raw_parts(out_handle.clone(), total),
            channels as u32,
            samples as u32,
            num_templates as u32,
            window_len as u32,
            center_offset as u32,
        );
    }

    let bytes = client.read_one(out_handle).expect("VRAM read template filter energy");
    f32::from_bytes(&bytes)[..total].to_vec()
}

/// [`ComputeTask`] wrapper for running [`execute_temporal_basis_project`] on any [`ComputeTarget`].
pub struct ProjectBasisTask<'a> {
    pub snippets: &'a [f32],
    pub num_spikes: usize,
    pub num_channels: usize,
    pub window_len: usize,
    pub basis: &'a [f32],
    pub num_components: usize,
}

impl ComputeTask for ProjectBasisTask<'_> {
    type Output = Vec<f32>;
    fn run<R: Runtime>(self, client: ComputeClient<R>) -> Self::Output {
        execute_temporal_basis_project(
            &client,
            self.snippets,
            self.num_spikes,
            self.num_channels,
            self.window_len,
            self.basis,
            self.num_components,
        )
    }
}

/// [`ComputeTask`] wrapper for running [`execute_temporal_basis_reconstruct`] on any [`ComputeTarget`].
pub struct ReconstructBasisTask<'a> {
    pub coeffs: &'a [f32],
    pub num_spikes: usize,
    pub num_channels: usize,
    pub num_components: usize,
    pub basis: &'a [f32],
    pub window_len: usize,
}

impl ComputeTask for ReconstructBasisTask<'_> {
    type Output = Vec<f32>;
    fn run<R: Runtime>(self, client: ComputeClient<R>) -> Self::Output {
        execute_temporal_basis_reconstruct(
            &client,
            self.coeffs,
            self.num_spikes,
            self.num_channels,
            self.num_components,
            self.basis,
            self.window_len,
        )
    }
}

/// [`ComputeTask`] wrapper for running [`execute_universal_template_filter`] on any [`ComputeTarget`].
pub struct TemplateFilterTask<'a> {
    pub trace: &'a [f32],
    pub channels: usize,
    pub samples: usize,
    pub templates: &'a [f32],
    pub num_templates: usize,
    pub window_len: usize,
    pub center_offset: usize,
}

impl ComputeTask for TemplateFilterTask<'_> {
    type Output = Vec<f32>;
    fn run<R: Runtime>(self, client: ComputeClient<R>) -> Self::Output {
        execute_universal_template_filter(
            &client,
            self.trace,
            self.channels,
            self.samples,
            self.templates,
            self.num_templates,
            self.window_len,
            self.center_offset,
        )
    }
}

/// Dispatches a [`ComputeTask`] on `target`, converting [`dsp_core::ComputeError`] into [`DspError`].
pub fn run_on_target<T: ComputeTask>(target: ComputeTarget, task: T) -> DspResult<T::Output> {
    target
        .run(task)
        .map_err(|e| DspError::Model(format!("compute runtime error on {target}: {e}")))
}
