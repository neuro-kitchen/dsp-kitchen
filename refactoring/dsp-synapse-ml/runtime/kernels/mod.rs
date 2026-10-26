//! Custom CubeCL (`#[cube]`) compute kernels for `dsp-synapse-ml`.

pub mod projection;

pub use projection::{
    ProjectBasisTask, ReconstructBasisTask, TemplateFilterTask,
    execute_temporal_basis_project, execute_temporal_basis_reconstruct,
    execute_universal_template_filter, run_on_target,
    temporal_basis_project_kernel, temporal_basis_reconstruct_kernel,
    universal_template_filter_kernel,
};
