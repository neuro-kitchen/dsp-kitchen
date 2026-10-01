pub mod dedup;
pub mod threshold;

pub use dedup::spatial_dedup_survival_kernel;
pub use threshold::{
    DetectionCarry, count_trough_candidates_kernel, execute_detect_spikes_in_vram,
    scan_candidate_counts_kernel, write_trough_candidates_kernel,
};
