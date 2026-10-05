pub mod gmm;
pub mod matching_pursuit;

pub use gmm::{gmm_e_step_kernel, gmm_mean_sums_kernel, gmm_scatter_kernel};
pub use matching_pursuit::{mp_gather_picks_kernel, mp_score_kernel, mp_subtract_kernel};
