//! Element-wise operations and statistics.
//!
//! On the device: scaling and clamping, unpacking stored integer samples to floats, per-channel
//! mean, standard deviation and noise. On the host: noise estimators ([`estimate_noise_std`],
//! median absolute deviation), percentiles and histograms ([`histogram`](mod@histogram)), running moments
//! ([`moments`]), lagged cross-correlation ([`xcorr`]), windows and `sinc` ([`windows`]).

pub mod kernels;
pub mod scaling;
pub mod clamp;
pub mod histogram;
pub mod moments;
pub mod unpack;
pub mod stats;
pub mod windows;
pub mod xcorr;

pub use scaling::execute_scaling;
pub use clamp::execute_clamp;
pub use unpack::{execute_unpack_stored, stored_word_bytes, stored_words, upload_stored, write_stored, write_stored_owned};
pub use stats::{
    estimate_noise_rms, estimate_noise_std, estimate_noise_trimmed, execute_channel_mean_std,
    execute_channel_noise_std,
    interquartile_range, peak_to_peak, standard_error,
};
pub use histogram::{bin_centers, histogram, percentile};
pub use moments::RunningMoments;
pub use xcorr::{cross_correlation, lagged_dot, parabolic_vertex_offset, peak_lag, LagPeak};
pub use windows::{bessel_i0, blackman_window, gaussian_window, hamming_window, hann_window, kaiser_window, sinc};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::buffer;
    use cubecl::prelude::*;

    fn scaling(client: &Client) {
        let input = buffer::upload(client, &[1.0f32, 2.0, 3.0, 4.0]);
        let output = buffer::empty::<f32>(client, 4);
        execute_scaling::<f32>(client, &input, &output, 4, 2.5, 10.0);
        assert_eq!(buffer::download::<f32>(client, output), vec![12.5, 15.0, 17.5, 20.0]);
    }
    runtime_test!(test_scaling_kernel, scaling);

    fn clamp(client: &Client) {
        let input = buffer::upload(client, &[-10.0f32, 5.0, 20.0, 0.0]);
        let output = buffer::empty::<f32>(client, 4);
        execute_clamp::<f32>(client, &input, &output, 4, -2.0, 10.0);
        assert_eq!(buffer::download::<f32>(client, output), vec![-2.0, 5.0, 10.0, 0.0]);
    }
    runtime_test!(test_clamp_kernel, clamp);
}
