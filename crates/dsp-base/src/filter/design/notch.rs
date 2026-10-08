use std::f64::consts::PI;

use super::sos::{Section, Sos};
use super::{FilterError, check_cutoff, check_sample_rate};

/// Second-order IIR notch at `freq_hz` with quality factor `q` (`scipy.signal.iirnotch`):
/// −3 dB bandwidth `freq_hz / q`, unity gain away from the notch.
///
/// # Errors
///
/// [`FilterError`] for a non-positive sample rate, `freq_hz` outside `(0, Nyquist)`, or a
/// non-positive or non-finite `q`.
pub fn notch_sos(freq_hz: f64, q: f64, sample_rate: f64) -> Result<Sos, FilterError> {
    check_sample_rate(sample_rate)?;
    check_cutoff(freq_hz, sample_rate)?;
    if !(q.is_finite() && q > 0.0) {
        return Err(FilterError::InvalidQ(q));
    }
    let w0 = 2.0 * PI * freq_hz / sample_rate;
    let bw = w0 / q;
    let beta = (bw / 2.0).tan();
    let gain = 1.0 / (1.0 + beta);
    let c = w0.cos();
    Ok(Sos::new(vec![Section {
        b: [gain, -2.0 * gain * c, gain],
        a: [1.0, -2.0 * gain * c, 2.0 * gain - 1.0],
    }]))
}
