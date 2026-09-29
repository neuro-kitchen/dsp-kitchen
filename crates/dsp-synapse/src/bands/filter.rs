use serde::{Deserialize, Serialize};
use dsp_base::filter::iir::{BandpassCoeffs, design_butterworth_bandpass_4th};

/// Standard electrophysiology frequency bands.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NeuralBand {
    /// Action Potential (AP) spike band: 300 – 6,000 Hz.
    Ap,
    /// Local Field Potential (LFP) low-frequency band: 0.5 – 300 Hz.
    Lfp,
    /// Sharp-wave ripple band: 150 – 250 Hz.
    Ripple,
}

impl NeuralBand {
    pub fn low_hz(&self) -> f64 {
        match self {
            NeuralBand::Ap => 300.0,
            NeuralBand::Lfp => 0.5,
            NeuralBand::Ripple => 150.0,
        }
    }

    pub fn high_hz(&self) -> f64 {
        match self {
            NeuralBand::Ap => 6000.0,
            NeuralBand::Lfp => 300.0,
            NeuralBand::Ripple => 250.0,
        }
    }

    /// Designs 4th-order cascaded Butterworth bandpass coefficients for this band.
    pub fn bandpass_coeffs(&self, sample_rate_hz: f64) -> BandpassCoeffs {
        design_butterworth_bandpass_4th(self.low_hz(), self.high_hz(), sample_rate_hz)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_neural_bands() {
        assert_eq!(NeuralBand::Ap.low_hz(), 300.0);
        assert_eq!(NeuralBand::Ap.high_hz(), 6000.0);
        let coeffs = NeuralBand::Ap.bandpass_coeffs(30000.0);
        assert!(coeffs.hp_sec.b0 > 0.0);
    }
}
