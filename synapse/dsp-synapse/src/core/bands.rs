use serde::{Deserialize, Serialize};
use dsp_base::filter::FilterSpec;

/// Standard electrophysiology frequency bands.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NeuralBand {
    /// Action Potential (AP) spike band: 300 – 6,000 Hz.
    Ap,
    /// Local Field Potential (LFP) low-frequency band: 0.5 – 300 Hz.
    Lfp,
    /// Sharp-wave ripple band: 150 – 250 Hz.
    Ripple,
    /// Intramuscular EMG (motor-unit action potentials) band: 300 – 5,000 Hz (EMUsort's 5 kHz upper
    /// edge, O'Connell et al. 2026, with a 300 Hz lower edge).
    Emg,
}

impl NeuralBand {
    pub fn low_hz(&self) -> f64 {
        match self {
            NeuralBand::Ap => 300.0,
            NeuralBand::Lfp => 0.5,
            NeuralBand::Ripple => 150.0,
            NeuralBand::Emg => 300.0,
        }
    }

    pub fn high_hz(&self) -> f64 {
        match self {
            NeuralBand::Ap => 6000.0,
            NeuralBand::Lfp => 300.0,
            NeuralBand::Ripple => 250.0,
            NeuralBand::Emg => 5000.0,
        }
    }

    /// Zero-phase Butterworth band-pass (order 5) for this band.
    pub fn filter_spec(&self) -> FilterSpec {
        FilterSpec::bandpass(self.low_hz(), self.high_hz())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_neural_bands() {
        assert_eq!(NeuralBand::Ap.low_hz(), 300.0);
        assert_eq!(NeuralBand::Ap.high_hz(), 6000.0);
        let sos = NeuralBand::Ap.filter_spec().design(30000.0).unwrap();
        assert_eq!(sos.len(), 5);
        assert!(NeuralBand::Lfp.filter_spec().design(2500.0).is_ok());
    }
}
