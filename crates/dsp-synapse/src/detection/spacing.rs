//! Minimal spacing between spikes on one channel, shared by every detector: candidates come from
//! dsp-base peak finding (`dsp_base::peaks`), each detector accepts them by its own score, and the
//! survivors are spaced by [`SpikeSpacing`].

use dsp_base::peaks::{select_by_distance, DistanceRule, Polarity};

use super::threshold::SpikePolarity;

/// Spikes on one channel are kept more than `refractory_samples` apart (a minimal distance of
/// `refractory_samples + 1`); `rule` decides which of two nearer spikes stays.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpikeSpacing {
    pub refractory_samples: usize,
    /// Default [`DistanceRule::LocallyExclusive`]: a spike stays unless a larger one is nearer,
    /// which chunked (streaming) detection reproduces exactly. [`DistanceRule::Scipy`] follows
    /// `scipy.signal.find_peaks` `distance`.
    pub rule: DistanceRule,
}

impl SpikeSpacing {
    /// `refractory_samples` with the default rule.
    pub fn new(refractory_samples: usize) -> Self {
        Self { refractory_samples, rule: DistanceRule::LocallyExclusive }
    }

    /// `refractory_ms` at `sample_rate_hz` (at least one sample) with `rule`.
    pub fn from_ms(refractory_ms: f64, sample_rate_hz: f64, rule: DistanceRule) -> Self {
        Self { refractory_samples: ((sample_rate_hz * refractory_ms * 1e-3).round() as usize).max(1), rule }
    }

    /// Minimal sample distance between kept spikes.
    pub fn distance(&self) -> usize {
        self.refractory_samples + 1
    }

    /// Keeps `(sample, score, payload)` candidates (any order) spaced by `self`, the larger
    /// `score` winning; returns the survivors ascending by sample. Duplicated samples keep the
    /// larger score.
    pub fn select<P>(&self, mut candidates: Vec<(usize, f32, P)>) -> Vec<(usize, f32, P)> {
        candidates.sort_by_key(|c| c.0);
        let samples: Vec<usize> = candidates.iter().map(|c| c.0).collect();
        let scores: Vec<f32> = candidates.iter().map(|c| c.1).collect();
        let keep = select_by_distance(&samples, &scores, self.distance(), self.rule);
        candidates.into_iter().zip(keep).filter_map(|(c, k)| k.then_some(c)).collect()
    }
}

impl From<SpikePolarity> for Polarity {
    fn from(p: SpikePolarity) -> Self {
        match p {
            SpikePolarity::Negative => Polarity::Negative,
            SpikePolarity::Positive => Polarity::Positive,
            SpikePolarity::Both => Polarity::Both,
        }
    }
}
