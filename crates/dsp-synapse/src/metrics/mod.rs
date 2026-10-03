pub mod comparison;
pub mod correlogram;
pub mod evoked;
pub mod firing;
pub mod isolation;
pub mod rate;

pub use crate::core::template::{UnitQualityLabel, WaveformTemplate, compute_mean_template};
pub use comparison::{
    PairwiseTrainMatch, SortingComparison, UnitMatchSummary, compare_sortings,
    compare_spike_trains,
};
pub use correlogram::{Correlogram, compute_autocorrelogram, compute_crosscorrelogram};
pub use evoked::{
    MepMetrics, PsthResult, StimulusTriggeredAverage, compute_psth,
    compute_stimulus_triggered_average, quantify_mep,
};
pub use firing::{
    IsiMetrics, compute_amplitude_cutoff, compute_amplitude_cutoff_with, compute_isi_violations,
    compute_llobet_contamination, compute_presence_ratio, count_refractory_violations, isi_histogram,
};
pub use isolation::{
    compute_d_prime, compute_isolation_distance, compute_silhouette_score, compute_snr,
};
pub use rate::{
    BurstEpoch, FiringRateCurve, compute_instantaneous_firing_rate, detect_burst_epochs,
};
