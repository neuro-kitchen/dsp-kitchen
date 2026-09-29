//! Analytical Refractory Period Contamination Metrics (`contamination.rs`).
//!
//! Implements the Hill et al. (2011) and Llobet et al. (2022) false-positive
//! contamination estimators used in `spikeinterface.qualitymetrics.compute_refrac_period_violations`.

/// Computes the Hill et al. (2011) false-positive contamination rate $C \in [0.0, 1.0]$:
/// $$n_v = 2 (\tau_r - \tau_c) N^2 \frac{C (1 - C/2)}{T}$$
/// where $n_v$ is the number of refractory violations, $N$ is total spikes,
/// $T$ is recording duration in seconds, $\tau_r$ is refractory period, and $\tau_c$ is censored period.
pub fn compute_hill_contamination(
    num_spikes: usize,
    num_violations: usize,
    total_duration_sec: f64,
    refractory_period_ms: f64,
    censored_period_ms: f64,
) -> f32 {
    if num_spikes < 2 || num_violations == 0 || total_duration_sec <= 0.0 {
        return 0.0;
    }

    let effective_tau_sec = ((refractory_period_ms - censored_period_ms).max(1e-4)) * 1e-3;
    let n_f = num_spikes as f64;
    let denom = 2.0 * effective_tau_sec * n_f * n_f;
    let r = (num_violations as f64) * total_duration_sec / denom.max(1e-12);

    // Solve C - 0.5 C^2 = r  =>  C = 1 - sqrt(1 - 2r)
    let disc = 1.0 - 2.0 * r;
    if disc <= 0.0 {
        1.0
    } else {
        (1.0 - disc.sqrt()).clamp(0.0, 1.0) as f32
    }
}

/// Computes the Llobet et al. (2022) false-positive contamination rate $C \in [0.0, 1.0]$:
/// $$C = 1 - \sqrt{1 - \frac{n_v \cdot T}{N^2 (\tau_r - \tau_c)}}$$
pub fn compute_llobet_contamination(
    num_spikes: usize,
    num_violations: usize,
    total_duration_sec: f64,
    refractory_period_ms: f64,
    censored_period_ms: f64,
) -> f32 {
    if num_spikes < 2 || num_violations == 0 || total_duration_sec <= 0.0 {
        return 0.0;
    }

    let effective_tau_sec = ((refractory_period_ms - censored_period_ms).max(1e-4)) * 1e-3;
    let n_f = num_spikes as f64;
    let ratio = (num_violations as f64 * total_duration_sec)
        / (n_f * n_f * effective_tau_sec).max(1e-12);

    if ratio >= 1.0 {
        1.0
    } else {
        (1.0 - (1.0 - ratio).sqrt()).clamp(0.0, 1.0) as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hill_and_llobet_contamination_estimators() {
        assert_eq!(compute_hill_contamination(1000, 0, 100.0, 1.5, 0.2), 0.0);
        assert_eq!(compute_llobet_contamination(1000, 0, 100.0, 1.5, 0.2), 0.0);

        let c_hill = compute_hill_contamination(1000, 1, 100.0, 1.5, 0.2);
        let c_llobet = compute_llobet_contamination(1000, 1, 100.0, 1.5, 0.2);
        assert!(c_hill > 0.01 && c_hill < 0.15, "c_hill = {}", c_hill);
        assert!(c_llobet > 0.01 && c_llobet < 0.15, "c_llobet = {}", c_llobet);
    }
}
