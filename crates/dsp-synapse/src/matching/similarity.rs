//! Lag-Maximized Template Cosine Similarity & Auto-Merge Matrix (`similarity.rs`).
//!
//! Computes the maximum normalized cross-correlation (cosine similarity) between
//! multi-channel `WaveformTemplate` pairs over integer lags $[-\Delta_{\max}, +\Delta_{\max}]$
//! to identify over-split clusters for merging.

use crate::metrics::WaveformTemplate;

/// Computes the lag-maximized cosine similarity in `[-1.0, 1.0]` and best sample shift
/// between two multi-channel `WaveformTemplate`s.
pub fn template_max_cosine_similarity(
    a: &WaveformTemplate,
    b: &WaveformTemplate,
    max_lag_samples: usize,
) -> (f32, isize) {
    let ch = a.num_channels.min(b.num_channels);
    let len = a.num_samples.min(b.num_samples);
    if ch == 0 || len == 0 {
        return (0.0, 0);
    }

    let norm_a = a.mean.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-8);
    let norm_b = b.mean.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-8);
    let denom = norm_a * norm_b;

    let max_lag = (max_lag_samples.min(len / 2)) as isize;
    let mut best_sim = f32::NEG_INFINITY;
    let mut best_lag = 0isize;

    for lag in -max_lag..=max_lag {
        let mut dot = 0.0f32;
        for c in 0..ch {
            let off_a = c * a.num_samples;
            let off_b = c * b.num_samples;
            for t in 0..len {
                let tb = t as isize + lag;
                if tb >= 0 && (tb as usize) < len {
                    dot += a.mean[off_a + t] * b.mean[off_b + (tb as usize)];
                }
            }
        }
        let sim = dot / denom;
        if sim > best_sim {
            best_sim = sim;
            best_lag = lag;
        }
    }

    (best_sim.clamp(-1.0, 1.0), best_lag)
}

/// Computes the symmetric `[U, U]` lag-maximized cosine similarity matrix across all unit templates.
pub fn compute_template_similarity_matrix(
    templates: &[WaveformTemplate],
    max_lag_samples: usize,
) -> Vec<f32> {
    let u = templates.len();
    let mut mat = vec![0.0f32; u * u];
    for i in 0..u {
        mat[i * u + i] = 1.0;
        for j in (i + 1)..u {
            let (sim, _lag) =
                template_max_cosine_similarity(&templates[i], &templates[j], max_lag_samples);
            mat[i * u + j] = sim;
            mat[j * u + i] = sim;
        }
    }
    mat
}

/// Suggests pairs of unit indices `(unit_i, unit_j, similarity)` whose template similarity
/// exceeds `merge_threshold` (e.g. `0.90`).
pub fn suggest_template_merges(
    templates: &[WaveformTemplate],
    max_lag_samples: usize,
    merge_threshold: f32,
) -> Vec<(usize, usize, f32)> {
    let u = templates.len();
    let mat = compute_template_similarity_matrix(templates, max_lag_samples);
    let mut merges = Vec::new();
    for i in 0..u {
        for j in (i + 1)..u {
            let sim = mat[i * u + j];
            if sim >= merge_threshold {
                merges.push((i, j, sim));
            }
        }
    }
    merges
}
