//! Lag-Maximized Template Cosine Similarity & Auto-Merge Matrix (`similarity.rs`).
//!
//! Computes the maximum normalized cross-correlation (cosine similarity) between
//! multi-channel `WaveformTemplate` pairs over integer lags $[-\Delta_{\max}, +\Delta_{\max}]$
//! to identify over-split clusters for merging.

use dsp_base::math::{cross_correlation, peak_lag};

use crate::metrics::WaveformTemplate;

/// Computes the lag-maximized cosine similarity in `[-1.0, 1.0]` and best sample shift
/// between two multi-channel `WaveformTemplate`s. Rows are paired by recording channel; channels
/// covered by only one template contribute to its norm but not to the dot product.
pub fn template_max_cosine_similarity(
    a: &WaveformTemplate,
    b: &WaveformTemplate,
    max_lag_samples: usize,
) -> (f32, isize) {
    let len = a.num_samples.min(b.num_samples);
    let pairs: Vec<(&[f32], &[f32])> = a
        .channel_ids
        .iter()
        .enumerate()
        .filter_map(|(r, &c)| b.channel_row(c).map(|rb| (a.row(r), rb)))
        .collect();
    if pairs.is_empty() || len == 0 {
        return (0.0, 0);
    }

    let norm_a = a.mean.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-8);
    let norm_b = b.mean.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-8);
    let denom = norm_a * norm_b;

    // Cross-correlation summed over the shared channels
    let max_lag = max_lag_samples.min(len / 2);
    let mut corr = vec![0.0f32; 2 * max_lag + 1];
    for (ra, rb) in &pairs {
        for (c, v) in corr.iter_mut().zip(cross_correlation(&ra[..len], &rb[..len], max_lag)) {
            *c += v;
        }
    }
    match peak_lag(&corr, max_lag) {
        Some(p) => ((p.value / denom).clamp(-1.0, 1.0), p.lag),
        None => (0.0, 0),
    }
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
