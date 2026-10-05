//! Kilosort4 universal templates: the single-channel temporal basis `wPCA` (`[n_pcs, nt]`) and the
//! universal templates `wTEMP` (`[n_templates, nt]`), learned from the recording
//! ([`learn_universal_templates`], Kilosort4's default) or loaded from Kilosort4's predefined
//! `wTEMP.npz` ([`UniversalTemplates::from_npz`]).
//!
//! Learning, as in the paper / upstream: isolated single-channel threshold crossings of the
//! whitened data → clips scaled by one common factor → `wPCA` = top right singular vectors of the
//! clip matrix → (optionally, EMUsort) HDBSCAN outlier removal → `wTEMP` = k-means centres,
//! rows L2-normalized.

use std::path::Path;

use cubecl::prelude::*;
use dsp_base::linalg::{symmetric_eigen_host, EigenOptions};
use dsp_core::{DspError, DspResult};
use dsp_io::container::npy::read_npz;
use dsp_synapse::sorting::{hdbscan, kmeans, KMeansOptions};

/// Half-window of the local-maximum test of clip detection: ±4 samples × ±5 channel indices.
pub const CLIP_LOCAL_SAMPLES: usize = 4;
pub const CLIP_LOCAL_CHANNELS: usize = 5;
/// Isolation window: no other peak within ±6 channel indices × ±`nt / 2` samples.
pub const CLIP_ISOLATION_CHANNELS: usize = 6;
/// Most clips gathered for learning.
pub const MAX_CLIPS: usize = 500_000;

/// `wPCA` and `wTEMP`, both `[rows, nt]` row-major.
#[derive(Debug, Clone, PartialEq)]
pub struct UniversalTemplates {
    pub nt: usize,
    pub n_pcs: usize,
    pub n_templates: usize,
    pub wpca: Vec<f32>,
    pub wtemp: Vec<f32>,
}

impl UniversalTemplates {
    /// Kilosort4's predefined `wTEMP.npz` (arrays `wPCA`, `wTEMP`; `wPCA` is stored in Fortran
    /// order, which the reader converts).
    pub fn from_npz(path: &Path) -> DspResult<Self> {
        let arrays = read_npz::<f32>(path)?;
        let get = |name: &str| {
            arrays.get(name).ok_or_else(|| DspError::UnsupportedFormat(format!("{}: no array '{name}'", path.display())))
        };
        let (wpca, wtemp) = (get("wPCA")?, get("wTEMP")?);
        let (Ok([n_pcs, nt]), Ok([n_templates, nt_t])) = (wpca.dims::<2>(path), wtemp.dims::<2>(path)) else {
            return Err(DspError::UnsupportedFormat(format!("{}: wPCA / wTEMP must be 2-D", path.display())));
        };
        if nt != nt_t {
            return Err(DspError::UnsupportedFormat(format!("{}: wPCA has {nt} samples, wTEMP {nt_t}", path.display())));
        }
        Ok(Self { nt, n_pcs, n_templates, wpca: wpca.data.clone(), wtemp: wtemp.data.clone() })
    }
}

/// How clips are found.
#[derive(Debug, Clone, PartialEq)]
pub struct ClipOptions {
    /// Samples per clip.
    pub nt: usize,
    /// Sample of the clip the peak sits at.
    pub nt0min: usize,
    /// Thresholds (whitened σ); the clips of all thresholds are pooled without duplicates.
    pub thresholds: Vec<f32>,
}

/// Isolated single-channel peaks of the whitened `[channels, samples]` batch `x` (channels in
/// probe order: neighbouring indices are neighbouring contacts), as `[n, nt]` clips appended to
/// `out` (until [`MAX_CLIPS`]). Returns the number added.
pub fn extract_clips(x: &[f32], channels: usize, samples: usize, opts: &ClipOptions, out: &mut Vec<f32>) -> usize {
    assert_eq!(x.len(), channels * samples);
    let nt = opts.nt;
    if samples < 2 * nt + 1 || nt == 0 {
        return 0;
    }
    let abs = |c: usize, t: usize| x[c * samples + t].abs();
    let window = |c: usize, t: usize, dc: usize, dt: usize| {
        let (c0, c1) = (c.saturating_sub(dc), (c + dc).min(channels - 1));
        let (t0, t1) = (t.saturating_sub(dt), (t + dt).min(samples - 1));
        (c0, c1, t0, t1)
    };
    let mut taken = std::collections::BTreeSet::new();
    let mut added = 0;
    for &th in &opts.thresholds {
        // Peaks: equal to the local max of |x| over ±4 samples × ±5 channels, above `th`
        let is_peak = |c: usize, t: usize| {
            let v = abs(c, t);
            if !(v > th) {
                return false;
            }
            let (c0, c1, t0, t1) = window(c, t, CLIP_LOCAL_CHANNELS, CLIP_LOCAL_SAMPLES);
            (c0..=c1).all(|cc| (t0..=t1).all(|tt| abs(cc, tt) <= v))
        };
        let peaks: Vec<(usize, usize)> = (0..channels).flat_map(|c| (nt..samples - nt).map(move |t| (c, t))).filter(|&(c, t)| is_peak(c, t)).collect();
        let peak_set: std::collections::BTreeSet<(usize, usize)> = peaks.iter().copied().collect();
        for &(c, t) in &peaks {
            // Isolated: the only peak within ±6 channels × ±nt/2 samples
            let (c0, c1, t0, t1) = window(c, t, CLIP_ISOLATION_CHANNELS, nt / 2);
            let others = peak_set.range((c0, t0)..=(c1, t1)).filter(|&&(cc, tt)| tt >= t0 && tt <= t1 && (cc, tt) != (c, t)).count();
            if others > 0 || !taken.insert((c, t)) || t < opts.nt0min || t - opts.nt0min + nt > samples {
                continue;
            }
            if out.len() / nt >= MAX_CLIPS {
                return added;
            }
            let start = c * samples + t - opts.nt0min;
            out.extend_from_slice(&x[start..start + nt]);
            added += 1;
        }
    }
    added
}

/// What [`learn_universal_templates`] does beyond Kilosort4.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LearnOptions {
    pub n_pcs: usize,
    pub n_templates: usize,
    /// EMUsort: HDBSCAN `min_cluster_size` for outlier removal before k-means (`None`: Kilosort4).
    pub outlier_min_cluster_size: Option<usize>,
    /// Seed of the k-means initialisations.
    pub seed: u64,
}

/// Fewest clips HDBSCAN outlier removal is run on (EMUsort: `max(20, min_cluster_size)`).
const MIN_CLIPS_FOR_OUTLIERS: usize = 20;
/// k-means initialisations (`KMeans(n_init = 10)` upstream).
const KMEANS_N_INIT: usize = 10;

/// `wPCA` and `wTEMP` from `[n, nt]` clips (see the module docs). The SVD runs as an eigen-
/// decomposition of the `nt × nt` Gram matrix on `client`'s device.
pub fn learn_universal_templates<R: Runtime>(client: &ComputeClient<R>, clips: &[f32], nt: usize, opts: &LearnOptions) -> DspResult<UniversalTemplates> {
    let n = clips.len() / nt.max(1);
    if n < opts.n_templates.max(opts.n_pcs) || nt == 0 {
        return Err(DspError::InvalidConfig(format!("{n} clips: too few to learn {} PCs and {} templates", opts.n_pcs, opts.n_templates)));
    }
    // One common scale: 1 / sqrt(std of the clip energies) (keeps relative amplitudes)
    let energies: Vec<f64> = clips.chunks_exact(nt).map(|c| c.iter().map(|&v| (v as f64).powi(2)).sum()).collect();
    let mean = energies.iter().sum::<f64>() / n as f64;
    let std = (energies.iter().map(|e| (e - mean).powi(2)).sum::<f64>() / n as f64).sqrt();
    let scale = if std > 0.0 { (1.0 / std.sqrt()) as f32 } else { 1.0 };
    let scaled: Vec<f32> = clips.iter().map(|&v| v * scale).collect();

    // wPCA: top right singular vectors (uncentred) = top eigenvectors of Cᵀ C
    let mut gram = vec![0.0f64; nt * nt];
    for c in scaled.chunks_exact(nt) {
        for i in 0..nt {
            let ci = c[i] as f64;
            for j in i..nt {
                gram[i * nt + j] += ci * c[j] as f64;
            }
        }
    }
    for i in 0..nt {
        for j in 0..i {
            gram[i * nt + j] = gram[j * nt + i];
        }
    }
    let eig = symmetric_eigen_host::<R, f32>(client, &gram, nt, EigenOptions::default());
    let mut wpca = vec![0.0f32; opts.n_pcs * nt];
    for p in 0..opts.n_pcs {
        for t in 0..nt {
            wpca[p * nt + t] = eig.vectors[t * nt + p] as f32;
        }
    }

    // EMUsort: drop HDBSCAN outliers before clustering
    let kept: Vec<f32> = match opts.outlier_min_cluster_size {
        Some(mcs) if n >= MIN_CLIPS_FOR_OUTLIERS.max(mcs) => {
            let labels = hdbscan(&scaled, n, nt, mcs);
            scaled.chunks_exact(nt).zip(&labels).filter(|(_, l)| **l >= 0).flat_map(|(c, _)| c.iter().copied()).collect()
        }
        _ => scaled,
    };
    let n_kept = kept.len() / nt;
    if n_kept < opts.n_templates {
        return Err(DspError::InvalidConfig(format!("{n_kept} clips after outlier removal: too few for {} templates", opts.n_templates)));
    }

    // wTEMP: k-means centres, rows L2-normalized
    let km = kmeans(&kept, n_kept, nt, opts.n_templates, &KMeansOptions { n_init: KMEANS_N_INIT, seed: opts.seed, ..Default::default() });
    let mut wtemp = km.centers;
    for row in wtemp.chunks_exact_mut(nt) {
        let norm = row.iter().map(|v| v * v).sum::<f32>().sqrt();
        if norm > 0.0 {
            row.iter_mut().for_each(|v| *v /= norm);
        }
    }
    Ok(UniversalTemplates { nt, n_pcs: opts.n_pcs, n_templates: opts.n_templates, wpca, wtemp })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn isolated_peaks_become_clips() {
        // 3 channels, one isolated trough on channel 1 and two close ones on channel 0
        let (channels, samples, nt) = (3usize, 400usize, 21usize);
        let mut x = vec![0.0f32; channels * samples];
        x[samples + 200] = -12.0;
        x[100] = -10.0;
        x[105] = -11.0;
        let mut clips = Vec::new();
        let n = extract_clips(&x, channels, samples, &ClipOptions { nt, nt0min: 7, thresholds: vec![6.0] }, &mut clips);
        assert_eq!(n, 1, "only the isolated peak");
        assert_eq!(clips[7], -12.0, "peak at nt0min");
        // A second threshold adds nothing new for the same peak
        let mut again = Vec::new();
        assert_eq!(extract_clips(&x, channels, samples, &ClipOptions { nt, nt0min: 7, thresholds: vec![6.0, 9.0] }, &mut again), 1);
    }
}
