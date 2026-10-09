//! Kilosort4's global merges, after the clustering of the matched spikes (paper, Methods,
//! *Global merges*, written from its description):
//!
//! Units are taken in decreasing spike count. For a unit, every other unit still in the pool whose
//! waveform similarity (correlation maximized over lags) is above `min_similarity` (0.5) is a
//! candidate, tested from the most to the least similar: the pair merges when its
//! cross-correlogram is refractory ([`dsp_synapse::metrics::ccg_refractory`]: the two never fire
//! within a few ms of each other, one neuron). A merged unit is tested again against the pool. When
//! nothing merges any more, the unit is complete and leaves the pool.
//!
//! Similarities come from the units' templates in PC space, as for the learned templates
//! (`learned::pair_similarities`: one `matmul` on the device, the `units²` maxima back), recomputed
//! after every merge. Choices of ours where the paper is silent: a merged template is the
//! spike-count-weighted mean of the two, without a relative shift.

use cubecl::prelude::Client;
use dsp_synapse::metrics::{ccg_refractory, RefractoryOptions};

use super::clustering::SpikeClusters;
use super::learned::pair_similarities;
use super::matching::lagged_pc_products;
use super::templates::UniversalTemplates;

/// Settings of [`global_merges`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GlobalMergeOptions {
    /// Off: the units of the clustering stand.
    pub enabled: bool,
    /// Candidates need at least this waveform similarity (correlation over lags).
    pub min_similarity: f64,
    /// Cross-correlogram test of a candidate pair.
    pub refractory: RefractoryOptions,
}

impl Default for GlobalMergeOptions {
    fn default() -> Self {
        Self { enabled: true, min_similarity: 0.5, refractory: RefractoryOptions::default() }
    }
}

/// Units after the global merges, and how many merges happened.
#[derive(Debug, Clone, PartialEq)]
pub struct MergedClusters {
    pub clusters: SpikeClusters,
    pub merges: usize,
}

/// Global merges of `clusters`, whose spikes fall at recording samples `samples` (module docs).
pub fn global_merges(
    client: &Client,
    clusters: &SpikeClusters,
    samples: &[u64],
    universal: &UniversalTemplates,
    sample_rate_hz: f64,
    opts: &GlobalMergeOptions,
) -> MergedClusters {
    assert_eq!(samples.len(), clusters.labels.len(), "one sample per spike");
    let (units, channels, np, nt) = (clusters.n_units, clusters.channels, clusters.n_pcs, universal.nt);
    let size = channels * np;
    if !opts.enabled || units < 2 {
        return MergedClusters { clusters: clusters.clone(), merges: 0 };
    }
    // Spike trains (sorted) and templates per unit
    let mut trains: Vec<Vec<u64>> = vec![Vec::new(); units];
    for (&u, &t) in clusters.labels.iter().zip(samples) {
        trains[u as usize].push(t);
    }
    trains.iter_mut().for_each(|t| t.sort_unstable());
    let mut templates: Vec<Vec<f32>> = (0..units).map(|u| clusters.templates[u * size..(u + 1) * size].to_vec()).collect();
    // `into[u]`: the unit `u` was merged into (itself while it stands)
    let mut into: Vec<usize> = (0..units).collect();
    let mut alive: Vec<bool> = trains.iter().map(|t| !t.is_empty()).collect();
    let mut complete = vec![false; units];

    let wtw = lagged_pc_products(&universal.wpca, np, nt);
    let lags = 2 * nt - 1;
    let mut order: Vec<usize> = (0..units).filter(|&u| alive[u]).collect();
    order.sort_by(|&a, &b| trains[b].len().cmp(&trains[a].len()).then(a.cmp(&b)));
    let mut merges = 0;
    for a in order {
        if !alive[a] {
            continue;
        }
        loop {
            // Similarities of `a` with the pool (alive, not complete)
            let pool: Vec<usize> = (0..units).filter(|&b| b != a && alive[b] && !complete[b]).collect();
            if pool.is_empty() {
                break;
            }
            let packed: Vec<f32> = std::iter::once(a).chain(pool.iter().copied()).flat_map(|u| templates[u].iter().copied()).collect();
            let n = pool.len() + 1;
            let sim = pair_similarities(client, &packed, n, channels, np, &wtw, lags);
            let norm = |i: usize| (sim[i * n + i].max(0.0) as f64).sqrt();
            let mut candidates: Vec<(f64, usize)> = (1..n)
                .filter(|&i| norm(0) > 0.0 && norm(i) > 0.0)
                .map(|i| (sim[i] as f64 / (norm(0) * norm(i)), pool[i - 1]))
                .filter(|&(r, _)| r > opts.min_similarity)
                .collect();
            candidates.sort_by(|x, y| y.0.total_cmp(&x.0).then(x.1.cmp(&y.1)));
            let partner = candidates.into_iter().map(|(_, b)| b).find(|&b| ccg_refractory(&trains[a], &trains[b], sample_rate_hz, &opts.refractory).1);
            let Some(b) = partner else { break };
            // Merge `b` into `a`, then test `a` again
            let (ca, cb) = (trains[a].len() as f32, trains[b].len() as f32);
            let tb = std::mem::take(&mut templates[b]);
            templates[a].iter_mut().zip(&tb).for_each(|(x, y)| *x = (ca * *x + cb * y) / (ca + cb));
            let tb = std::mem::take(&mut trains[b]);
            let mut merged = Vec::with_capacity(trains[a].len() + tb.len());
            let (mut i, mut j) = (0, 0);
            while i < trains[a].len() || j < tb.len() {
                if j == tb.len() || (i < trains[a].len() && trains[a][i] <= tb[j]) {
                    merged.push(trains[a][i]);
                    i += 1;
                } else {
                    merged.push(tb[j]);
                    j += 1;
                }
            }
            trains[a] = merged;
            alive[b] = false;
            into[b] = a;
            merges += 1;
        }
        complete[a] = true;
    }

    // Compact labels in the order of the surviving units
    let root = |mut u: usize| {
        while into[u] != u {
            u = into[u];
        }
        u
    };
    let survivors: Vec<usize> = (0..units).filter(|&u| alive[u]).collect();
    let mut new_id = vec![u32::MAX; units];
    for (i, &u) in survivors.iter().enumerate() {
        new_id[u] = i as u32;
    }
    let labels = clusters.labels.iter().map(|&u| new_id[root(u as usize)]).collect();
    let templates = survivors.iter().flat_map(|&u| templates[u].iter().copied()).collect();
    MergedClusters { clusters: SpikeClusters { labels, n_units: survivors.len(), templates, ..clusters.clone() }, merges }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FS: f64 = 30_000.0;

    /// A train with a 3 ms dead time between spikes, `period` samples apart on average.
    fn train(offset: u64, period: u64, n: u64) -> Vec<u64> {
        (0..n).map(|i| offset + i * period + (i * 7919 % (period / 2))).collect()
    }

    fn universal() -> UniversalTemplates {
        let (np, nt) = (3usize, 9usize);
        let mut wpca = vec![0.0f32; np * nt];
        for p in 0..np {
            wpca[p * nt + 2 + 2 * p] = 1.0;
        }
        UniversalTemplates { nt, n_pcs: np, n_templates: 2, wtemp: wpca[..2 * nt].to_vec(), wpca }
    }

    /// Two halves of one neuron (same template, interleaved spikes: refractory CCG) merge; a unit
    /// with the same template that fires independently does not.
    #[test]
    fn halves_of_one_neuron_merge_and_independent_units_do_not() {
        let Ok(target) = dsp_core::compute::ComputeTarget::from_env() else { return };
        let client = target.client().expect("client");
        let (channels, np) = (4usize, 3usize);
        let t: Vec<f32> = [1.0f32, 0.2, 0.0, 0.5, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0].to_vec();
        // Unit 0 and 1: alternate spikes of one train (every spike ≥ 3 ms from the next)
        let whole = train(0, 3_000, 4_000);
        let (a, b): (Vec<u64>, Vec<u64>) = (whole.iter().step_by(2).copied().collect(), whole.iter().skip(1).step_by(2).copied().collect());
        // Unit 2: same template, an unrelated train (overlaps the others in time)
        let c = train(1_234, 2_917, 4_000);
        let mut labels = Vec::new();
        let mut samples = Vec::new();
        for (u, s) in [(0u32, &a), (1, &b), (2, &c)] {
            labels.extend(std::iter::repeat_n(u, s.len()));
            samples.extend(s.iter().copied());
        }
        let templates: Vec<f32> = t.iter().chain(&t).chain(&t).copied().collect();
        let clusters = SpikeClusters { labels, n_units: 3, templates, channels, n_pcs: np, sections: 1 };
        let out = global_merges(&client, &clusters, &samples, &universal(), FS, &GlobalMergeOptions::default());
        assert_eq!(out.merges, 1, "{out:?}");
        assert_eq!(out.clusters.n_units, 2);
        let l = &out.clusters.labels;
        assert_eq!(l[0], l[a.len()], "the two halves are one unit");
        assert_ne!(l[0], l[a.len() + b.len()], "the independent unit stays apart");
    }

    #[test]
    fn disabled_keeps_the_units() {
        let Ok(target) = dsp_core::compute::ComputeTarget::from_env() else { return };
        let client = target.client().expect("client");
        let clusters = SpikeClusters { labels: vec![0, 1], n_units: 2, templates: vec![1.0; 2 * 12], channels: 4, n_pcs: 3, sections: 1 };
        let opts = GlobalMergeOptions { enabled: false, ..Default::default() };
        let out = global_merges(&client, &clusters, &[0, 10], &universal(), FS, &opts);
        assert_eq!(out.clusters, clusters);
    }
}
