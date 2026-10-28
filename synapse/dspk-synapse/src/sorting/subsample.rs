//! Choosing a subset of detected peaks to learn from (features, clusters, templates), as
//! SpikeInterface's `select_peaks(method="uniform")` (`sortingcomponents/peak_selection.py`, MIT),
//! the method SpyKING CIRCUS 2 and Tridesclous 2 use: random, without replacement, `n_peaks` in
//! total or at most `n_peaks` of each channel.
//!
//! Upstream's other methods (`smart_sampling_*`) are not used by these sorters and are left out:
//! after their quantile transform the values are already uniform, so accepting with probability
//! `1 − s` favours low SNRs and low coordinates rather than flattening anything.
//!
//! On the host: it reads only peak times and channels, already on the host after detection. The
//! result is the chosen peaks' indices, sorted by sample (as upstream).

use super::kmeans::Rng;

/// Settings of [`subsample_peaks`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SubsampleOptions {
    /// Peaks kept: in total, or of each channel with `per_channel`.
    pub n_peaks: usize,
    /// At most `n_peaks` of each channel (channels with fewer keep all theirs).
    pub per_channel: bool,
    pub seed: u64,
}

/// Indices of the chosen peaks (peak `i` at `samples[i]` on `channels[i]`), sorted by sample.
///
/// # Panics
///
/// If `samples` and `channels` differ in length.
pub fn subsample_peaks(samples: &[u64], channels: &[usize], opts: &SubsampleOptions) -> Vec<usize> {
    assert_eq!(samples.len(), channels.len(), "one channel per peak");
    let mut rng = Rng(opts.seed);
    let mut chosen = if opts.per_channel {
        // Channels in increasing order, as upstream iterates `np.unique`
        let mut by: std::collections::BTreeMap<usize, Vec<usize>> = std::collections::BTreeMap::new();
        for (i, &c) in channels.iter().enumerate() {
            by.entry(c).or_default().push(i);
        }
        by.into_values().flat_map(|group| uniform(group, opts.n_peaks, &mut rng)).collect()
    } else {
        uniform((0..samples.len()).collect(), opts.n_peaks, &mut rng)
    };
    chosen.sort_unstable_by_key(|&i| (samples[i], i));
    chosen
}

/// `n` of `pool` at random without replacement (all when `n ≥ pool.len()`): a partial
/// Fisher–Yates shuffle, whose first `n` positions are a uniform sample.
fn uniform(mut pool: Vec<usize>, n: usize, rng: &mut Rng) -> Vec<usize> {
    if n >= pool.len() {
        return pool;
    }
    for k in 0..n {
        let j = k + (rng.next_u64() % (pool.len() - k) as u64) as usize;
        pool.swap(k, j);
    }
    pool.truncate(n);
    pool
}

#[cfg(test)]
mod tests {
    use super::*;

    fn peaks() -> (Vec<u64>, Vec<usize>) {
        ((0..1000).map(|i| (i * 7919 % 1000) as u64).collect(), (0..1000).map(|i| i % 4).collect())
    }

    #[test]
    fn takes_n_sorted_by_sample_and_reproducible() {
        let (samples, channels) = peaks();
        let opts = SubsampleOptions { n_peaks: 100, per_channel: false, seed: 1 };
        let a = subsample_peaks(&samples, &channels, &opts);
        assert_eq!(a.len(), 100);
        assert!(a.windows(2).all(|w| samples[w[0]] <= samples[w[1]]), "sorted by sample");
        let mut unique = a.clone();
        unique.dedup();
        assert_eq!(unique.len(), 100, "without replacement");
        assert_eq!(a, subsample_peaks(&samples, &channels, &opts), "same seed, same subset");
        assert_ne!(a, subsample_peaks(&samples, &channels, &SubsampleOptions { seed: 2, ..opts }));
    }

    #[test]
    fn per_channel_caps_each_channel() {
        let (samples, mut channels) = peaks();
        channels[0] = 9; // a channel with a single peak keeps it
        let opts = SubsampleOptions { n_peaks: 30, per_channel: true, seed: 1 };
        let c = subsample_peaks(&samples, &channels, &opts);
        for ch in 0..4 {
            assert_eq!(c.iter().filter(|&&i| channels[i] == ch).count(), 30);
        }
        assert!(c.contains(&0));
    }

    #[test]
    fn asking_for_more_than_there_are_keeps_all() {
        let (samples, channels) = peaks();
        let all = subsample_peaks(&samples, &channels, &SubsampleOptions { n_peaks: 5000, per_channel: false, seed: 0 });
        assert_eq!(all.len(), 1000);
    }

    /// Every peak is equally likely: over many seeds each is chosen about n / N of the time.
    #[test]
    fn every_peak_is_equally_likely() {
        let samples: Vec<u64> = (0..20).collect();
        let channels = vec![0usize; 20];
        let mut hits = [0u32; 20];
        for seed in 0..4000 {
            for i in subsample_peaks(&samples, &channels, &SubsampleOptions { n_peaks: 5, per_channel: false, seed }) {
                hits[i] += 1;
            }
        }
        // Expected 1000 each (4000 · 5 / 20)
        assert!(hits.iter().all(|&h| (850..=1150).contains(&h)), "{hits:?}");
    }
}
