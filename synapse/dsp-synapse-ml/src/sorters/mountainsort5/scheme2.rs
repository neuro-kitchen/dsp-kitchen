//! MountainSort 5 scheme 2 after phase 1 (`schemes/sorting_scheme2.py`, `core/SnippetClassifier.py`):
//! one classifier per channel, trained on the phase-1 units, then every spike of the recording
//! classified.
//!
//! **Training** (channel `m`, its snippet mask `N(m)`): 200 noise snippets at times spread
//! uniformly over the training stretch (label 0, offset 0); and for each unit whose median template
//! reaches `0.4 · detect_threshold` on `m` (in the `detect_sign` sense), up to 200 of its snippets
//! (evenly spaced among its spikes in time order) on `N(m)`, rolled by `−offset` where `offset` is
//! the template's peak sample on `m` minus `T1` (label = unit). PCA of all of them (`classifier_npca`,
//! default `max(12, 3 · |N(m)|)` components, at most the number of snippets) and their features.
//!
//! **Classification**: a spike detected on channel `m` takes the label and offset of its **second**
//! nearest training point in `m`'s feature space (the first may be the snippet itself); its time
//! moves by `−offset`; label 0 is dropped; a unit's spikes within the time radius after one of its
//! kept spikes are dropped ([`remove_duplicate_events`]). Projections and the nearest-neighbour
//! search run on the device ([`fn@super::kernels::project_classifier_kernel`],
//! [`fn@super::kernels::second_nearest_kernel`]).
//!
//! Choices of ours. The classifiers' PCA is randomized (scikit-learn's iteration count) even below
//! 8000 features, where upstream's is exact: the nearest-neighbour distances depend only on the
//! subspace the components span, which the randomized solver finds to its tolerance, and the exact
//! solver costs a full eigendecomposition per channel (hundreds of components on a dense probe).
//! Memory: a unit's template is the median over its spikes on the channels within
//! the snippet mask radius of its peak channel only, so a unit trains the classifiers of those
//! channels only (upstream: every channel where its whole-probe template reaches the level; farther
//! than the mask radius from the peak it does not on spike data).

use cubecl::prelude::*;
use cubecl::server::Handle;
use dsp_base::core::buffer;
use dsp_base::linalg::{DeviceRows, TopComponents, TopComponentsOptions};
use dsp_core::compute::LaunchGeometry;
use dsp_synapse::features::ChannelNeighbourhoods;

use super::kernels::{project_classifier_kernel, second_nearest_kernel};
use super::snippets::MaskedSnippets;

/// Training settings of the classifiers (MountainSort 5 defaults in [`Default`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ClassifierParams {
    /// Phase 2's detection threshold: a unit trains channel `m` when its template reaches
    /// `0.4 ·` this there.
    pub detect_threshold: f32,
    pub detect_sign: i8,
    /// Noise snippets, and snippets per unit and channel.
    pub max_num_snippets_per_training_batch: usize,
    /// `None`: `max(12, 3 · |N(m)|)`.
    pub classifier_npca: Option<usize>,
    pub pca: TopComponentsOptions,
}

impl Default for ClassifierParams {
    fn default() -> Self {
        Self { detect_threshold: 5.5, detect_sign: -1, max_num_snippets_per_training_batch: 200, classifier_npca: None, pca: TopComponentsOptions::default() }
    }
}

/// Fraction of the threshold a template must reach on a channel (upstream's literal `0.4`).
pub const TRAINING_LEVEL: f32 = 0.4;

/// Upstream `subsample_snippets`: `max_num` of `n` evenly spaced (`i · n / max_num`), or all.
pub fn subsample_indices(n: usize, max_num: usize) -> Vec<usize> {
    if n > max_num { (0..max_num).map(|i| i * n / max_num).collect() } else { (0..n).collect() }
}

/// Upstream `np.floor(np.linspace(T1, n − T2 − 1, count))`: the noise snippet times of the
/// training stretch (`n` samples).
pub fn noise_times(n: usize, n_before: usize, n_after: usize, count: usize) -> Vec<usize> {
    let (a, b) = (n_before as f64, n as f64 - n_after as f64 - 1.0);
    match count {
        0 => Vec::new(),
        1 => vec![a as usize],
        _ => (0..count).map(|i| (a + (b - a) * i as f64 / (count - 1) as f64).floor() as usize).collect(),
    }
}

/// The trained classifiers of every channel, on the host and the device (module docs).
pub struct Classifiers {
    pub n_mask: Vec<u32>,
    pub n_comp: Vec<u32>,
    pub n_train: Vec<u32>,
    /// Training labels (0: noise) and offsets, concatenated over channels (`train_off`).
    pub labels: Vec<u32>,
    pub offsets: Vec<i32>,
    pub train_off: Vec<u32>,
    pub max_comp: usize,
    width: usize,
    max_neighbours: usize,
    h: DeviceTables,
}

struct DeviceTables {
    n_mask: Handle,
    n_comp: Handle,
    n_train: Handle,
    mean_off: Handle,
    comp_off: Handle,
    feat_off: Handle,
    means: Handle,
    components: Handle,
    features: Handle,
    lens: [usize; 3],
}

impl Classifiers {
    /// Trains every channel's classifier (module docs).
    ///
    /// - `mask`: the snippet mask of each channel (the rows of phase 2's snippets).
    /// - `noise`: the noise snippets, dense on every channel (one row of all channels).
    /// - `templates`: `[K, T, M]` unit templates (zeros where not computed).
    /// - `unit_snippets`: each unit's subsampled snippets (row = unit index `0..K`, in time order),
    ///   on channels covering the masks of the channels the unit may train.
    pub fn fit(
        client: &Client,
        mask: &ChannelNeighbourhoods,
        noise: &MaskedSnippets,
        templates: &[f32],
        k: usize,
        unit_snippets: &MaskedSnippets,
        params: &ClassifierParams,
    ) -> Self {
        let (t, m) = (noise.width, noise.channels);
        let n_before = noise.n_before as i32;
        let level = TRAINING_LEVEL * params.detect_threshold;
        let signed = |x: f32| match params.detect_sign {
            s if s < 0 => -x,
            s if s > 0 => x,
            _ => x.abs(),
        };
        let mut members: Vec<Vec<usize>> = vec![Vec::new(); k];
        for (i, &u) in unit_snippets.event_channels.iter().enumerate() {
            members[u as usize].push(i);
        }
        let noise_dense: Vec<Vec<f32>> = (0..noise.len()).map(|i| noise.dense(i)).collect();
        let unit_dense: Vec<Vec<f32>> = (0..unit_snippets.len()).map(|i| unit_snippets.dense(i)).collect();

        let mut out = Self {
            n_mask: Vec::with_capacity(m),
            n_comp: Vec::with_capacity(m),
            n_train: Vec::with_capacity(m),
            labels: Vec::new(),
            offsets: Vec::new(),
            train_off: Vec::with_capacity(m + 1),
            max_comp: 1,
            width: t,
            max_neighbours: mask.max_neighbours,
            h: DeviceTables::empty(client),
        };
        let (mut means, mut components, mut features) = (Vec::new(), Vec::new(), Vec::new());
        let (mut mean_off, mut comp_off, mut feat_off) = (Vec::new(), Vec::new(), Vec::new());
        out.train_off.push(0);
        for ch in 0..m {
            let nb: Vec<usize> = mask.of(ch).collect();
            let nm = nb.len();
            let d = t * nm;
            let mut rows: Vec<f32> = Vec::new();
            let (mut labels, mut offsets) = (Vec::new(), Vec::new());
            let take = |dense: &[f32], shift: i32, rows: &mut Vec<f32>| {
                for s in 0..t {
                    let from = (s as i32 + shift).rem_euclid(t as i32) as usize;
                    rows.extend(nb.iter().map(|&c| dense[from * m + c]));
                }
            };
            for dense in &noise_dense {
                take(dense, 0, &mut rows);
                labels.push(0);
                offsets.push(0);
            }
            for u in 0..k {
                let tp = &templates[u * t * m..(u + 1) * t * m];
                let (mut best, mut at) = (f32::NEG_INFINITY, 0usize);
                for s in 0..t {
                    let v = signed(tp[s * m + ch]);
                    if v > best {
                        best = v;
                        at = s;
                    }
                }
                if best < level {
                    continue;
                }
                let offset = at as i32 - n_before;
                // np.roll(·, −offset): sample s takes s + offset
                for &i in &members[u] {
                    take(&unit_dense[i], offset, &mut rows);
                    labels.push(u as u32 + 1);
                    offsets.push(offset);
                }
            }
            let l = labels.len();
            let npca = params.classifier_npca.unwrap_or((3 * nm).max(12));
            // The rows go to the device once for every pass of the fit and the transform
            let resident = buffer::upload(client, &rows);
            let index: Vec<u32> = (0..l as u32).collect();
            let src = DeviceRows { data: &resident, len: rows.len(), dim: d, rows: &index };
            let pca = TopComponents::fit(client, &src, npca, &params.pca);
            let feats = pca.transform(client, &src, params.pca.batch_elements);
            mean_off.push(means.len() as u32);
            comp_off.push(components.len() as u32);
            feat_off.push(features.len() as u32);
            means.extend_from_slice(&pca.mean);
            // `[d, k]`: component c of feature f at f · k + c
            components.extend((0..d * pca.k).map(|e| pca.components[(e % pca.k) * d + e / pca.k]));
            features.extend_from_slice(&feats);
            out.n_mask.push(nm as u32);
            out.n_comp.push(pca.k as u32);
            out.n_train.push(l as u32);
            out.max_comp = out.max_comp.max(pca.k);
            out.labels.extend(labels);
            out.offsets.extend(offsets);
            out.train_off.push(out.labels.len() as u32);
        }
        let up = |v: &[u32]| buffer::upload(client, if v.is_empty() { &[0u32][..] } else { v });
        let upf = |v: &[f32]| buffer::upload(client, if v.is_empty() { &[0.0f32][..] } else { v });
        out.h = DeviceTables {
            n_mask: up(&out.n_mask),
            n_comp: up(&out.n_comp),
            n_train: up(&out.n_train),
            mean_off: up(&mean_off),
            comp_off: up(&comp_off),
            feat_off: up(&feat_off),
            means: upf(&means),
            components: upf(&components),
            features: upf(&features),
            lens: [means.len().max(1), components.len().max(1), features.len().max(1)],
        };
        out
    }

    /// Label (0: noise) and offset of each event (module docs). `snippets`: the events' masked
    /// snippets on the device (`[events, T, max_neighbours]`, rows of the training `mask`).
    pub fn classify(&self, client: &Client, snippets: &Handle, channels: &[u32]) -> (Vec<u32>, Vec<i32>) {
        let n = channels.len();
        if n == 0 {
            return (Vec::new(), Vec::new());
        }
        let m = self.n_mask.len();
        let h = &self.h;
        let ch_h = buffer::upload(client, channels);
        let total = n * self.max_comp;
        let proj = buffer::empty::<f32>(client, total);
        let geom = LaunchGeometry::elementwise(client, total);
        // SAFETY: every array is passed with the length it was created with
        unsafe {
            project_classifier_kernel::launch::<f32>(
                client,
                geom.cube_count,
                geom.cube_dim,
                BufferArg::from_raw_parts(snippets.clone(), n * self.width * self.max_neighbours),
                BufferArg::from_raw_parts(ch_h.clone(), n),
                BufferArg::from_raw_parts(h.n_mask.clone(), m),
                BufferArg::from_raw_parts(h.n_comp.clone(), m),
                BufferArg::from_raw_parts(h.mean_off.clone(), m),
                BufferArg::from_raw_parts(h.comp_off.clone(), m),
                BufferArg::from_raw_parts(h.means.clone(), h.lens[0]),
                BufferArg::from_raw_parts(h.components.clone(), h.lens[1]),
                BufferArg::from_raw_parts(proj.clone(), total),
                total as u32,
                self.width as u32,
                self.max_neighbours as u32,
                self.max_comp as u32,
            );
        }
        let second = buffer::empty::<u32>(client, n);
        let geom = LaunchGeometry::elementwise(client, n);
        // SAFETY: as above
        unsafe {
            second_nearest_kernel::launch::<f32>(
                client,
                geom.cube_count,
                geom.cube_dim,
                BufferArg::from_raw_parts(proj, total),
                BufferArg::from_raw_parts(ch_h, n),
                BufferArg::from_raw_parts(h.n_comp.clone(), m),
                BufferArg::from_raw_parts(h.n_train.clone(), m),
                BufferArg::from_raw_parts(h.feat_off.clone(), m),
                BufferArg::from_raw_parts(h.features.clone(), h.lens[2]),
                BufferArg::from_raw_parts(second.clone(), n),
                n as u32,
                self.max_comp as u32,
            );
        }
        let picks = buffer::download::<u32>(client, second);
        channels
            .iter()
            .zip(picks)
            .map(|(&ch, j)| {
                let i = (self.train_off[ch as usize] + j) as usize;
                (self.labels[i], self.offsets[i])
            })
            .unzip()
    }
}

impl DeviceTables {
    fn empty(client: &Client) -> Self {
        let z = || buffer::upload(client, &[0u32]);
        let f = || buffer::upload(client, &[0.0f32]);
        Self {
            n_mask: z(),
            n_comp: z(),
            n_train: z(),
            mean_off: z(),
            comp_off: z(),
            feat_off: z(),
            means: f(),
            components: f(),
            features: f(),
            lens: [1, 1, 1],
        }
    }
}

/// Upstream `remove_duplicate_events`: indices kept of `times` (sorted) with `labels`; per label,
/// spikes within `tol` samples after a kept spike are dropped.
pub fn remove_duplicate_events(times: &[i64], labels: &[u32], tol: i64) -> Vec<usize> {
    let mut last: std::collections::HashMap<u32, i64> = std::collections::HashMap::new();
    let mut keep = Vec::with_capacity(times.len());
    for (i, (&t, &l)) in times.iter().zip(labels).enumerate() {
        match last.get(&l) {
            Some(&kept) if t <= kept + tol => {}
            _ => {
                last.insert(l, t);
                keep.push(i);
            }
        }
    }
    keep
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn upstream_helpers() {
        assert_eq!(subsample_indices(10, 4), vec![0, 2, 5, 7]);
        assert_eq!(subsample_indices(3, 4), vec![0, 1, 2]);
        assert_eq!(noise_times(100, 20, 20, 3), vec![20, 49, 79]);
        // Unit 1: 0 kept, 3 and 5 within 5 of it dropped, 9 kept; unit 2 independent
        let times = [0i64, 3, 4, 5, 9];
        let labels = [1u32, 1, 2, 1, 1];
        assert_eq!(remove_duplicate_events(&times, &labels, 5), vec![0, 2, 4]);
    }
}
