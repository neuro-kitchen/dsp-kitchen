//! MountainSort 5 scheme 1 (`schemes/sorting_scheme1.py`) after detection: the events' masked
//! snippets are clustered and aligned.
//!
//! 1. PCA of the dense snippets to `npca_per_channel · M` components ([`TopComponents`]), then
//!    [`super::isosplit6_subdivision`].
//! 2. Median templates; unless `skip_alignment`: template offsets ([`align_templates`]), each
//!    cluster's snippets rolled by its offset and its times moved by `−offset`, then PCA,
//!    subdivision and templates again on the aligned snippets, and every time moved to its
//!    template's peak ([`offsets_to_peak`]).
//! 3. Times sorted; events whose time left their segment's margins (`[start + T1, end − T2)`)
//!    dropped; units renumbered by peak channel (empty units last).
//!
//! The events may come from several **segments** (scheme 2's training chunks): upstream
//! concatenates the chunks into one recording; here each chunk is its own segment (no event
//! straddles a junction, margins at both ends of every chunk).

use std::ops::Range;

use cubecl::prelude::*;
use dsp_base::linalg::{TopComponents, TopComponentsOptions};

use super::snippets::MaskedSnippets;
use super::subdivision::{isosplit6_subdivision_with_progress, SubdivisionOptions};
use super::templates::{align_templates, median_templates, offsets_to_peak, peak_channels};

/// Clustering settings of scheme 1 (MountainSort 5 defaults in [`Default`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ClusteringParams {
    pub npca_per_channel: usize,
    pub subdivision: SubdivisionOptions,
    pub skip_alignment: bool,
    /// −1 negative peaks, +1 positive, 0 both (the peak the times move to).
    pub detect_sign: i8,
    pub pca: TopComponentsOptions,
}

impl Default for ClusteringParams {
    fn default() -> Self {
        Self {
            npca_per_channel: 3,
            subdivision: SubdivisionOptions::default(),
            skip_alignment: false,
            detect_sign: -1,
            pca: TopComponentsOptions::default(),
        }
    }
}

/// Events of scheme 1: global sample and segment of each, with their snippets.
#[derive(Debug, Clone)]
pub struct Events {
    pub samples: Vec<u64>,
    pub segments: Vec<u32>,
    pub snippets: MaskedSnippets,
}

/// Result of scheme 1.
#[derive(Debug, Clone, PartialEq)]
pub struct Scheme1Output {
    /// Global samples, sorted.
    pub samples: Vec<u64>,
    /// The event (index into the input [`Events`]) of each spike.
    pub events: Vec<usize>,
    /// `1..=k` (ordered by peak channel).
    pub labels: Vec<u32>,
    pub k: usize,
    /// `[k, T, M]`, the median snippets of the final clustering (in the new label order).
    pub templates: Vec<f32>,
    pub peak_channels: Vec<usize>,
}

/// Clusters `events` (module docs). `segments`: the global sample range of each segment.
pub fn cluster_events(client: &Client, events: Events, segments: &[Range<u64>], params: &ClusteringParams) -> Scheme1Output {
    cluster_events_with_progress(client, events, segments, params, &mut |_, _| {})
}

/// [`cluster_events`], reporting `(done, total)` spikes clustered (each spike counts once per
/// clustering: twice with the alignment).
pub fn cluster_events_with_progress(
    client: &Client,
    mut events: Events,
    segments: &[Range<u64>],
    params: &ClusteringParams,
    progress: &mut dyn FnMut(u64, u64),
) -> Scheme1Output {
    let (t, m) = (events.snippets.width, events.snippets.channels);
    let n_before = events.snippets.n_before;
    let n_after = t - n_before;
    let l = events.snippets.len() as u64;
    let total = if params.skip_alignment { l } else { 2 * l };
    progress(0, total);
    let mut finished = 0u64;
    let mut report = |n: usize| {
        finished += n as u64;
        progress(finished, total);
    };
    let mut labels = cluster(client, &events.snippets, params, &mut report);
    let mut k = labels.iter().copied().max().unwrap_or(0) as usize;
    let mut templates = median_templates(&events.snippets, &labels, k);
    let mut times: Vec<i64> = events.samples.iter().map(|&s| s as i64).collect();
    if !params.skip_alignment && k > 0 {
        let offsets = align_templates(client, &templates, k, t, m);
        for i in 0..labels.len() {
            let off = offsets[labels[i] as usize - 1];
            events.snippets.roll(i, off);
            times[i] -= off as i64;
        }
        labels = cluster(client, &events.snippets, params, &mut report);
        k = labels.iter().copied().max().unwrap_or(0) as usize;
        templates = median_templates(&events.snippets, &labels, k);
        let to_peak = offsets_to_peak(&templates, k, t, m, params.detect_sign, n_before);
        for i in 0..labels.len() {
            times[i] += to_peak[labels[i] as usize - 1] as i64;
        }
    }
    let peaks = peak_channels(&templates, k, t, m);
    // Sort by time, drop events outside their segment's margins
    let mut order: Vec<usize> = (0..times.len()).collect();
    order.sort_by_key(|&i| (times[i], i));
    let inside = |i: usize| {
        let seg = &segments[events.segments[i] as usize];
        times[i] >= (seg.start + n_before as u64) as i64 && times[i] < seg.end as i64 - n_after as i64
    };
    let kept: Vec<usize> = order.into_iter().filter(|&i| inside(i)).collect();
    // Units by peak channel, empty units last (stable on ties)
    let mut used = vec![false; k];
    kept.iter().for_each(|&i| used[labels[i] as usize - 1] = true);
    let mut by_channel: Vec<usize> = (0..k).collect();
    by_channel.sort_by_key(|&c| (!used[c], peaks[c]));
    let mut new_label = vec![0u32; k];
    for (rank, &c) in by_channel.iter().enumerate() {
        new_label[c] = rank as u32 + 1;
    }
    let mut new_templates = vec![0.0f32; k * t * m];
    let mut new_peaks = vec![0usize; k];
    for c in 0..k {
        let r = new_label[c] as usize - 1;
        new_templates[r * t * m..(r + 1) * t * m].copy_from_slice(&templates[c * t * m..(c + 1) * t * m]);
        new_peaks[r] = peaks[c];
    }
    Scheme1Output {
        samples: kept.iter().map(|&i| times[i] as u64).collect(),
        events: kept.clone(),
        labels: kept.iter().map(|&i| new_label[labels[i] as usize - 1]).collect(),
        k,
        templates: new_templates,
        peak_channels: new_peaks,
    }
}

/// PCA of the dense snippets, then the subdivision method: labels `1..=K`.
fn cluster(client: &Client, snippets: &MaskedSnippets, params: &ClusteringParams, done: &mut dyn FnMut(usize)) -> Vec<u32> {
    let l = snippets.len();
    if l == 0 {
        return Vec::new();
    }
    let rows = snippets.rows(client, None);
    let pca = TopComponents::fit(client, &rows, params.npca_per_channel * snippets.channels, &params.pca);
    let features = pca.transform(client, &rows, params.pca.batch_elements);
    isosplit6_subdivision_with_progress(client, &features, l, pca.k, &params.subdivision, done)
}
