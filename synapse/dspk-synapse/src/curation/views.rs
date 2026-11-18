//! The Curation views' charts, computed from a sorting (and, for waveforms and features, from
//! windows of its recording) for the units selected: what the app's Curation draws. The same
//! functions run in the app (desktop, a browser page) and on a server for a page it serves, so a
//! view looks the same wherever it was made. Two passes: [`from_spikes`] needs only the spike
//! trains (fast: shown at once), [`from_signal`] reads snippets (waveforms, features).

use serde::{Deserialize, Serialize};

use super::charts::{Chart, ChartGrid, Ink, Layer};
use super::sorting::{autocorrelogram, compute_isi_histogram, crosscorrelogram, SortingData};
use super::waveforms::{Snippets, Windows};

/// A view of Curation (how the app shows it, title and icon, is the app's).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ViewKind {
    Table,
    Waveforms,
    Features,
    Acg,
    Isi,
    Amplitude,
    Template,
    Similarity,
    Correlograms,
    Templates,
    Probe,
    /// Phy's feature view: PCs on the lead unit's best channels, pairwise.
    FeatureGrid,
    /// Phy's template-feature view: spikes projected on two units' templates.
    TemplateFeatures,
    /// Spikes per second over the recording.
    FiringRate,
    /// The lead unit's spike amplitudes as a histogram (cut off by the threshold?).
    AmplitudeHist,
    /// Every unit by amplitude and depth (Phy's cluster scatter).
    UnitScatter,
}

impl ViewKind {
    pub const ALL: [ViewKind; 16] = [
        Self::Table,
        Self::Waveforms,
        Self::Features,
        Self::FeatureGrid,
        Self::TemplateFeatures,
        Self::Acg,
        Self::Isi,
        Self::Amplitude,
        Self::AmplitudeHist,
        Self::FiringRate,
        Self::Template,
        Self::Similarity,
        Self::Correlograms,
        Self::Templates,
        Self::UnitScatter,
        Self::Probe,
    ];
}

/// The views [`from_signal`] makes (they read the recording).
pub const SIGNAL_VIEWS: [ViewKind; 4] = [ViewKind::Waveforms, ViewKind::Features, ViewKind::FeatureGrid, ViewKind::TemplateFeatures];

/// Waveforms and features: spikes per unit.
pub const SPIKES_PER_UNIT: usize = 300;
/// Correlograms: units shown at most (a grid of their pairs).
pub const CCG_MAX_UNITS: usize = 4;
/// Autocorrelogram and correlograms: ± window and bin (ms).
pub const ACG_HALF_MS: f64 = 50.0;
pub const ACG_BIN_MS: f64 = 1.0;
/// ISI: range and bin (ms).
pub const ISI_MAX_MS: f64 = 50.0;
pub const ISI_BIN_MS: f64 = 0.5;
/// The refractory period marked on the ISI (ms).
pub const REFRACTORY_MS: f64 = 1.5;
/// Amplitude over time: spikes drawn per unit at most.
pub const AMPLITUDE_POINTS: usize = 5000;
/// Firing rate: bin length (s).
pub const RATE_BIN_S: f64 = 10.0;
/// Amplitude histogram: bins.
pub const AMPLITUDE_BINS: usize = 60;
/// Feature grid: the lead unit's best channels shown (its extent's first, largest first).
pub const FEATURE_GRID_CHANNELS: usize = 3;

/// What the plots show (each view's ⋯ menu sets them).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PlotSettings {
    /// Correlograms: ± window and bin (ms).
    pub ccg_half_ms: f64,
    pub ccg_bin_ms: f64,
    /// ISI: range and bin (ms).
    pub isi_max_ms: f64,
    pub isi_bin_ms: f64,
    /// Waveforms and features: spikes per unit.
    pub spikes: usize,
}

impl Default for PlotSettings {
    fn default() -> Self {
        Self { ccg_half_ms: ACG_HALF_MS, ccg_bin_ms: ACG_BIN_MS, isi_max_ms: ISI_MAX_MS, isi_bin_ms: ISI_BIN_MS, spikes: SPIKES_PER_UNIT }
    }
}

/// What to compute: for `units` (unit indices; the first leads), the views `wanted`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ViewsRequest {
    pub units: Vec<usize>,
    pub wanted: Vec<ViewKind>,
    pub settings: PlotSettings,
    /// The recording's length (s).
    pub duration_s: f32,
    /// The lead unit's channel in the signal read (the view's numbering: a view of a channel
    /// subset numbers its channels from 0).
    pub channel: usize,
    /// The feature grid's channels, in the signal's numbering (the lead unit's best).
    pub grid_channels: Vec<usize>,
}

/// The charts made, by view, and the unit scatter's points (amplitude, depth, unit) for picking.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Views {
    pub charts: Vec<(ViewKind, ChartGrid)>,
    pub scatter: Vec<(f32, f32, usize)>,
    /// The basis the features view was drawn with: a lasso on it selects by the same projection.
    #[serde(default)]
    pub features: Option<FeatureBasis>,
}

/// How the features view projects a spike: its channel (the signal's numbering), the snippet's
/// samples around the spike, and the PCA basis.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FeatureBasis {
    pub channel: usize,
    pub before: u64,
    pub after: u64,
    pub basis: super::waveforms::PcaBasis,
}

/// The ink of a snippet row's unit (`None`: unsorted).
fn ink_of(u: Option<u32>) -> Ink {
    u.map_or(Ink::Unsorted, |u| Ink::Unit(u as usize))
}

/// The views of `request` that need only the spike trains: autocorrelogram, ISI, template,
/// correlograms, amplitude over time, probe, firing rate, amplitude histogram, unit scatter.
/// `stale` is asked between views: a newer request makes this one stop.
pub fn from_spikes(sorting: &SortingData, request: &ViewsRequest, stale: impl Fn() -> bool) -> Views {
    let mut out = Views::default();
    let Some(&lead) = request.units.first() else { return out };
    if stale() || lead >= sorting.units.len() {
        return out;
    }
    let (units, wanted, set, duration) = (&request.units, &request.wanted, request.settings, request.duration_s);
    let lead_ink = Ink::Unit(lead);
    let ids: Vec<usize> = units.iter().map(|&u| sorting.units[u].info.unit_id).collect();
    let times = sorting.unit_times(lead);
    let acg = autocorrelogram(&times, sorting.sample_rate, set.ccg_half_ms, set.ccg_bin_ms);
    let isi = compute_isi_histogram(&times, sorting.sample_rate, set.isi_max_ms, set.isi_bin_ms);
    let (half, bin) = (set.ccg_half_ms as f32, set.ccg_bin_ms as f32);
    let bars = |counts: &[u32], ink, start: f32, width: f32| Layer::Bars { ink, start, width, counts: counts.iter().map(|&c| c as f32).collect() };
    let mut put = |kind, grid| out.charts.push((kind, grid));
    put(
        ViewKind::Acg,
        ChartGrid::one(Chart {
            layers: vec![bars(&acg, lead_ink, -half - bin / 2.0, bin), Layer::Marker { x: 0.0 }],
            x_label: "lag (ms)".into(),
            y_label: "spike pairs".into(),
            x: (-half - bin / 2.0, half + bin / 2.0),
            y: (0.0, 1.0),
            y_fit: true,
            title: String::new(),
        }),
    );
    put(
        ViewKind::Isi,
        ChartGrid::one(Chart {
            layers: vec![bars(&isi, lead_ink, 0.0, set.isi_bin_ms as f32), Layer::Marker { x: REFRACTORY_MS as f32 }],
            x_label: "interval (ms)".into(),
            y_label: "intervals".into(),
            x: (0.0, set.isi_max_ms as f32),
            y: (0.0, 1.0),
            y_fit: true,
            title: String::new(),
        }),
    );
    // The lead unit's template, one cell per channel of its extent
    let template = &sorting.units[lead].template;
    let (before, rate) = (sorting.units[lead].extent.before as f32, sorting.sample_rate as f32);
    let amp = template.iter().flatten().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-6) * 1.1;
    let cells: Vec<Chart> = template
        .iter()
        .zip(&sorting.units[lead].extent.channels)
        .map(|(row, ch)| Chart {
            layers: vec![Layer::Lines { ink: lead_ink, alpha: 1.0, width: 1.6, lines: vec![row.iter().enumerate().map(|(i, &v)| ((i as f32 - before) / rate * 1e3, v)).collect()] }],
            x_label: "time (ms)".into(),
            y_label: "template".into(),
            x: (-before / rate * 1e3, (row.len() as f32 - before) / rate * 1e3),
            y: (-amp, amp),
            y_fit: false,
            title: format!("ch {ch}"),
        })
        .collect();
    if !cells.is_empty() {
        let cols = (cells.len() as f32).sqrt().ceil() as usize;
        put(ViewKind::Template, ChartGrid { cells, cols });
    }
    if wanted.contains(&ViewKind::Correlograms) {
        let shown: Vec<usize> = units.iter().copied().take(CCG_MAX_UNITS).collect();
        let trains: Vec<Vec<u64>> = shown.iter().map(|&u| sorting.unit_times(u)).collect();
        let mut cells = Vec::new();
        for (i, a) in trains.iter().enumerate() {
            for (j, b) in trains.iter().enumerate() {
                // A unit with itself is its autocorrelogram: a cross-correlogram of a train with
                // itself counts every spike against itself at zero lag (a bar the size of the unit)
                let counts = if i == j { autocorrelogram(a, sorting.sample_rate, set.ccg_half_ms, set.ccg_bin_ms) } else { crosscorrelogram(a, b, sorting.sample_rate, set.ccg_half_ms, set.ccg_bin_ms) };
                let ink = if i == j { Ink::Unit(shown[i]) } else { Ink::Unsorted };
                cells.push(Chart {
                    layers: vec![bars(&counts, ink, -half - bin / 2.0, bin), Layer::Marker { x: 0.0 }],
                    x_label: "lag (ms)".into(),
                    y_label: String::new(),
                    x: (-half - bin / 2.0, half + bin / 2.0),
                    y: (0.0, 1.0),
                    y_fit: true,
                    title: format!("{} × {}", ids[i], ids[j]),
                });
            }
        }
        put(ViewKind::Correlograms, ChartGrid { cols: shown.len().max(1), cells });
    }
    if stale() {
        return out;
    }
    let mut layers = Vec::new();
    let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
    for &u in units {
        let pts = sorting.unit_amplitudes(u, AMPLITUDE_POINTS);
        for p in &pts {
            lo = lo.min(p.1);
            hi = hi.max(p.1);
        }
        layers.push(Layer::Points { ink: Ink::Unit(u), alpha: 0.5, radius: 1.2, points: pts });
    }
    if lo.is_finite() {
        let pad = (hi - lo).max(1e-6) * 0.05;
        put(ViewKind::Amplitude, ChartGrid::one(Chart { layers, x_label: "time (s)".into(), y_label: "amplitude".into(), x: (0.0, duration), y: (lo.min(0.0) - pad, hi + pad), y_fit: false, title: String::new() }));
    }
    if wanted.contains(&ViewKind::Probe) {
        let channels: Vec<(f32, f32)> = sorting.positions.iter().flatten().copied().collect();
        if !channels.is_empty() {
            let (x0, x1) = channels.iter().fold((f32::INFINITY, f32::NEG_INFINITY), |(a, b), p| (a.min(p.0), b.max(p.0)));
            let (y0, y1) = channels.iter().fold((f32::INFINITY, f32::NEG_INFINITY), |(a, b), p| (a.min(p.1), b.max(p.1)));
            let mut layers = vec![Layer::Points { ink: Ink::Grid, alpha: 1.0, radius: 1.5, points: channels }];
            for u in sorting.alive() {
                if let Some(p) = sorting.unit_position(u) {
                    let on = units.contains(&u);
                    layers.push(Layer::Points { ink: Ink::Unit(u), alpha: if on { 1.0 } else { 0.55 }, radius: if on { 6.0 } else { 3.0 }, points: vec![p] });
                }
            }
            let pad = ((x1 - x0).max(y1 - y0) * 0.04).max(10.0);
            put(ViewKind::Probe, ChartGrid::one(Chart { layers, x_label: "x (µm)".into(), y_label: "depth (µm)".into(), x: (x0 - pad, x1 + pad), y: (y0 - pad, y1 + pad), y_fit: false, title: String::new() }));
        }
    }
    // Each selected unit's rate over the recording
    if wanted.contains(&ViewKind::FiringRate) {
        let mut hi = 0.0f32;
        let layers: Vec<Layer> = units
            .iter()
            .map(|&u| {
                let rate = sorting.firing_rate(u, duration as f64, RATE_BIN_S);
                hi = rate.iter().fold(hi, |m, &v| m.max(v));
                let line = rate.iter().enumerate().map(|(i, &v)| (((i as f64 + 0.5) * RATE_BIN_S) as f32, v)).collect();
                Layer::Lines { ink: Ink::Unit(u), alpha: 1.0, width: 1.6, lines: vec![line] }
            })
            .collect();
        put(ViewKind::FiringRate, ChartGrid::one(Chart { layers, x_label: "time (s)".into(), y_label: "spikes / s".into(), x: (0.0, duration), y: (0.0, hi.max(1.0) * 1.1), y_fit: false, title: String::new() }));
    }
    // The lead unit's amplitudes: a cut-off left edge means spikes lost under the threshold
    if wanted.contains(&ViewKind::AmplitudeHist) {
        if let Some((lo, width, counts)) = sorting.amplitude_histogram(lead, AMPLITUDE_BINS) {
            let hi = lo + width * counts.len() as f32;
            put(ViewKind::AmplitudeHist, ChartGrid::one(Chart { layers: vec![bars(&counts, lead_ink, lo, width)], x_label: "amplitude".into(), y_label: "spikes".into(), x: (lo, hi), y: (0.0, 1.0), y_fit: true, title: String::new() }));
        }
    }
    // Every unit by amplitude and depth, sized by its spikes; the selected ones larger
    if wanted.contains(&ViewKind::UnitScatter) {
        let points = sorting.unit_points(duration as f64);
        if !points.is_empty() {
            let layers = points
                .iter()
                .map(|p| {
                    let on = units.contains(&p.unit);
                    let radius = (1.5 + (p.spikes as f32).sqrt() / 12.0).min(7.0) + if on { 2.0 } else { 0.0 };
                    Layer::Points { ink: Ink::Unit(p.unit), alpha: if on { 1.0 } else { 0.6 }, radius, points: vec![(p.amplitude, p.depth)] }
                })
                .collect();
            let fold = |f: fn(&super::UnitPoint) -> f32| points.iter().map(f).fold((f32::INFINITY, f32::NEG_INFINITY), |(a, b), v| (a.min(v), b.max(v)));
            let ((a0, a1), (d0, d1)) = (fold(|p| p.amplitude), fold(|p| p.depth));
            let (pa, pd) = (((a1 - a0) * 0.06).max(1e-3), ((d1 - d0) * 0.06).max(10.0));
            put(ViewKind::UnitScatter, ChartGrid::one(Chart { layers, x_label: "amplitude".into(), y_label: "depth (µm)".into(), x: (a0 - pa, a1 + pa), y: (d0 - pd, d1 + pd), y_fit: false, title: String::new() }));
            out.scatter = points.iter().map(|p| (p.amplitude, p.depth, p.unit)).collect();
        }
    }
    out
}

/// The views of `request` read from the recording ([`SIGNAL_VIEWS`]): waveforms, features (PC1 ·
/// PC2), the feature grid and the template features, from snippets cut out of `windows`.
pub fn from_signal(sorting: &SortingData, windows: &Windows, request: &ViewsRequest, stale: impl Fn() -> bool) -> Views {
    let mut out = Views::default();
    let (units, wanted, set) = (&request.units, &request.wanted, request.settings);
    if stale() || units.is_empty() || !SIGNAL_VIEWS.iter().any(|k| wanted.contains(k)) {
        return out;
    }
    let snippets = Snippets::read(sorting, windows, units, request.channel, set.spikes);
    let (before, rate) = (snippets.before as f32, sorting.sample_rate as f32);
    let ms = |i: usize| (i as f32 - before) / rate * 1e3;
    let mut groups: Vec<Option<u32>> = snippets.rows.iter().map(|(u, _, _)| *u).collect();
    groups.sort();
    groups.dedup();
    let rows_of = |g: &Option<u32>| snippets.rows.iter().filter(|(u, _, _)| u == g).map(|(_, _, r)| r).collect::<Vec<&Vec<f32>>>();
    let mut layers = Vec::new();
    let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
    for g in &groups {
        let rows = rows_of(g);
        for &v in rows.iter().flat_map(|r| r.iter()) {
            if v.is_finite() {
                lo = lo.min(v);
                hi = hi.max(v);
            }
        }
        layers.push(Layer::Lines { ink: ink_of(*g), alpha: 0.12, width: 1.0, lines: rows.iter().map(|r| r.iter().enumerate().map(|(i, &v)| (ms(i), v)).collect()).collect() });
    }
    for g in groups.iter().filter(|g| g.is_some()) {
        let rows = rows_of(g);
        if let Some(n) = rows.first().map(|r| r.len()) {
            let mean: Vec<(f32, f32)> = (0..n).map(|i| (ms(i), rows.iter().map(|r| r[i]).sum::<f32>() / rows.len() as f32)).collect();
            layers.push(Layer::Lines { ink: ink_of(*g), alpha: 1.0, width: 2.2, lines: vec![mean] });
        }
    }
    if lo.is_finite() {
        let n = (snippets.before + snippets.after) as usize;
        let pad = (hi - lo).max(1e-6) * 0.05;
        out.charts.push((
            ViewKind::Waveforms,
            ChartGrid::one(Chart { layers, x_label: "time (ms)".into(), y_label: "µV".into(), x: (ms(0), ms(n.saturating_sub(1))), y: (lo - pad, hi + pad), y_fit: false, title: String::new() }),
        ));
    }
    // Points (one per snippet row) grouped by unit, as a scatter chart
    let by_group = |points: &[(f32, f32)]| -> Vec<(Ink, Vec<(f32, f32)>)> {
        groups.iter().map(|g| (ink_of(*g), snippets.rows.iter().zip(points).filter(|((u, _, _), _)| u == g).map(|(_, p)| *p).collect())).collect()
    };
    // The host basis, kept: what is drawn is what a lasso on it selects
    if let Some(basis) = snippets.pca_basis() {
        let pcs: Vec<(f32, f32)> = snippets.rows.iter().map(|(_, _, r)| basis.project(r)).map(|p| (p[0], p[1])).collect();
        if let Some(c) = Chart::scatter(by_group(&pcs), 1.6, "PC1".into(), "PC2".into(), String::new()) {
            out.charts.push((ViewKind::Features, ChartGrid::one(c)));
        }
        out.features = Some(FeatureBasis { channel: request.channel, before: snippets.before, after: snippets.after, basis });
    }
    // Phy's feature view: on each best channel PC1 × PC2, between two channels their PC1s (the
    // same spikes are cut on every channel, in the same order)
    if wanted.contains(&ViewKind::FeatureGrid) && !stale() {
        let channels = &request.grid_channels;
        let reads: Vec<Snippets> = channels.iter().map(|&c| Snippets::read(sorting, windows, units, c, set.spikes)).collect();
        let n = snippets.rows.len();
        if n >= 3 && reads.iter().all(|r| r.rows.len() == n) {
            let pcs: Vec<Vec<[f32; 2]>> = reads.iter().map(Snippets::pca2).collect();
            let k = reads.len();
            let mut cells = Vec::new();
            for i in 0..k {
                for j in 0..k {
                    let points: Vec<(f32, f32)> = (0..n).map(|r| if i == j { (pcs[i][r][0], pcs[i][r][1]) } else { (pcs[j][r][0], pcs[i][r][0]) }).collect();
                    let (ci, cj) = (channels[i], channels[j]);
                    let (xl, yl, title) = if i == j { ("PC1".to_string(), "PC2".to_string(), format!("ch {ci}")) } else { (format!("PC1 ch {cj}"), format!("PC1 ch {ci}"), format!("ch {cj} × ch {ci}")) };
                    cells.extend(Chart::scatter(by_group(&points), 1.4, xl, yl, title));
                }
            }
            if cells.len() == k * k {
                out.charts.push((ViewKind::FeatureGrid, ChartGrid { cells, cols: k }));
            }
        }
    }
    // Phy's template-feature view: the spikes of the first two units on both templates
    if wanted.contains(&ViewKind::TemplateFeatures) && units.len() >= 2 {
        let channel = sorting.units[units[0]].info.primary_channel;
        if let (Some(a), Some(b)) = (sorting.template_on(units[0], channel), sorting.template_on(units[1], channel)) {
            let points: Vec<(f32, f32)> = snippets.template_features(a, b).into_iter().map(|p| (p[0], p[1])).collect();
            let (ia, ib) = (sorting.units[units[0]].info.unit_id, sorting.units[units[1]].info.unit_id);
            if let Some(c) = Chart::scatter(by_group(&points), 1.4, format!("on unit {ia}'s template"), format!("on unit {ib}'s template"), String::new()) {
                out.charts.push((ViewKind::TemplateFeatures, ChartGrid::one(c)));
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::super::sorting::testing::sorting_of_units;
    use super::*;

    /// From the spike trains: the views asked for, coloured by unit, and nothing for no units.
    #[test]
    fn the_spike_views_are_made_for_the_units_asked() {
        let mut s = sorting_of_units(&[(5, 0, 6), (6, 1, 4)]);
        s.amplitudes = vec![1.0; 10];
        s.positions = vec![Some((0.0, 20.0)), Some((16.0, 40.0))];
        let request = ViewsRequest {
            units: vec![0, 1],
            wanted: vec![ViewKind::Correlograms, ViewKind::FiringRate, ViewKind::UnitScatter, ViewKind::Probe],
            settings: PlotSettings::default(),
            duration_s: 0.1,
            channel: 0,
            grid_channels: vec![0],
        };
        let views = from_spikes(&s, &request, || false);
        let kinds: Vec<ViewKind> = views.charts.iter().map(|(k, _)| *k).collect();
        for k in [ViewKind::Acg, ViewKind::Isi, ViewKind::Template, ViewKind::Correlograms, ViewKind::Amplitude, ViewKind::FiringRate, ViewKind::UnitScatter, ViewKind::Probe] {
            assert!(kinds.contains(&k), "{k:?} in {kinds:?}");
        }
        let ccg = &views.charts.iter().find(|(k, _)| *k == ViewKind::Correlograms).unwrap().1;
        assert_eq!((ccg.cols, ccg.cells.len()), (2, 4), "two units: a 2 × 2 grid");
        assert!(matches!(ccg.cells[0].layers[0], Layer::Bars { ink: Ink::Unit(0), .. }));
        // A unit with itself is its autocorrelogram: no count at zero lag (a spike is not its own
        // neighbour), on both diagonal cells
        for cell in [&ccg.cells[0], &ccg.cells[3]] {
            let Layer::Bars { counts, .. } = &cell.layers[0] else { panic!("bars") };
            assert_eq!(counts[counts.len() / 2], 0.0, "{}: zero lag", cell.title);
        }
        assert_eq!(views.scatter.len(), 2);
        // Asked for no views of the optional kinds: only those always made
        let few = from_spikes(&s, &ViewsRequest { wanted: vec![], ..request.clone() }, || false);
        assert!(!few.charts.iter().any(|(k, _)| *k == ViewKind::Correlograms));
        // No units: nothing; stale at once: nothing
        assert!(from_spikes(&s, &ViewsRequest { units: vec![], ..request.clone() }, || false).charts.is_empty());
        assert!(from_spikes(&s, &request, || true).charts.is_empty());
    }
}
