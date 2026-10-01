//! Rasterizers for the phy-style sorting widgets: waveforms, feature scatter, correlograms,
//! ISI histograms and amplitudes over time.
//!
//! Every plot has one data→pixel mapping, used both to draw and to build the axis labels
//! (`axes`), so the Slint labels always line up with the raster.

use std::sync::Arc;

use slint::{Rgba8Pixel, SharedPixelBuffer};

use crate::shared::axis::nice_step;
use crate::shared::canvas::{blend_color, Canvas};
use crate::shared::render_worker::Frame;

use super::sorting::{Sorting, NUM_PCS};

const BG: Rgba8Pixel = Rgba8Pixel { r: 9, g: 13, b: 19, a: 255 };
const GRID: Rgba8Pixel = Rgba8Pixel { r: 22, g: 27, b: 34, a: 255 };
const GREY: Rgba8Pixel = Rgba8Pixel { r: 110, g: 118, b: 129, a: 255 };
const REFRACTORY: Rgba8Pixel = Rgba8Pixel { r: 248, g: 81, b: 73, a: 255 };
const ACCENT: Rgba8Pixel = Rgba8Pixel { r: 56, g: 189, b: 248, a: 255 };

/// Colors of selected clusters, in selection order (phy convention: first selected is blue).
pub const SELECTION_COLORS: [Rgba8Pixel; 8] = [
    Rgba8Pixel { r: 74, g: 158, b: 255, a: 255 },  // blue
    Rgba8Pixel { r: 255, g: 90, b: 90, a: 255 },   // red
    Rgba8Pixel { r: 255, g: 210, b: 63, a: 255 },  // yellow
    Rgba8Pixel { r: 61, g: 220, b: 132, a: 255 },  // green
    Rgba8Pixel { r: 34, g: 211, b: 238, a: 255 },  // cyan
    Rgba8Pixel { r: 232, g: 121, b: 249, a: 255 }, // magenta
    Rgba8Pixel { r: 251, g: 146, b: 60, a: 255 },  // orange
    Rgba8Pixel { r: 163, g: 230, b: 53, a: 255 },  // lime
];

pub fn selection_color(index: usize) -> Rgba8Pixel {
    SELECTION_COLORS[index % SELECTION_COLORS.len()]
}

/// Axes shown in the feature view.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum FeatureAxes {
    #[default]
    Pc1Pc2,
    Pc1Pc3,
    Pc2Pc3,
    DepthPc1,
}

impl FeatureAxes {
    pub const ALL: [FeatureAxes; 4] = [Self::Pc1Pc2, Self::Pc1Pc3, Self::Pc2Pc3, Self::DepthPc1];

    pub fn label(self) -> &'static str {
        match self {
            Self::Pc1Pc2 => "PC1 vs PC2",
            Self::Pc1Pc3 => "PC1 vs PC3",
            Self::Pc2Pc3 => "PC2 vs PC3",
            Self::DepthPc1 => "PC1 vs depth",
        }
    }

    /// (x, y) value of spike `i`.
    fn values(self, s: &Sorting, i: usize) -> (f32, f32) {
        let pc = |k: usize| s.features[i * NUM_PCS + k];
        match self {
            Self::Pc1Pc2 => (pc(0), pc(1)),
            Self::Pc1Pc3 => (pc(0), pc(2)),
            Self::Pc2Pc3 => (pc(1), pc(2)),
            Self::DepthPc1 => (pc(0), s.depth_um[i]),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SpikePlot {
    Waveforms,
    Features(FeatureAxes),
    Correlograms,
    Isi,
    Amplitudes,
}

#[derive(Clone)]
pub struct SpikePlotRequest {
    pub plot: SpikePlot,
    pub width: u32,
    pub height: u32,
    pub scale: f32,
    pub sorting: Arc<Sorting>,
    /// Selected cluster ids, in selection order (their colors follow `SELECTION_COLORS`).
    pub selected: Vec<u32>,
    /// Shared timeline window (seconds), shaded in the amplitude view.
    pub window: (f64, f64),
}

/// Axis labels: `lanes` along the left gutter (y_frac from the top), `ticks` under the plot
/// (x frac), and a short `note` shown in the plot corner.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Axes {
    pub lanes: Vec<(String, f32)>,
    pub ticks: Vec<(String, f32)>,
    pub note: String,
}

/// Waveform spikes drawn per cluster.
const WAVEFORM_SPIKES: usize = 80;
const CCG_BIN_MS: f32 = 1.0;
const CCG_WINDOW_MS: f32 = 50.0;
const MAX_CCG_CLUSTERS: usize = 4;
const ISI_BIN_MS: f32 = 0.5;
const ISI_MAX_MS: f32 = 50.0;
/// Fraction of the amplitude view used by the side histogram.
const AMP_HIST_FRAC: f32 = 0.15;

/// Nice 1-2-5 ticks covering `[lo, hi]`: (value, label).
fn nice_ticks(lo: f64, hi: f64, target: usize) -> Vec<(f64, String)> {
    if !(hi > lo) {
        return Vec::new();
    }
    let step = nice_step((hi - lo) / target.max(1) as f64);
    let decimals = (-step.log10().floor()).max(0.0) as usize;
    let mut out = Vec::new();
    let mut v = (lo / step).ceil() * step;
    while v <= hi + step * 1e-9 {
        out.push((v, format!("{:.*}", decimals, if v.abs() < step * 1e-9 { 0.0 } else { v })));
        v += step;
    }
    out
}

/// Robust [1%, 99%] range, padded by 5%.
fn robust_range(mut values: Vec<f32>) -> (f32, f32) {
    if values.is_empty() {
        return (-1.0, 1.0);
    }
    values.sort_by(f32::total_cmp);
    let q = |p: f32| values[((values.len() - 1) as f32 * p) as usize];
    let (lo, hi) = (q(0.01), q(0.99));
    let pad = ((hi - lo) * 0.05).max(1e-3);
    (lo - pad, hi + pad)
}

/// Channels shown in the waveform view: the first selected cluster's template channels, by id.
fn waveform_channels(req: &SpikePlotRequest) -> Vec<usize> {
    let mut chans = req
        .selected
        .first()
        .and_then(|&id| req.sorting.cluster(id))
        .map(|c| c.template_channels.clone())
        .unwrap_or_default();
    chans.sort_unstable();
    chans
}

fn feature_ranges(req: &SpikePlotRequest, axes: FeatureAxes) -> ((f32, f32), (f32, f32)) {
    let s = &req.sorting;
    let (xs, ys): (Vec<f32>, Vec<f32>) = (0..s.num_spikes()).map(|i| axes.values(s, i)).unzip();
    (robust_range(xs), robust_range(ys))
}

fn amplitude_max(req: &SpikePlotRequest) -> f32 {
    let s = &req.sorting;
    let ids: Vec<u32> = if req.selected.is_empty() { s.clusters.iter().map(|c| c.id).collect() } else { req.selected.clone() };
    ids.iter()
        .filter_map(|&id| s.cluster(id))
        .flat_map(|c| c.spikes.iter().map(|&i| s.amplitudes[i].abs()))
        .fold(1.0f32, f32::max)
        * 1.1
}

fn ms_ticks(lo_ms: f64, hi_ms: f64, target: usize, x_of: impl Fn(f64) -> f32) -> Vec<(String, f32)> {
    nice_ticks(lo_ms, hi_ms, target).into_iter().map(|(v, l)| (format!("{l} ms"), x_of(v))).collect()
}

/// How many labels fit along `px` physical pixels (one per ~90 logical px on x, ~45 on y).
fn tick_budget(px: u32, scale: f32, spacing: f32) -> usize {
    ((px as f32 / scale.max(0.1)) / spacing).clamp(2.0, 8.0) as usize
}

/// Axis labels for `req` (cheap; computed on the UI thread).
pub fn axes(req: &SpikePlotRequest) -> Axes {
    let s = &req.sorting;
    let x_budget = tick_budget(req.width, req.scale, 90.0);
    let y_budget = tick_budget(req.height, req.scale, 45.0);
    match req.plot {
        SpikePlot::Waveforms => {
            let chans = waveform_channels(req);
            let n = chans.len().max(1) as f32;
            let ns = s.batch.num_samples.max(2);
            let to_ms = |i: f64| (i - s.pre_samples as f64) / s.sample_rate * 1000.0;
            let (lo, hi) = (to_ms(0.0), to_ms((ns - 1) as f64));
            Axes {
                lanes: chans.iter().enumerate().map(|(r, c)| (format!("Ch {c}"), (r as f32 + 0.5) / n)).collect(),
                ticks: ms_ticks(lo, hi, x_budget, |v| ((v - lo) / (hi - lo)) as f32),
                note: format!("≤{WAVEFORM_SPIKES} spikes / cluster"),
            }
        }
        SpikePlot::Features(fa) => {
            let ((x0, x1), (y0, y1)) = feature_ranges(req, fa);
            Axes {
                lanes: nice_ticks(y0 as f64, y1 as f64, y_budget)
                    .into_iter()
                    .map(|(v, l)| (l, 1.0 - ((v as f32 - y0) / (y1 - y0))))
                    .collect(),
                ticks: nice_ticks(x0 as f64, x1 as f64, x_budget)
                    .into_iter()
                    .map(|(v, l)| (l, (v as f32 - x0) / (x1 - x0)))
                    .collect(),
                note: fa.label().into(),
            }
        }
        SpikePlot::Correlograms => {
            let sel: Vec<u32> = req.selected.iter().copied().take(MAX_CCG_CLUSTERS).collect();
            let n = sel.len().max(1) as f32;
            Axes {
                lanes: sel.iter().enumerate().map(|(r, id)| (format!("#{id}"), (r as f32 + 0.5) / n)).collect(),
                ticks: sel.iter().enumerate().map(|(c, id)| (format!("#{id}"), (c as f32 + 0.5) / n)).collect(),
                note: format!("±{CCG_WINDOW_MS:.0} ms · {CCG_BIN_MS:.0} ms bins"),
            }
        }
        SpikePlot::Isi => Axes {
            lanes: Vec::new(),
            ticks: ms_ticks(0.0, ISI_MAX_MS as f64, x_budget, |v| (v / ISI_MAX_MS as f64) as f32),
            note: "refractory 1.5 ms".into(),
        },
        SpikePlot::Amplitudes => {
            let amax = amplitude_max(req);
            let plot_frac = 1.0 - AMP_HIST_FRAC;
            let dur = s.duration_sec.max(1e-9);
            Axes {
                lanes: nice_ticks(0.0, amax as f64, y_budget).into_iter().map(|(v, l)| (format!("{l} µV"), 1.0 - v as f32 / amax)).collect(),
                ticks: nice_ticks(0.0, dur, tick_budget((req.width as f32 * plot_frac) as u32, req.scale, 90.0)).into_iter().map(|(v, l)| (format!("{l} s"), (v / dur) as f32 * plot_frac)).collect(),
                note: "|trough| amplitude".into(),
            }
        }
    }
}

pub fn render(req: &SpikePlotRequest) -> Frame {
    let (w, h) = (req.width.max(1), req.height.max(1));
    let mut buffer = SharedPixelBuffer::<Rgba8Pixel>::new(w, h);
    let mut canvas = Canvas { pixels: buffer.make_mut_slice(), width: w as usize, height: h as usize };
    canvas.pixels.fill(BG);
    let ax = axes(req);
    for &(_, f) in &ax.ticks {
        canvas.vline((f * canvas.width as f32) as usize, 0, canvas.height, GRID);
    }
    match req.plot {
        SpikePlot::Waveforms => draw_waveforms(&mut canvas, req),
        SpikePlot::Features(fa) => draw_features(&mut canvas, req, fa),
        SpikePlot::Correlograms => draw_correlograms(&mut canvas, req),
        SpikePlot::Isi => draw_isi(&mut canvas, req),
        SpikePlot::Amplitudes => draw_amplitudes(&mut canvas, req),
    }
    buffer
}

fn draw_waveforms(canvas: &mut Canvas, req: &SpikePlotRequest) {
    let s = &req.sorting;
    let chans = waveform_channels(req);
    let (w, h) = (canvas.width as f32, canvas.height as f32);
    let ns = s.batch.num_samples;
    if chans.is_empty() || ns < 2 {
        return;
    }
    let row_h = h / chans.len() as f32;

    // Shared vertical scale: largest template excursion on the shown channels
    let mut vmax = 1.0f32;
    for &id in &req.selected {
        if let Some(c) = s.cluster(id) {
            vmax = c.template.iter().fold(vmax, |m, v| m.max(v.abs()));
        }
    }
    let k = 0.45 * row_h / vmax;
    let x_of = |i: usize| i as f32 / (ns - 1) as f32 * (w - 1.0);

    for (r, _) in chans.iter().enumerate() {
        canvas.hline(((r as f32 + 0.5) * row_h) as usize, 0, canvas.width, GRID);
    }

    let thin = req.scale.max(1.0);
    for (sel_i, &id) in req.selected.iter().enumerate() {
        let Some(c) = s.cluster(id) else { continue };
        let color = selection_color(sel_i);
        let step = c.spikes.len().div_ceil(WAVEFORM_SPIKES).max(1);
        for &spike in c.spikes.iter().step_by(step) {
            let ids = s.batch.spike_channel_ids(spike);
            for (r, ch) in chans.iter().enumerate() {
                let Some(local) = ids.iter().position(|x| x == ch) else { continue };
                let wave = s.batch.channel_slice(spike, local);
                let center = (r as f32 + 0.5) * row_h;
                let pts: Vec<(f32, f32)> = wave.iter().enumerate().map(|(i, v)| (x_of(i), center - v * k)).collect();
                canvas.polyline(&pts, color, 0.12, thin);
            }
        }
        // Mean template on top
        for (r, ch) in chans.iter().enumerate() {
            let Some(local) = c.template_channels.iter().position(|x| x == ch) else { continue };
            let wave = &c.template[local * ns..(local + 1) * ns];
            let center = (r as f32 + 0.5) * row_h;
            let pts: Vec<(f32, f32)> = wave.iter().enumerate().map(|(i, v)| (x_of(i), center - v * k)).collect();
            canvas.polyline(&pts, color, 1.0, 2.0 * thin);
        }
    }
}

fn draw_features(canvas: &mut Canvas, req: &SpikePlotRequest, fa: FeatureAxes) {
    let s = &req.sorting;
    let ((x0, x1), (y0, y1)) = feature_ranges(req, fa);
    let (w, h) = (canvas.width as f32, canvas.height as f32);
    let map = |(x, y): (f32, f32)| ((x - x0) / (x1 - x0) * w, (1.0 - (y - y0) / (y1 - y0)) * h);

    // Background: every spike in grey (subsampled for very large sortings)
    let step = s.num_spikes().div_ceil(20_000).max(1);
    for i in (0..s.num_spikes()).step_by(step) {
        let (px, py) = map(fa.values(s, i));
        canvas.dot(px, py, 1.0 * req.scale, GREY, 0.35);
    }
    for (sel_i, &id) in req.selected.iter().enumerate() {
        let Some(c) = s.cluster(id) else { continue };
        for &i in &c.spikes {
            let (px, py) = map(fa.values(s, i));
            canvas.dot(px, py, 1.6 * req.scale, selection_color(sel_i), 0.85);
        }
    }
}

fn draw_correlograms(canvas: &mut Canvas, req: &SpikePlotRequest) {
    let s = &req.sorting;
    let sel: Vec<u32> = req.selected.iter().copied().take(MAX_CCG_CLUSTERS).collect();
    let n = sel.len();
    if n == 0 {
        return;
    }
    let (cw, ch) = (canvas.width as f32 / n as f32, canvas.height as f32 / n as f32);
    let pad = 4.0 * req.scale;
    for (r, &a) in sel.iter().enumerate() {
        for (c, &b) in sel.iter().enumerate() {
            let ccg = s.correlogram(a, b, CCG_BIN_MS, CCG_WINDOW_MS);
            let color = if r == c {
                selection_color(r)
            } else {
                blend_color(selection_color(r), selection_color(c), 0.5)
            };
            let (x0, y0) = (c as f32 * cw + pad, r as f32 * ch + pad);
            let (iw, ih) = (cw - 2.0 * pad, ch - 2.0 * pad);
            canvas.rect(x0, y0, x0 + iw, y0 + ih, GRID, 0.6);
            let max = ccg.counts.iter().copied().max().unwrap_or(0).max(1) as f32;
            let bins = ccg.counts.len().max(1) as f32;
            for (bi, &count) in ccg.counts.iter().enumerate() {
                let bx0 = x0 + bi as f32 / bins * iw;
                let bx1 = x0 + (bi + 1) as f32 / bins * iw;
                let top = y0 + ih * (1.0 - count as f32 / max);
                canvas.rect(bx0, top, bx1, y0 + ih, color, if r == c { 0.9 } else { 0.7 });
            }
            // Zero lag
            canvas.vline_alpha((x0 + iw / 2.0) as usize, y0 as usize, (y0 + ih) as usize, GREY, 0.5);
        }
    }
}

fn draw_isi(canvas: &mut Canvas, req: &SpikePlotRequest) {
    let s = &req.sorting;
    let (w, h) = (canvas.width as f32, canvas.height as f32);
    let x_of = |ms: f32| ms / ISI_MAX_MS * w;
    canvas.rect(0.0, 0.0, x_of(1.5), h, REFRACTORY, 0.15);
    for (sel_i, &id) in req.selected.iter().enumerate() {
        let hist = s.isi_histogram(id, ISI_BIN_MS, ISI_MAX_MS);
        let max = hist.iter().copied().max().unwrap_or(0).max(1) as f32;
        let color = selection_color(sel_i);
        let mut outline = Vec::with_capacity(hist.len() * 2);
        for (b, &count) in hist.iter().enumerate() {
            let (bx0, bx1) = (x_of(b as f32 * ISI_BIN_MS), x_of((b + 1) as f32 * ISI_BIN_MS));
            let top = h * (1.0 - 0.95 * count as f32 / max);
            canvas.rect(bx0, top, bx1, h, color, 0.25);
            outline.push((bx0, top));
            outline.push((bx1, top));
        }
        canvas.polyline(&outline, color, 1.0, req.scale.max(1.0));
    }
}

fn draw_amplitudes(canvas: &mut Canvas, req: &SpikePlotRequest) {
    let s = &req.sorting;
    let (w, h) = (canvas.width as f32, canvas.height as f32);
    let plot_w = w * (1.0 - AMP_HIST_FRAC);
    let dur = s.duration_sec.max(1e-9) as f32;
    let amax = amplitude_max(req);
    let y_of = |a: f32| (1.0 - a.abs() / amax) * h;

    // Current timeline window
    let (t0, t1) = req.window;
    canvas.rect(t0 as f32 / dur * plot_w, 0.0, (t1 as f32 / dur * plot_w).max(t0 as f32 / dur * plot_w + 1.0), h, ACCENT, 0.12);
    canvas.vline(plot_w as usize, 0, canvas.height, GRID);

    const BINS: usize = 40;
    for (sel_i, &id) in req.selected.iter().enumerate() {
        let Some(c) = s.cluster(id) else { continue };
        let color = selection_color(sel_i);
        let mut hist = [0u32; BINS];
        for &i in &c.spikes {
            let a = s.amplitudes[i];
            canvas.dot(s.times_sec[i] as f32 / dur * plot_w, y_of(a), 1.5 * req.scale, color, 0.75);
            hist[((a.abs() / amax * BINS as f32) as usize).min(BINS - 1)] += 1;
        }
        let max = hist.iter().copied().max().unwrap_or(0).max(1) as f32;
        for (b, &count) in hist.iter().enumerate() {
            let (ya, yb) = (h * (1.0 - (b + 1) as f32 / BINS as f32), h * (1.0 - b as f32 / BINS as f32));
            let len = (w - plot_w - 4.0) * count as f32 / max;
            canvas.rect(plot_w + 2.0, ya, plot_w + 2.0 + len, yb, color, 0.5);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::Dataset;

    fn request(plot: SpikePlot) -> SpikePlotRequest {
        let ds = Dataset::generate_synthetic(8, 30_000.0, 2.0);
        let sorting = Arc::new(Sorting::run(&ds, &crate::modules::spikes::sorting::tests::synthetic_params(8)));
        let selected = sorting.clusters.iter().take(2).map(|c| c.id).collect();
        SpikePlotRequest { plot, width: 300, height: 200, scale: 1.0, sorting, selected, window: (0.0, 0.1) }
    }

    #[test]
    fn test_every_plot_draws_and_labels() {
        for plot in [
            SpikePlot::Waveforms,
            SpikePlot::Features(FeatureAxes::Pc1Pc2),
            SpikePlot::Features(FeatureAxes::DepthPc1),
            SpikePlot::Correlograms,
            SpikePlot::Isi,
            SpikePlot::Amplitudes,
        ] {
            let req = request(plot);
            let buf = render(&req);
            assert_eq!((buf.width(), buf.height()), (300, 200));
            assert!(buf.as_slice().iter().any(|p| *p != BG), "{plot:?} drew nothing");
            let ax = axes(&req);
            assert!(!ax.ticks.is_empty(), "{plot:?} has no ticks");
            assert!(ax.ticks.iter().all(|t| (-0.001..=1.001).contains(&t.1)), "{plot:?} tick outside plot");
        }
    }

    #[test]
    fn test_empty_selection_is_safe() {
        let mut req = request(SpikePlot::Waveforms);
        req.selected.clear();
        for plot in [SpikePlot::Waveforms, SpikePlot::Correlograms, SpikePlot::Isi, SpikePlot::Amplitudes] {
            req.plot = plot;
            render(&req);
            axes(&req);
        }
    }

    #[test]
    fn test_nice_ticks() {
        // Range 1.5 / 6 targets -> 0.5 step
        let t = nice_ticks(-0.5, 1.0, 6);
        let labels: Vec<&str> = t.iter().map(|x| x.1.as_str()).collect();
        assert_eq!(labels, vec!["-0.5", "0.0", "0.5", "1.0"]);
        assert!(nice_ticks(1.0, 1.0, 5).is_empty());
    }
}
