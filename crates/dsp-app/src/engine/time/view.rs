//! State of one time view (traces or heatmap): its source, channel selection, lanes, gain,
//! amplitude scaling, canvas.

use std::sync::Arc;

use serde::{Deserialize, Serialize};
use dsp_core::RecordingSource;
use dsp_base::resampler::{MinMaxCache, MinMaxSummary};

use crate::engine::axis::nice_step;
use crate::engine::canvas::Pixel;
use crate::engine::data::{Dataset, SpikeEventStore};

use super::renderer::{px_per_unit, scale_bar_value, RenderRequest, TimeViewKind, CHANNEL_COLORS, NOMINAL_UV};
use super::timeline::TimelineState;

/// Identifies a view for as long as the app runs (also its render key).
pub type ViewId = u64;

/// A labelled position on the time axis.
#[derive(Debug, Clone, PartialEq)]
pub struct AxisTick {
    pub time_sec: f64,
    /// Horizontal position as a fraction of the plot width.
    pub frac: f32,
    pub label: String,
}

/// A channel label in the left gutter.
#[derive(Debug, Clone, PartialEq)]
pub struct LaneLabel {
    pub label: String,
    pub color: Pixel,
    /// Vertical center as a fraction of the plot height.
    pub y_frac: f32,
}

const LABEL_GRAY: Pixel = Pixel::rgb(139, 148, 158);

/// See [`TimeView::hover_target`].
#[derive(Debug, Clone, PartialEq)]
pub enum HoverTarget {
    Text(String),
    Sample { channel: usize, sample: u64, prefix: String, unit: String },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TimeView {
    pub id: ViewId,
    pub kind: TimeViewKind,
    pub title: String,
    /// Channels this view shows, in display order.
    pub selection: Vec<usize>,
    /// Traces: how many channels are on screen at once.
    pub lanes: usize,
    /// Traces: index into `selection` of the top lane.
    pub scroll: usize,
    pub gain: f32,
    /// Source id within the open file (empty = the file's default source).
    #[serde(default)]
    pub source: String,
    /// Source display name, for the title.
    #[serde(default)]
    pub source_name: String,
    /// Fit the amplitude to the visible data.
    #[serde(default = "yes")]
    pub auto_scale: bool,
    /// Subtract each channel's offset over the window (default on: offsets such as a 36 °C
    /// temperature or an electrode's DC level would otherwise dominate the auto-scale).
    #[serde(default = "yes")]
    pub remove_dc: bool,
    /// Units that fill a lane at gain 1: reported by the renderer (auto-scale) or nominal.
    #[serde(skip, default = "nominal")]
    pub amp_scale: f32,
    /// Unit of the source's values (for the scale bar and readout).
    #[serde(skip, default = "micro")]
    pub unit: String,
    /// Plot area in physical pixels, and physical px per logical px.
    #[serde(skip)]
    pub canvas_width: u32,
    #[serde(skip)]
    pub canvas_height: u32,
    #[serde(skip, default = "one")]
    pub scale_factor: f32,
    #[serde(skip)]
    pub needs_render: bool,
    #[serde(skip)]
    pub hover: String,
    /// Latest hover request of this view (older readouts arriving late are dropped).
    #[serde(skip)]
    pub hover_seq: u64,
    /// Whether the last frame had the complete min/max cache file (redraw once it completes).
    #[serde(skip)]
    pub has_lod: bool,
}

fn one() -> f32 {
    1.0
}
fn yes() -> bool {
    true
}
fn nominal() -> f32 {
    NOMINAL_UV
}
fn micro() -> String {
    "µV".into()
}

impl TimeView {
    pub fn new(id: ViewId, kind: TimeViewKind, selection: Vec<usize>) -> Self {
        Self {
            id,
            kind,
            title: format!("{} {id}", kind_name(kind)),
            selection,
            lanes: 8,
            scroll: 0,
            gain: 1.0,
            source: String::new(),
            source_name: String::new(),
            auto_scale: true,
            remove_dc: true,
            amp_scale: NOMINAL_UV,
            unit: micro(),
            canvas_width: 0,
            canvas_height: 0,
            scale_factor: 1.0,
            needs_render: true,
            hover: String::new(),
            hover_seq: 0,
            has_lod: false,
        }
    }

    /// `Traces — EMG` (or `Traces 3` before a source is known).
    pub fn retitle(&mut self) {
        self.title = if self.source_name.is_empty() {
            format!("{} {}", kind_name(self.kind), self.id)
        } else {
            format!("{} — {}", kind_name(self.kind), self.source_name)
        };
    }

    /// Shows source `id` (all `channels` of it), resetting scroll and scale.
    pub fn set_source(&mut self, id: &str, name: &str, unit: &str, channels: usize) {
        let changed = self.source != id;
        self.source = id.to_string();
        self.source_name = name.to_string();
        self.unit = unit.to_string();
        if changed {
            self.set_selection(0..channels, channels);
            self.scroll = 0;
            self.amp_scale = NOMINAL_UV;
        }
        self.retitle();
        self.needs_render = true;
    }

    /// Lanes actually on screen (traces), at least 1.
    pub fn lanes_on_screen(&self) -> usize {
        self.lanes.clamp(1, self.selection.len().max(1))
    }

    fn max_scroll(&self) -> usize {
        self.selection.len().saturating_sub(self.lanes_on_screen())
    }

    /// Channels drawn: the visible page for traces, every selected channel for heatmap.
    pub fn drawn_channels(&self) -> &[usize] {
        match self.kind {
            TimeViewKind::Traces => {
                let start = self.scroll.min(self.max_scroll());
                let end = (start + self.lanes_on_screen()).min(self.selection.len());
                &self.selection[start..end]
            }
            _ => &self.selection,
        }
    }

    pub fn range_label(&self) -> String {
        let n = self.selection.len();
        match self.kind {
            TimeViewKind::Heatmap => format!("{n} channels"),
            TimeViewKind::Traces => {
                let start = self.scroll.min(self.max_scroll());
                let shown = self.drawn_channels().len();
                if n == 0 {
                    "No channels".into()
                } else {
                    format!("{}–{} of {n}", start + 1, start + shown)
                }
            }
        }
    }

    pub fn gain_label(&self) -> String {
        format!("{:.2}×", self.gain)
    }

    pub fn set_canvas(&mut self, width: u32, height: u32, scale: f32) {
        if (width, height, scale) != (self.canvas_width, self.canvas_height, self.scale_factor) {
            self.canvas_width = width;
            self.canvas_height = height;
            self.scale_factor = scale;
            self.needs_render = true;
        }
    }

    pub fn scroll_by(&mut self, delta: i64) {
        let next = (self.scroll.min(self.max_scroll()) as i64 + delta).clamp(0, self.max_scroll() as i64);
        self.scroll = next as usize;
        self.needs_render = true;
    }

    pub fn page(&mut self, forward: bool) {
        let step = self.lanes_on_screen() as i64;
        self.scroll_by(if forward { step } else { -step });
    }

    pub fn set_lanes(&mut self, lanes: usize) {
        self.lanes = lanes.clamp(1, self.selection.len().max(1));
        self.scroll = self.scroll.min(self.max_scroll());
        self.needs_render = true;
    }

    pub fn zoom_gain(&mut self, factor: f32) {
        self.gain = (self.gain * factor).clamp(0.1, 20.0);
        self.needs_render = true;
    }

    /// Replaces the selection (duplicates and out-of-range channels removed, order kept).
    pub fn set_selection(&mut self, channels: impl IntoIterator<Item = usize>, total: usize) {
        let mut seen = vec![false; total];
        self.selection = channels
            .into_iter()
            .filter(|&c| c < total && !std::mem::replace(&mut seen[c], true))
            .collect();
        self.scroll = self.scroll.min(self.max_scroll());
        self.needs_render = true;
    }

    /// Toggles one channel; added channels keep ascending channel order.
    #[cfg(test)]
    pub fn toggle_channel(&mut self, ch: usize, total: usize) {
        if let Some(i) = self.selection.iter().position(|&c| c == ch) {
            self.selection.remove(i);
        } else if ch < total {
            let at = self.selection.partition_point(|&c| c < ch);
            self.selection.insert(at, ch);
        }
        self.scroll = self.scroll.min(self.max_scroll());
        self.needs_render = true;
    }

    /// Scrolls so `ch` is on screen (roughly centred). Returns false if not selected.
    pub fn reveal(&mut self, ch: usize) -> bool {
        let Some(i) = self.selection.iter().position(|&c| c == ch) else { return false };
        self.scroll = i.saturating_sub(self.lanes_on_screen() / 2).min(self.max_scroll());
        self.needs_render = true;
        true
    }

    // ------------------------------------------------------------------------
    // Presentation
    // ------------------------------------------------------------------------

    /// Time-axis ticks at a 1-2-5 step, roughly one per 110 logical pixels.
    pub fn time_ticks(&self, timeline: &TimelineState) -> Vec<AxisTick> {
        let t0 = timeline.window_start_sec;
        let win = timeline.visible_window_sec;
        let logical_w = self.canvas_width as f32 / self.scale_factor.max(0.1);
        let target = (logical_w / 110.0).clamp(2.0, 12.0) as f64;
        let step = nice_step(win / target);
        let decimals = (-step.log10().floor()).max(0.0) as usize;

        let mut ticks = Vec::new();
        let mut i = (t0 / step).ceil() as i64;
        loop {
            let t = i as f64 * step;
            if t > t0 + win + 1e-12 {
                break;
            }
            ticks.push(AxisTick {
                time_sec: t,
                frac: ((t - t0) / win) as f32,
                label: format!("{t:.decimals$} s"),
            });
            i += 1;
        }
        ticks
    }

    /// Channel labels for the left gutter (every lane for traces, sparse rows for heatmap).
    pub fn lane_labels(&self) -> Vec<LaneLabel> {
        let drawn = self.drawn_channels();
        let n = drawn.len().max(1) as f32;
        match self.kind {
            TimeViewKind::Traces => drawn
                .iter()
                .enumerate()
                .map(|(lane, &ch)| LaneLabel {
                    label: format!("Ch {ch}"),
                    color: CHANNEL_COLORS[ch % CHANNEL_COLORS.len()],
                    y_frac: (lane as f32 + 0.5) / n,
                })
                .collect(),
            _ => {
                let step = (nice_step(drawn.len() as f64 / 12.0).round() as usize).max(1);
                drawn
                    .iter()
                    .enumerate()
                    .step_by(step)
                    .map(|(row, &ch)| LaneLabel {
                        label: format!("{ch}"),
                        color: LABEL_GRAY,
                        y_frac: (row as f32 + 0.5) / n,
                    })
                    .collect()
            }
        }
    }

    /// Pixels per unit in a lane, with the current gain and amplitude scale.
    fn px_per_unit(&self) -> (f32, f32) {
        let lane_h = self.canvas_height as f32 / self.drawn_channels().len().max(1) as f32;
        (lane_h, px_per_unit(lane_h, self.gain, self.amp_scale))
    }

    /// Scale bar amplitude (1-2-5 series, in the source unit), about 45% of a lane; 0 in
    /// heatmap mode. Same formula as the renderer's bar.
    pub fn scale_bar_value(&self) -> f32 {
        if self.kind != TimeViewKind::Traces || self.drawn_channels().is_empty() {
            return 0.0;
        }
        let (lane_h, k) = self.px_per_unit();
        scale_bar_value(lane_h, k)
    }

    /// e.g. `50 µV`, `0.2 a.u.`
    pub fn scale_bar_label(&self) -> String {
        let bar = self.scale_bar_value();
        if bar > 0.0 { format!("{} {}", fmt_amount(bar), self.unit) } else { String::new() }
    }

    /// Vertical center of the scale bar as a fraction of the plot height (matches the raster:
    /// bar bottom sits `6 * scale` px above the plot bottom).
    pub fn scale_bar_center_frac(&self) -> f32 {
        let h = self.canvas_height.max(1) as f32;
        let lanes = self.drawn_channels().len().max(1) as f32;
        let bar_px = self.scale_bar_value() * px_per_unit(h / lanes, self.gain, self.amp_scale);
        ((h - 6.0 * self.scale_factor - bar_px / 2.0) / h).clamp(0.0, 1.0)
    }

    /// Channel under a physical-pixel y in the plot.
    pub fn channel_at(&self, y_px: f32) -> Option<usize> {
        let drawn = self.drawn_channels();
        if drawn.is_empty() {
            return None;
        }
        let y_frac = (y_px / self.canvas_height.max(1) as f32).clamp(0.0, 0.9999);
        drawn.get((y_frac * drawn.len() as f32) as usize).copied()
    }

    /// What the cursor at physical pixel (x, y) in the plot points at: a readout that needs no
    /// data (`Text`), or the sample whose value completes it (`Sample`, read off the UI thread).
    pub fn hover_target(&self, x_px: f32, y_px: f32, dataset: &Dataset, timeline: &TimelineState) -> HoverTarget {
        let Some(ch) = self.channel_at(y_px) else { return HoverTarget::Text(String::new()) };
        let w = self.canvas_width.max(1) as f32;
        let t = timeline.window_start_sec + (x_px / w).clamp(0.0, 1.0) as f64 * timeline.visible_window_sec;
        let rel = (t - dataset.start_time_sec) * dataset.sample_rate;
        if rel < 0.0 || rel >= dataset.total_samples as f64 {
            return HoverTarget::Text(format!("Ch {ch}  ·  {t:.4} s  ·  no data"));
        }
        HoverTarget::Sample { channel: ch, sample: rel as u64, prefix: format!("Ch {ch}  ·  {t:.4} s  ·  "), unit: self.unit.clone() }
    }

    pub fn render_request(
        &self,
        timeline: &TimelineState,
        source: Arc<dyn RecordingSource>,
        lod: Option<Arc<MinMaxCache>>,
        summary: Option<Arc<MinMaxSummary>>,
        events: Arc<SpikeEventStore>,
        highlights: Vec<(Pixel, Vec<(f64, usize)>)>,
    ) -> RenderRequest {
        let start_time_sec = source.info().start_time_sec;
        RenderRequest {
            source,
            lod,
            summary,
            events,
            width: self.canvas_width.max(1),
            height: self.canvas_height.max(1),
            scale: self.scale_factor,
            mode: self.kind,
            channels: self.drawn_channels().to_vec(),
            window_start_sec: timeline.window_start_sec,
            window_sec: timeline.visible_window_sec,
            start_time_sec,
            amplitude_scale: self.gain,
            auto_scale: self.auto_scale,
            remove_dc: self.remove_dc,
            scale_hint: if self.auto_scale { self.amp_scale } else { 0.0 },
            grid_times: self.time_ticks(timeline).iter().map(|t| t.time_sec).collect(),
            scale_bar: self.kind == TimeViewKind::Traces,
            highlights,
        }
    }
}

/// Compact amount for labels: `50`, `0.2`, `1.5e-3`.
pub fn fmt_amount(v: f32) -> String {
    let a = v.abs();
    if a == 0.0 {
        "0".into()
    } else if a >= 1e6 || a < 0.01 {
        format!("{v:.2e}")
    } else if a >= 1000.0 {
        format!("{v:.0}")
    } else if a >= 10.0 {
        format!("{v:.1}").trim_end_matches(".0").to_string()
    } else {
        let t = format!("{v:.3}");
        t.trim_end_matches('0').trim_end_matches('.').to_string()
    }
}

pub fn kind_name(kind: TimeViewKind) -> &'static str {
    match kind {
        TimeViewKind::Traces => "Traces",
        TimeViewKind::Heatmap => "Heatmap",
    }
}

/// Parses a channel list like `"0-31, 40, 64-95"` (inclusive ranges, any order).
pub fn parse_channel_ranges(text: &str, total: usize) -> Result<Vec<usize>, String> {
    let mut out = Vec::new();
    for part in text.split(',').map(str::trim).filter(|p| !p.is_empty()) {
        let (a, b) = match part.split_once('-') {
            Some((a, b)) => (a.trim(), b.trim()),
            None => (part, part),
        };
        let parse = |v: &str| v.parse::<usize>().map_err(|_| format!("'{part}' is not a channel or range"));
        let (lo, hi) = (parse(a)?, parse(b)?);
        if lo > hi {
            return Err(format!("'{part}': start is after end"));
        }
        if hi >= total {
            return Err(format!("'{part}': channel {hi} does not exist (0–{})", total.saturating_sub(1)));
        }
        out.extend(lo..=hi);
    }
    if out.is_empty() {
        return Err("Enter channels, e.g. 0-31, 40".into());
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_channel_ranges() {
        assert_eq!(parse_channel_ranges("0-3, 7,5-5", 10).unwrap(), vec![0, 1, 2, 3, 7, 5]);
        assert!(parse_channel_ranges("3-1", 10).is_err());
        assert!(parse_channel_ranges("0-10", 10).is_err());
        assert!(parse_channel_ranges("a", 10).is_err());
        assert!(parse_channel_ranges(" , ", 10).is_err());
    }

    #[test]
    fn test_selection_paging_and_toggle() {
        let mut v = TimeView::new(1, TimeViewKind::Traces, (0..16).collect());
        assert_eq!(v.drawn_channels(), &[0, 1, 2, 3, 4, 5, 6, 7]);
        v.page(true);
        assert_eq!(v.range_label(), "9–16 of 16");
        v.page(true);
        assert_eq!(v.scroll, 8);
        v.set_lanes(4);
        assert_eq!(v.drawn_channels(), &[8, 9, 10, 11]);
        v.scroll_by(-100);
        assert_eq!(v.scroll, 0);

        v.toggle_channel(2, 16);
        assert!(!v.selection.contains(&2));
        v.toggle_channel(2, 16);
        assert_eq!(&v.selection[..4], &[0, 1, 2, 3]);

        v.set_selection([5, 3, 5, 99], 16);
        assert_eq!(v.selection, vec![5, 3]);
        assert_eq!(v.drawn_channels(), &[5, 3]);
        assert!(v.reveal(3));
        assert!(!v.reveal(0));

        v.kind = TimeViewKind::Heatmap;
        assert_eq!(v.range_label(), "2 channels");
    }

    #[test]
    fn test_time_ticks() {
        let mut v = TimeView::new(1, TimeViewKind::Traces, vec![0]);
        v.set_canvas(1100, 400, 1.0);
        let mut tl = TimelineState::new(2.0);
        tl.window_start_sec = 0.03;
        tl.visible_window_sec = 0.1;
        let ticks = v.time_ticks(&tl);
        assert_eq!(ticks.first().unwrap().label, "0.03 s");
        assert_eq!(ticks.len(), 11);
    }

    #[test]
    fn test_channel_at_maps_rows() {
        let mut v = TimeView::new(1, TimeViewKind::Heatmap, (10..42).collect());
        v.set_canvas(800, 320, 1.0);
        assert_eq!(v.channel_at(205.0), Some(30));
        v.kind = TimeViewKind::Traces;
        v.set_lanes(4);
        assert_eq!(v.channel_at(0.0), Some(10));
        assert_eq!(v.channel_at(319.0), Some(13));
    }
}
