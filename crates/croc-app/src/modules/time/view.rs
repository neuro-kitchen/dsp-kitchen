//! State of one time view (traces or heatmap): channel selection, lanes, gain, canvas.

use std::sync::Arc;

use serde::{Deserialize, Serialize};
use slint::Rgba8Pixel;

use crate::data::{Dataset, SignalSource, SpikeEventStore};
use crate::shared::axis::nice_step;
use crate::shared::dock::ViewId;
use crate::shared::workspace::DockView;

use super::renderer::{px_per_uv, RenderRequest, TimeViewKind, CHANNEL_COLORS};
use super::timeline::TimelineState;

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
    pub color: Rgba8Pixel,
    /// Vertical center as a fraction of the plot height.
    pub y_frac: f32,
}

const LABEL_GRAY: Rgba8Pixel = Rgba8Pixel { r: 139, g: 148, b: 158, a: 255 };

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
}

fn one() -> f32 {
    1.0
}

impl DockView for TimeView {
    fn id(&self) -> ViewId {
        self.id
    }
    fn set_canvas(&mut self, width: u32, height: u32, scale: f32) {
        TimeView::set_canvas(self, width, height, scale);
    }
    fn mark_dirty(&mut self) {
        self.needs_render = true;
    }
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
            canvas_width: 0,
            canvas_height: 0,
            scale_factor: 1.0,
            needs_render: true,
            hover: String::new(),
        }
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

    /// Scale bar amplitude (1-2-5 µV) sized to about 45% of a lane; 0 in heatmap mode.
    pub fn scale_bar_uv(&self) -> f32 {
        if self.kind != TimeViewKind::Traces || self.drawn_channels().is_empty() {
            return 0.0;
        }
        let lane_h = self.canvas_height as f32 / self.drawn_channels().len() as f32;
        let k = px_per_uv(lane_h, self.gain);
        if k <= 0.0 {
            return 0.0;
        }
        nice_step((0.45 * lane_h / k) as f64) as f32
    }

    /// Vertical center of the scale bar as a fraction of the plot height (matches the raster:
    /// bar bottom sits `6 * scale` px above the plot bottom).
    pub fn scale_bar_center_frac(&self) -> f32 {
        let h = self.canvas_height.max(1) as f32;
        let lanes = self.drawn_channels().len().max(1) as f32;
        let bar_px = self.scale_bar_uv() * px_per_uv(h / lanes, self.gain);
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

    /// Readout for the cursor at physical pixel (x, y) in the plot.
    pub fn hover_readout(&self, x_px: f32, y_px: f32, dataset: &Dataset, timeline: &TimelineState) -> String {
        let Some(ch) = self.channel_at(y_px) else { return String::new() };
        let w = self.canvas_width.max(1) as f32;
        let t = timeline.window_start_sec + (x_px / w).clamp(0.0, 1.0) as f64 * timeline.visible_window_sec;
        let sample = ((t * dataset.sample_rate) as usize).min(dataset.total_samples.saturating_sub(1));
        let value = dataset.channel(ch).get(sample).copied().unwrap_or(0.0);
        format!("Ch {ch}  ·  {t:.4} s  ·  {value:.1} µV")
    }

    pub fn render_request(
        &self,
        timeline: &TimelineState,
        source: Arc<dyn SignalSource>,
        events: Arc<SpikeEventStore>,
        highlights: Vec<(Rgba8Pixel, Vec<(f64, usize)>)>,
    ) -> RenderRequest {
        RenderRequest {
            source,
            events,
            width: self.canvas_width.max(1),
            height: self.canvas_height.max(1),
            scale: self.scale_factor,
            mode: self.kind,
            channels: self.drawn_channels().to_vec(),
            window_start_sec: timeline.window_start_sec,
            window_sec: timeline.visible_window_sec,
            amplitude_scale: self.gain,
            grid_times: self.time_ticks(timeline).iter().map(|t| t.time_sec).collect(),
            scale_bar_uv: self.scale_bar_uv(),
            highlights,
        }
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
