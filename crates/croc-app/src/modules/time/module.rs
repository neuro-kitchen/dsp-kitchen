//! Time module view model: traces/heatmap views over the shared recording, the timeline,
//! per-view channel selection, and the module's panels.

use std::sync::Arc;
use std::time::Instant;

use serde::{Deserialize, Serialize};
use slint::Rgba8Pixel;

use crate::data::{Dataset, SignalSource, SpikeEventStore};
use crate::shared::dock::{Dock, DropSide, ViewId};
use crate::shared::render_worker::RenderJob;
use crate::shared::workspace::DockWorkspace;

use super::renderer::{render_on_worker, TimeViewKind};
use super::timeline::TimelineState;
use super::view::{kind_name, parse_channel_ranges, TimeView};

/// Render key namespace of this module.
pub const MODULE_ID: u8 = 0;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TimePanels {
    pub data: bool,
    pub properties: bool,
    pub timeline: bool,
}

impl Default for TimePanels {
    fn default() -> Self {
        Self { data: true, properties: true, timeline: true }
    }
}

/// One row of the data panel's channel list.
#[derive(Debug, Clone, PartialEq)]
pub struct ChannelRow {
    pub channel: usize,
    pub checked: bool,
    pub events: usize,
}

/// Spikes of selected clusters to mark on traces: (color, [(time_sec, channel)]).
pub type SpikeMarks = Vec<(Rgba8Pixel, Vec<(f64, usize)>)>;

pub struct TimeModule {
    pub ws: DockWorkspace<TimeView>,
    pub timeline: TimelineState,
    pub panels: TimePanels,
    /// Data panel text filter (matches channel numbers).
    pub channel_filter: String,
    /// Overlay: color spikes of the selected clusters (from the spike module) on traces.
    pub show_sorted_spikes: bool,
    last_frame_instant: Instant,
}

impl TimeModule {
    pub fn new(dataset: &Dataset) -> Self {
        let mut m = Self {
            ws: DockWorkspace::new(Vec::new(), Dock::empty(), None),
            timeline: TimelineState::new(dataset.total_duration_sec()),
            panels: TimePanels::default(),
            channel_filter: String::new(),
            show_sorted_spikes: true,
            last_frame_instant: Instant::now(),
        };
        m.reset_layout(dataset.total_channels);
        m
    }

    /// Default arrangement: traces over a heatmap, both showing every channel.
    pub fn reset_layout(&mut self, channels: usize) {
        let all: Vec<usize> = (0..channels).collect();
        let views = vec![TimeView::new(1, TimeViewKind::Traces, all.clone()), TimeView::new(2, TimeViewKind::Heatmap, all)];
        self.ws.reset(views, Dock::stacked(1, 2, 0.65), Some(1));
    }

    /// Adds a view of `kind` beside the focused one, copying its channel selection.
    pub fn add_view(&mut self, kind: TimeViewKind, channels: usize) -> ViewId {
        let src = self.ws.focused_view().cloned();
        let mut v = TimeView::new(self.ws.next_id(), kind, src.as_ref().map_or_else(|| (0..channels).collect(), |s| s.selection.clone()));
        if let Some(s) = src {
            v.lanes = s.lanes;
            v.gain = s.gain;
        }
        let side = if kind == TimeViewKind::Heatmap { DropSide::Bottom } else { DropSide::Right };
        self.ws.add(v, side)
    }

    pub fn set_kind(&mut self, id: ViewId, kind: TimeViewKind) {
        if let Some(v) = self.ws.view_mut(id) {
            if v.kind != kind {
                v.kind = kind;
                v.title = format!("{} {}", kind_name(kind), v.id);
                v.needs_render = true;
            }
        }
    }

    fn mark_all_dirty(&mut self) {
        self.ws.mark_all_dirty();
    }

    /// Traces are stale (e.g. cluster selection changed and the overlay is on).
    pub fn mark_traces_dirty(&mut self) {
        for v in self.ws.views.iter_mut().filter(|v| v.kind == TimeViewKind::Traces) {
            v.needs_render = true;
        }
    }

    /// Render jobs for visible, stale views; clears their flags.
    pub fn take_jobs(&mut self, dataset: &Arc<Dataset>, events: &Arc<SpikeEventStore>, marks: &SpikeMarks) -> Vec<RenderJob> {
        let visible = self.ws.visible();
        let source: Arc<dyn SignalSource> = dataset.clone();
        let marks = if self.show_sorted_spikes { marks.clone() } else { Vec::new() };
        let mut out = Vec::new();
        for v in self.ws.views.iter_mut().filter(|v| v.needs_render && visible.contains(&v.id)) {
            v.needs_render = false;
            let req = v.render_request(&self.timeline, source.clone(), events.clone(), marks.clone());
            let per_px = req.window_sec * dataset.sample_rate / req.width.max(1) as f64;
            out.push(RenderJob {
                key: (MODULE_ID, v.id),
                samples_per_px: Some(per_px),
                render: Box::new(move || render_on_worker(&req)),
            });
        }
        out
    }

    // ------------------------------------------------------------------------
    // Per-view intents (pointer positions are logical px within the plot)
    // ------------------------------------------------------------------------

    pub fn zoom_at(&mut self, id: ViewId, factor: f32, x: f32) {
        let Some(v) = self.ws.view(id) else { return };
        let ratio = (x * v.scale_factor) as f64 / v.canvas_width.max(1) as f64;
        self.timeline.zoom_at(factor as f64, ratio);
        self.mark_all_dirty();
    }

    pub fn pan_pixels(&mut self, id: ViewId, dx: f32) {
        let Some(v) = self.ws.view(id) else { return };
        let dt = -((dx * v.scale_factor) as f64) / v.canvas_width.max(1) as f64 * self.timeline.visible_window_sec;
        self.timeline.pan_time(dt);
        self.mark_all_dirty();
    }

    pub fn zoom_gain(&mut self, id: ViewId, factor: f32) {
        if let Some(v) = self.ws.view_mut(id) {
            v.zoom_gain(factor);
        }
    }

    pub fn scroll_channels(&mut self, id: ViewId, delta: i32) {
        if let Some(v) = self.ws.view_mut(id) {
            v.scroll_by(delta as i64);
        }
    }

    pub fn set_lanes(&mut self, id: ViewId, lanes: usize) {
        if let Some(v) = self.ws.view_mut(id) {
            v.set_lanes(lanes);
        }
    }

    pub fn hover(&mut self, id: ViewId, x: f32, y: f32, dataset: &Dataset) {
        let timeline = self.timeline.clone();
        if let Some(v) = self.ws.view_mut(id) {
            let s = v.scale_factor;
            v.hover = v.hover_readout(x * s, y * s, dataset, &timeline);
        }
    }

    /// Double-click on a heatmap: reveal that channel in a traces view.
    pub fn plot_double_clicked(&mut self, id: ViewId, y: f32) {
        let Some(v) = self.ws.view(id) else { return };
        if v.kind != TimeViewKind::Heatmap {
            return;
        }
        let Some(ch) = v.channel_at(y * v.scale_factor) else { return };
        let visible = self.ws.visible();
        let target = self
            .ws
            .views
            .iter()
            .filter(|t| t.kind == TimeViewKind::Traces && t.selection.contains(&ch))
            .max_by_key(|t| visible.contains(&t.id))
            .map(|t| t.id);
        if let Some(t) = target {
            self.ws.view_mut(t).expect("found").reveal(ch);
            self.ws.activate(t);
        }
    }

    // ------------------------------------------------------------------------
    // Focused-view intents (keyboard, properties, data panel)
    // ------------------------------------------------------------------------

    pub fn focused_zoom_gain(&mut self, factor: f32) {
        if let Some(v) = self.ws.focused_view_mut() {
            v.zoom_gain(factor);
        }
    }

    pub fn focused_scroll(&mut self, delta: i32) {
        if let Some(v) = self.ws.focused_view_mut() {
            v.scroll_by(delta as i64);
        }
    }

    pub fn focused_page(&mut self, forward: bool) {
        if let Some(v) = self.ws.focused_view_mut() {
            v.page(forward);
        }
    }

    pub fn toggle_channel(&mut self, ch: usize, total: usize) {
        if let Some(v) = self.ws.focused_view_mut() {
            v.toggle_channel(ch, total);
        }
    }

    pub fn select_all(&mut self, total: usize) {
        if let Some(v) = self.ws.focused_view_mut() {
            v.set_selection(0..total, total);
        }
    }

    pub fn select_none(&mut self, total: usize) {
        if let Some(v) = self.ws.focused_view_mut() {
            v.set_selection([], total);
        }
    }

    pub fn select_invert(&mut self, total: usize) {
        if let Some(v) = self.ws.focused_view_mut() {
            let keep: Vec<usize> = (0..total).filter(|c| !v.selection.contains(c)).collect();
            v.set_selection(keep, total);
        }
    }

    /// Applies a range expression to the focused view. Returns an error message to show.
    pub fn select_ranges(&mut self, text: &str, total: usize) -> Result<(), String> {
        let channels = parse_channel_ranges(text, total)?;
        if let Some(v) = self.ws.focused_view_mut() {
            v.set_selection(channels, total);
        }
        Ok(())
    }

    /// Channel list rows for the data panel, filtered by `channel_filter`.
    pub fn channel_rows(&self, total: usize, events: &SpikeEventStore) -> Vec<ChannelRow> {
        let selected = self.ws.focused_view().map(|v| v.selection.as_slice()).unwrap_or(&[]);
        let filter = self.channel_filter.trim();
        (0..total)
            .filter(|c| filter.is_empty() || c.to_string().contains(filter))
            .map(|c| ChannelRow { channel: c, checked: selected.contains(&c), events: events.count(c) })
            .collect()
    }

    // ------------------------------------------------------------------------
    // Timeline
    // ------------------------------------------------------------------------

    pub fn on_toggle_play(&mut self) {
        self.timeline.toggle_play();
        self.last_frame_instant = Instant::now();
    }

    /// The time window moved: every time view is stale.
    pub fn timeline_changed(&mut self) {
        self.mark_all_dirty();
    }

    pub fn zoom_center(&mut self, factor: f32) {
        self.timeline.zoom_at(factor as f64, 0.5);
        self.mark_all_dirty();
    }

    pub fn pan_fraction(&mut self, fraction: f32) {
        self.timeline.pan_time(fraction as f64 * self.timeline.visible_window_sec);
        self.mark_all_dirty();
    }

    pub fn pan_overview(&mut self, ratio: f32) {
        self.timeline.pan_time(ratio as f64 * self.timeline.total_duration_sec);
        self.mark_all_dirty();
    }

    pub fn set_window_edges(&mut self, start_ratio: f32, end_ratio: f32) {
        let dur = self.timeline.total_duration_sec;
        let (a, b) = (start_ratio.min(end_ratio) as f64, start_ratio.max(end_ratio) as f64);
        self.timeline.set_window_range(a * dur, b * dur);
        self.mark_all_dirty();
    }

    /// Current window (seconds).
    pub fn window(&self) -> (f64, f64) {
        (self.timeline.window_start_sec, self.timeline.window_start_sec + self.timeline.visible_window_sec)
    }

    /// Spike count per bin across all channels, normalized to [0, 1].
    pub fn overview_density(&self, bins: usize, events: &SpikeEventStore) -> Vec<f32> {
        let dur = self.timeline.total_duration_sec;
        if bins == 0 || dur <= 0.0 {
            return Vec::new();
        }
        let mut counts = vec![0u32; bins];
        for &t in &events.times_sec {
            counts[(((t / dur) * bins as f64) as usize).min(bins - 1)] += 1;
        }
        let max = counts.iter().copied().max().unwrap_or(0).max(1) as f32;
        counts.iter().map(|&c| c as f32 / max).collect()
    }

    /// Frame tick (~60 Hz). Returns `true` if the timeline moved.
    pub fn on_tick(&mut self) -> bool {
        let now = Instant::now();
        let dt = now.duration_since(self.last_frame_instant).as_secs_f64();
        self.last_frame_instant = now;
        if self.timeline.is_playing {
            self.timeline.advance(dt);
            self.mark_all_dirty();
            true
        } else {
            false
        }
    }

    /// A new recording: fresh timeline, selections clipped to its channels (a view left with
    /// nothing selected shows every channel).
    pub fn dataset_changed(&mut self, dataset: &Dataset) {
        self.timeline = TimelineState::new(dataset.total_duration_sec());
        let total = dataset.total_channels;
        for v in &mut self.ws.views {
            let kept: Vec<usize> = v.selection.iter().copied().filter(|&c| c < total).collect();
            let sel = if kept.is_empty() { (0..total).collect() } else { kept };
            v.set_selection(sel, total);
        }
        self.ws.relayout();
    }

    // ------------------------------------------------------------------------
    // Session
    // ------------------------------------------------------------------------

    pub fn session(&self) -> TimeSession {
        TimeSession {
            workspace: self.ws.clone(),
            panels: self.panels.clone(),
            window_sec: self.timeline.visible_window_sec,
            show_sorted_spikes: self.show_sorted_spikes,
        }
    }

    pub fn is_valid(session: &TimeSession, channels: usize) -> bool {
        session.workspace.is_consistent() && session.workspace.views.iter().all(|v| v.selection.iter().all(|&c| c < channels))
    }

    pub fn restore(&mut self, session: TimeSession) {
        self.ws.adopt(session.workspace);
        self.panels = session.panels;
        self.show_sorted_spikes = session.show_sorted_spikes;
        self.timeline.set_window_duration(session.window_sec);
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeSession {
    pub workspace: DockWorkspace<TimeView>,
    pub panels: TimePanels,
    pub window_sec: f64,
    pub show_sorted_spikes: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn module(channels: usize) -> (TimeModule, Arc<Dataset>, Arc<SpikeEventStore>) {
        let ds = Arc::new(Dataset::generate_synthetic(channels, 10_000.0, 1.0));
        let ev = Arc::new(SpikeEventStore::detect(ds.as_ref()));
        let mut m = TimeModule::new(&ds);
        m.ws.resized(1000.0, 600.0, 1.0);
        (m, ds, ev)
    }

    #[test]
    fn test_default_layout_and_jobs() {
        let (mut m, ds, ev) = module(8);
        assert_eq!(m.ws.layout.seats.len(), 2);
        assert_eq!(m.ws.focused_view().unwrap().kind, TimeViewKind::Traces);
        assert_eq!(m.take_jobs(&ds, &ev, &Vec::new()).len(), 2);
        assert!(m.take_jobs(&ds, &ev, &Vec::new()).is_empty());
        m.pan_fraction(0.1);
        let jobs = m.take_jobs(&ds, &ev, &Vec::new());
        assert_eq!(jobs.len(), 2);
        assert!(jobs.iter().all(|j| j.key.0 == MODULE_ID));
    }

    #[test]
    fn test_add_view_copies_selection_and_tabs_render_active_only() {
        let (mut m, ds, ev) = module(8);
        m.select_ranges("0-3", 8).unwrap();
        let id = m.add_view(TimeViewKind::Traces, 8);
        assert_eq!(m.ws.view(id).unwrap().selection, vec![0, 1, 2, 3]);
        m.take_jobs(&ds, &ev, &Vec::new());
        let ids = m.ws.dock.views();
        m.ws.drop_view(ids[1], ids[0], DropSide::Centre);
        m.pan_fraction(0.1);
        assert_eq!(m.take_jobs(&ds, &ev, &Vec::new()).len(), 2);
    }

    #[test]
    fn test_selection_intents_apply_to_focused_view() {
        let (mut m, _, ev) = module(16);
        let other = m.ws.dock.views()[1];
        m.select_none(16);
        assert!(m.ws.focused_view().unwrap().selection.is_empty());
        assert_eq!(m.ws.view(other).unwrap().selection.len(), 16);
        m.select_ranges("0-3, 8", 16).unwrap();
        assert_eq!(m.ws.focused_view().unwrap().selection, vec![0, 1, 2, 3, 8]);
        assert!(m.select_ranges("99", 16).is_err());
        m.toggle_channel(8, 16);
        m.select_invert(16);
        assert_eq!(m.ws.focused_view().unwrap().selection.len(), 12);
        m.channel_filter = "1".into();
        let rows = m.channel_rows(16, &ev);
        assert_eq!(rows.iter().map(|r| r.channel).collect::<Vec<_>>(), vec![1, 10, 11, 12, 13, 14, 15]);
        assert!(!rows[0].checked && rows[1].checked);
    }

    #[test]
    fn test_divider_drag_and_heatmap_reveal() {
        let (mut m, _, _) = module(32);
        let before = m.ws.layout.seats[0].height;
        m.ws.divider_pressed(0);
        m.ws.divider_dragged(0, 50.0);
        assert!((m.ws.layout.seats[0].height - (before + 50.0)).abs() < 0.5);

        let (traces, heat) = (m.ws.dock.views()[0], m.ws.dock.views()[1]);
        let row_h = m.ws.view(heat).unwrap().canvas_height as f32 / 32.0;
        m.plot_double_clicked(heat, row_h * 20.5);
        assert_eq!(m.ws.focused, Some(traces));
        assert!(m.ws.view(traces).unwrap().drawn_channels().contains(&20));
    }

    #[test]
    fn test_session_round_trip_and_validation() {
        let (mut a, _, _) = module(16);
        a.select_ranges("2-5", 16).unwrap();
        a.panels.data = false;
        let json = serde_json::to_string(&a.session()).unwrap();
        let s: TimeSession = serde_json::from_str(&json).unwrap();
        assert!(TimeModule::is_valid(&s, 16));
        assert!(!TimeModule::is_valid(&s, 4));

        let (mut b, _, _) = module(16);
        b.restore(s);
        assert_eq!(b.ws.focused_view().unwrap().selection, vec![2, 3, 4, 5]);
        assert!(!b.panels.data);
        assert_eq!(b.ws.dock, a.ws.dock);
    }
}
