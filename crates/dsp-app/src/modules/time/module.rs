//! Time module view model: traces/heatmap views, each on a source of the open file, the shared
//! timeline, per-view channel selection, and the module's panels.

use std::sync::Arc;
use std::time::Instant;

use serde::{Deserialize, Serialize};
use slint::Rgba8Pixel;
use dsp_core::RecordingSource;

use crate::data::{SourceSet, SpikeEventStore};
use crate::shared::dock::{Dock, DropSide, ViewId};
use crate::shared::render_worker::RenderJob;
use crate::shared::workspace::DockWorkspace;

use super::hover::HoverRequest;
use super::renderer::{render_on_worker, TimeViewKind};
use super::timeline::TimelineState;
use super::view::{parse_channel_ranges, HoverTarget, TimeView};

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
    pub fn new(sources: &SourceSet) -> Self {
        let mut m = Self {
            ws: DockWorkspace::new(Vec::new(), Dock::empty(), None),
            timeline: TimelineState::new(sources.extent_sec()),
            panels: TimePanels::default(),
            channel_filter: String::new(),
            show_sorted_spikes: true,
            last_frame_instant: Instant::now(),
        };
        m.reset_layout(sources);
        m
    }

    /// Default arrangement: traces over a heatmap of the default source, every channel.
    pub fn reset_layout(&mut self, sources: &SourceSet) {
        let e = sources.default_entry();
        let mut views = vec![TimeView::new(1, TimeViewKind::Traces, Vec::new()), TimeView::new(2, TimeViewKind::Heatmap, Vec::new())];
        for v in &mut views {
            v.set_source(&e.id, &e.name, &e.unit, e.channels);
        }
        self.ws.reset(views, Dock::stacked(1, 2, 0.65), Some(1));
    }

    /// Adds a view of `kind` beside the focused one, on the same source with its selection.
    pub fn add_view(&mut self, kind: TimeViewKind, sources: &SourceSet) -> ViewId {
        let src = self.ws.focused_view().cloned();
        let mut v = TimeView::new(self.ws.next_id(), kind, Vec::new());
        match src {
            Some(s) => {
                let e = sources.entry(&s.source);
                v.set_source(&e.id, &e.name, &e.unit, e.channels);
                v.selection = s.selection.clone();
                v.lanes = s.lanes;
                v.gain = s.gain;
                v.auto_scale = s.auto_scale;
                v.remove_dc = s.remove_dc;
                v.amp_scale = s.amp_scale;
            }
            None => {
                let e = sources.default_entry();
                v.set_source(&e.id, &e.name, &e.unit, e.channels);
            }
        }
        let side = if kind == TimeViewKind::Heatmap { DropSide::Bottom } else { DropSide::Right };
        self.ws.add(v, side)
    }

    /// Adds a traces view of source `index` (every channel), beside the focused view.
    pub fn add_view_for(&mut self, index: usize, sources: &SourceSet) -> Option<ViewId> {
        let e = sources.entries().get(index)?;
        let mut v = TimeView::new(self.ws.next_id(), TimeViewKind::Traces, Vec::new());
        v.set_source(&e.id, &e.name, &e.unit, e.channels);
        Some(self.ws.add(v, DropSide::Right))
    }

    /// The focused view shows source `index` instead.
    pub fn set_focused_source(&mut self, index: usize, sources: &SourceSet) {
        let Some(e) = sources.entries().get(index) else { return };
        if let Some(v) = self.ws.focused_view_mut() {
            v.set_source(&e.id, &e.name, &e.unit, e.channels);
        }
    }

    /// Source id of the focused view (empty when there is none: the default).
    pub fn focused_source(&self) -> String {
        self.ws.focused_view().map(|v| v.source.clone()).unwrap_or_default()
    }

    pub fn set_focused_auto_scale(&mut self, on: bool) {
        if let Some(v) = self.ws.focused_view_mut() {
            v.auto_scale = on;
            v.needs_render = true;
        }
    }

    pub fn set_focused_remove_dc(&mut self, on: bool) {
        if let Some(v) = self.ws.focused_view_mut() {
            v.remove_dc = on;
            v.needs_render = true;
        }
    }

    /// The renderer reports the amplitude scale it drew `id` with.
    pub fn frame_scale(&mut self, id: ViewId, scale: f32) {
        if let Some(v) = self.ws.view_mut(id) {
            v.amp_scale = scale;
        }
    }

    pub fn set_kind(&mut self, id: ViewId, kind: TimeViewKind) {
        if let Some(v) = self.ws.view_mut(id) {
            if v.kind != kind {
                v.kind = kind;
                v.retitle();
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

    /// Render jobs for visible, stale views; clears their flags. Events and spike marks belong
    /// to the default source and are drawn only on its views.
    pub fn take_jobs(&mut self, sources: &SourceSet, events: &Arc<SpikeEventStore>, marks: &SpikeMarks) -> Vec<RenderJob> {
        let visible = self.ws.visible();
        let marks = if self.show_sorted_spikes { marks.clone() } else { Vec::new() };
        let default = sources.index_of("");
        let no_events = Arc::new(SpikeEventStore::default());
        let mut out = Vec::new();
        for v in self.ws.views.iter_mut().filter(|v| visible.contains(&v.id)) {
            let dataset = sources.get(&v.source);
            // Redraw once the min/max cache file completes
            let lod = dataset.lod();
            if lod.is_some() != v.has_lod {
                v.has_lod = lod.is_some();
                v.needs_render = true;
            }
            if !v.needs_render {
                continue;
            }
            v.needs_render = false;
            let on_default = sources.index_of(&v.source) == default;
            let source: Arc<dyn RecordingSource> = dataset.clone();
            let (ev, mk) = if on_default { (events.clone(), marks.clone()) } else { (no_events.clone(), Vec::new()) };
            let req = v.render_request(&self.timeline, source, lod, Some(dataset.summary()), ev, mk);
            let per_px = req.window_sec * dataset.sample_rate / req.width.max(1) as f64;
            out.push(RenderJob {
                key: (MODULE_ID, v.id),
                samples_per_px: Some(per_px),
                render: Box::new(move |ctx| render_on_worker(&req, ctx)),
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

    /// Pointer at logical (x, y) in view `id`: sets a readout that needs no data now, or returns
    /// the sample read to request (the view keeps its previous readout until the answer).
    pub fn hover(&mut self, id: ViewId, x: f32, y: f32, sources: &SourceSet) -> Option<HoverRequest> {
        let timeline = self.timeline.clone();
        let v = self.ws.view_mut(id)?;
        let s = v.scale_factor;
        let dataset = sources.get(&v.source);
        v.hover_seq += 1;
        match v.hover_target(x * s, y * s, &dataset, &timeline) {
            HoverTarget::Text(text) => {
                v.hover = text;
                None
            }
            HoverTarget::Sample { channel, sample, prefix, unit } => {
                Some(HoverRequest { view: id, seq: v.hover_seq, dataset, channel, sample, prefix, unit })
            }
        }
    }

    /// A readout arrived; returns whether it is the view's latest (and was applied).
    pub fn hover_answered(&mut self, id: ViewId, seq: u64, text: String) -> bool {
        match self.ws.view_mut(id) {
            Some(v) if v.hover_seq == seq => {
                v.hover = text;
                true
            }
            _ => false,
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

    /// A new file: fresh timeline and every view re-attached to its sources.
    pub fn dataset_changed(&mut self, sources: &SourceSet) {
        self.timeline = TimelineState::new(sources.extent_sec());
        self.attach_sources(sources);
        self.ws.relayout();
    }

    /// Binds every view to a source of `sources`: views whose source is missing show the
    /// default; selections are clipped to their source's channels (a view left with nothing
    /// selected shows every channel).
    pub fn attach_sources(&mut self, sources: &SourceSet) {
        for v in &mut self.ws.views {
            let e = sources.entry(&v.source);
            let (id, name, unit, total) = (e.id.clone(), e.name.clone(), e.unit.clone(), e.channels);
            if v.source != id {
                v.source.clear(); // force the reset in set_source
            } else {
                let kept: Vec<usize> = v.selection.iter().copied().filter(|&c| c < total).collect();
                let sel = if kept.is_empty() { (0..total).collect() } else { kept };
                v.set_selection(sel, total);
            }
            v.set_source(&id, &name, &unit, total);
        }
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

    pub fn is_valid(session: &TimeSession) -> bool {
        session.workspace.is_consistent()
    }

    /// Adopts a saved session, re-binding its views to `sources` (older sessions have no
    /// sources: their views show the default source).
    pub fn restore(&mut self, session: TimeSession, sources: &SourceSet) {
        self.ws.adopt(session.workspace);
        self.panels = session.panels;
        self.show_sorted_spikes = session.show_sorted_spikes;
        self.timeline.set_window_duration(session.window_sec);
        self.attach_sources(sources);
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
    use crate::data::Dataset;

    fn module(channels: usize) -> (TimeModule, SourceSet, Arc<SpikeEventStore>) {
        let sources = SourceSet::single(Dataset::generate_synthetic(channels, 10_000.0, 1.0));
        let ev = Arc::new(SpikeEventStore::detect(sources.default_dataset().as_ref()));
        let mut m = TimeModule::new(&sources);
        m.ws.resized(1000.0, 600.0, 1.0);
        (m, sources, ev)
    }

    /// EMG: 3 ch at 1 kHz from t = 0 (default); Temp: 1 ch at 100 Hz from t = 1 s, 5 s long.
    fn two_sources() -> SourceSet {
        let mut temp = Dataset::generate_synthetic(1, 100.0, 5.0);
        temp.start_time_sec = 1.0;
        temp.unit = "a.u.".into();
        SourceSet::many(vec![("EMG", Dataset::generate_synthetic(3, 1000.0, 2.0), true), ("Temp", temp, false)])
    }

    #[test]
    fn test_default_layout_and_jobs() {
        let (mut m, src, ev) = module(8);
        assert_eq!(m.ws.layout.seats.len(), 2);
        assert_eq!(m.ws.focused_view().unwrap().kind, TimeViewKind::Traces);
        assert_eq!(m.take_jobs(&src, &ev, &Vec::new()).len(), 2);
        assert!(m.take_jobs(&src, &ev, &Vec::new()).is_empty());
        m.pan_fraction(0.1);
        let jobs = m.take_jobs(&src, &ev, &Vec::new());
        assert_eq!(jobs.len(), 2);
        assert!(jobs.iter().all(|j| j.key.0 == MODULE_ID));
    }

    #[test]
    fn test_add_view_copies_selection_and_tabs_render_active_only() {
        let (mut m, src, ev) = module(8);
        m.select_ranges("0-3", 8).unwrap();
        let id = m.add_view(TimeViewKind::Traces, &src);
        assert_eq!(m.ws.view(id).unwrap().selection, vec![0, 1, 2, 3]);
        m.take_jobs(&src, &ev, &Vec::new());
        let ids = m.ws.dock.views();
        m.ws.drop_view(ids[1], ids[0], DropSide::Centre);
        m.pan_fraction(0.1);
        assert_eq!(m.take_jobs(&src, &ev, &Vec::new()).len(), 2);
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
    fn test_views_pick_sources() {
        let src = two_sources();
        let mut m = TimeModule::new(&src);
        m.ws.resized(1000.0, 600.0, 1.0);
        // Timeline covers both sources: Temp ends at 1 + 5 = 6 s
        assert!((m.timeline.total_duration_sec - 6.0).abs() < 1e-9);
        let focused = m.ws.focused_view().unwrap();
        assert_eq!((focused.source.as_str(), focused.title.as_str(), focused.selection.len()), ("EMG", "Traces — EMG", 3));

        // Switch the focused view to Temp: its channels, unit and title
        m.set_focused_source(1, &src);
        let v = m.ws.focused_view().unwrap();
        assert_eq!((v.source.as_str(), v.selection.clone(), v.unit.as_str(), v.title.as_str()), ("Temp", vec![0], "a.u.", "Traces — Temp"));
        assert_eq!(m.focused_source(), "Temp");

        // A view for a chosen source, and a copy of the focused one
        let emg = m.add_view_for(0, &src).unwrap();
        assert_eq!(m.ws.view(emg).unwrap().selection.len(), 3);
        m.ws.focus(emg);
        let copy = m.add_view(TimeViewKind::Heatmap, &src);
        assert_eq!(m.ws.view(copy).unwrap().source, "EMG");

        // Events and marks only reach views of the default source (EMG)
        let ev = Arc::new(SpikeEventStore::detect(src.default_dataset().as_ref()));
        let jobs = m.take_jobs(&src, &ev, &Vec::new());
        assert!(!jobs.is_empty());
    }

    #[test]
    fn test_session_round_trip_and_validation() {
        let src = two_sources();
        let mut a = TimeModule::new(&src);
        a.ws.resized(1000.0, 600.0, 1.0);
        a.select_ranges("1-2", 3).unwrap();
        a.panels.data = false;
        let other = a.ws.dock.views()[1];
        a.ws.focus(other);
        a.set_focused_source(1, &src);
        a.set_focused_remove_dc(true);
        let json = serde_json::to_string(&a.session()).unwrap();
        let s: TimeSession = serde_json::from_str(&json).unwrap();
        assert!(TimeModule::is_valid(&s));

        let mut b = TimeModule::new(&src);
        b.restore(s.clone(), &src);
        let first = b.ws.dock.views()[0];
        assert_eq!(b.ws.view(first).unwrap().selection, vec![1, 2]);
        let second = b.ws.view(other).unwrap();
        assert_eq!((second.source.as_str(), second.remove_dc, second.title.as_str()), ("Temp", true, "Heatmap — Temp"));
        assert!(!b.panels.data);
        assert_eq!(b.ws.dock, a.ws.dock);

        // A file without the saved source: those views fall back to the default
        let single = SourceSet::single(Dataset::generate_synthetic(16, 1000.0, 1.0));
        let mut c = TimeModule::new(&single);
        c.restore(s, &single);
        assert!(c.ws.views.iter().all(|v| v.source == dsp_io::sources::MAIN && v.selection.iter().all(|&ch| ch < 16)));
    }
}
