//! Main Application ViewModel (Android MVVM style).
//!
//! Owns the dataset, the shared timeline, the list of views and the dock layout that places
//! them. Processes user intents from the View and produces view state and render requests.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use crate::model::{Dataset, SignalSource, SpikeEventStore, TimelineState};
use crate::view::{RenderRequest, ViewMode};

use super::dock::{dragged_ratio, lay_out, Dock, DockLayout, DropSide, ViewId};
use super::persist::{Panels, Session};
use super::view_state::{parse_channel_ranges, ViewKind, ViewState};

/// Seat chrome that surrounds a plot, in logical px. The Slint seat must use the same values
/// (they are pushed to the `Metrics` global at startup).
pub const SEAT_HEADER: f32 = 30.0;
pub const GUTTER: f32 = 64.0;
pub const AXIS: f32 = 22.0;

pub const MAX_RECENT: usize = 8;

/// One row of the data panel's channel list.
#[derive(Debug, Clone, PartialEq)]
pub struct ChannelRow {
    pub channel: usize,
    pub checked: bool,
    pub events: usize,
}

pub struct AppViewModel {
    pub dataset: Arc<Dataset>,
    pub dataset_path: Option<PathBuf>,
    pub events: Arc<SpikeEventStore>,
    pub timeline: TimelineState,
    pub views: Vec<ViewState>,
    pub dock: Dock,
    pub focused: Option<ViewId>,
    pub panels: Panels,
    pub recent: Vec<PathBuf>,
    /// Data panel text filter (matches channel numbers).
    pub channel_filter: String,
    /// Workspace size (logical px) and display scale.
    pub workspace: (f32, f32),
    pub scale_factor: f32,
    /// Current flattened layout, logical px relative to the workspace.
    pub layout: DockLayout,
    /// Set when the layout, panels, or view list changed (UI rebuilds seats).
    pub layout_dirty: bool,
    next_id: ViewId,
    divider_drag: Option<(usize, f32)>,
    last_frame_instant: Instant,
}

impl AppViewModel {
    pub fn new(dataset: Dataset, dataset_path: Option<PathBuf>) -> Self {
        let events = SpikeEventStore::detect(&dataset);
        let timeline = TimelineState::new(dataset.total_duration_sec());
        let mut vm = Self {
            dataset: Arc::new(dataset),
            dataset_path,
            events: Arc::new(events),
            timeline,
            views: Vec::new(),
            dock: Dock::empty(),
            focused: None,
            panels: Panels::default(),
            recent: Vec::new(),
            channel_filter: String::new(),
            workspace: (1000.0, 600.0),
            scale_factor: 1.0,
            layout: DockLayout::default(),
            layout_dirty: true,
            next_id: 1,
            divider_drag: None,
            last_frame_instant: Instant::now(),
        };
        vm.reset_layout();
        vm
    }

    // ========================================================================
    // Lookup
    // ========================================================================

    pub fn view(&self, id: ViewId) -> Option<&ViewState> {
        self.views.iter().find(|v| v.id == id)
    }

    pub fn view_mut(&mut self, id: ViewId) -> Option<&mut ViewState> {
        self.views.iter_mut().find(|v| v.id == id)
    }

    pub fn focused_view(&self) -> Option<&ViewState> {
        self.focused.and_then(|id| self.view(id))
    }

    fn focused_view_mut(&mut self) -> Option<&mut ViewState> {
        let id = self.focused?;
        self.view_mut(id)
    }

    fn all_channels(&self) -> Vec<usize> {
        (0..self.dataset.total_channels).collect()
    }

    fn mark_all_dirty(&mut self) {
        for v in &mut self.views {
            v.needs_render = true;
        }
    }

    // ========================================================================
    // Views & layout
    // ========================================================================

    fn new_view(&mut self, kind: ViewKind, selection: Vec<usize>) -> ViewId {
        let id = self.next_id;
        self.next_id += 1;
        self.views.push(ViewState::new(id, kind, selection));
        id
    }

    /// Default arrangement: traces over a heatmap, both showing every channel.
    pub fn reset_layout(&mut self) {
        self.views.clear();
        let all = self.all_channels();
        let traces = self.new_view(ViewMode::Traces, all.clone());
        let heat = self.new_view(ViewMode::Heatmap, all);
        self.dock = Dock::stacked(traces, heat, 0.65);
        self.focused = Some(traces);
        self.relayout();
    }

    /// Every view as a tab of one seat.
    pub fn compact_layout(&mut self) {
        let ids: Vec<ViewId> = self.dock.views();
        let active = self.focused.and_then(|f| ids.iter().position(|&v| v == f)).unwrap_or(0);
        self.dock = Dock::Tabs { views: ids, active };
        self.relayout();
    }

    /// Adds a view of `kind` beside the focused one, copying its channel selection.
    pub fn add_view(&mut self, kind: ViewKind) -> ViewId {
        let source = self.focused_view().cloned();
        let selection = source.as_ref().map(|v| v.selection.clone()).unwrap_or_else(|| self.all_channels());
        let id = self.new_view(kind, selection);
        if let Some(src) = source {
            let v = self.view_mut(id).expect("just added");
            v.lanes = src.lanes;
            v.gain = src.gain;
        }
        let side = if kind == ViewMode::Heatmap { DropSide::Bottom } else { DropSide::Right };
        let target = self.focused.unwrap_or(0);
        if !self.dock.insert(id, target, side) {
            // Focus pointed at nothing usable: put it next to any existing view
            let any = self.dock.views().first().copied().unwrap_or(0);
            self.dock.insert(id, any, side);
        }
        self.focused = Some(id);
        self.relayout();
        id
    }

    pub fn close_view(&mut self, id: ViewId) {
        self.dock.remove(id);
        self.views.retain(|v| v.id != id);
        if self.focused == Some(id) {
            self.focused = self.dock.views().first().copied();
        }
        self.relayout();
    }

    pub fn set_kind(&mut self, id: ViewId, kind: ViewKind) {
        if let Some(v) = self.view_mut(id) {
            if v.kind != kind {
                v.kind = kind;
                v.title = format!("{} {}", super::view_state::kind_name(kind), v.id);
                v.needs_render = true;
                self.layout_dirty = true;
            }
        }
    }

    pub fn focus(&mut self, id: ViewId) {
        if self.view(id).is_some() && self.focused != Some(id) {
            self.focused = Some(id);
            self.layout_dirty = true;
        }
    }

    pub fn activate(&mut self, id: ViewId) {
        self.dock.activate(id);
        self.focused = Some(id);
        self.relayout();
    }

    pub fn dock_drop(&mut self, view: ViewId, target: ViewId, side: DropSide) {
        if self.dock.move_view(view, target, side) {
            self.focused = Some(view);
            self.relayout();
        }
    }

    pub fn workspace_resized(&mut self, width: f32, height: f32, scale: f32) {
        if (width, height, scale) != (self.workspace.0, self.workspace.1, self.scale_factor) {
            self.workspace = (width, height);
            self.scale_factor = scale.max(0.1);
            self.relayout();
        }
    }

    pub fn divider_pressed(&mut self, index: usize) {
        self.divider_drag = self.dock.ratio(index).map(|r| (index, r));
    }

    /// Divider drag by `delta` logical px since the press.
    pub fn divider_dragged(&mut self, index: usize, delta: f32) {
        let Some((i, start)) = self.divider_drag else { return };
        let Some(extent) = self.layout.dividers.iter().find(|d| d.index == index).map(|d| d.extent) else { return };
        if i == index {
            self.dock.set_ratio(index, dragged_ratio(start, delta, extent));
            self.relayout();
        }
    }

    /// Recomputes seat geometry and every visible view's canvas size.
    pub fn relayout(&mut self) {
        let (w, h) = self.workspace;
        self.layout = lay_out(&self.dock, (0.0, 0.0, w, h));
        let scale = self.scale_factor;
        let sizes: Vec<(ViewId, u32, u32)> = self
            .layout
            .seats
            .iter()
            .filter_map(|s| {
                let id = s.active_view()?;
                let pw = ((s.width - GUTTER) * scale).max(1.0) as u32;
                let ph = ((s.height - SEAT_HEADER - AXIS) * scale).max(1.0) as u32;
                Some((id, pw, ph))
            })
            .collect();
        for (id, pw, ph) in sizes {
            if let Some(v) = self.view_mut(id) {
                v.set_canvas(pw, ph, scale);
            }
        }
        self.layout_dirty = true;
    }

    /// Views on screen (active tab of each seat).
    pub fn visible_views(&self) -> Vec<ViewId> {
        self.layout.seats.iter().filter_map(|s| s.active_view()).collect()
    }

    /// Render requests for visible views that are stale; clears their flags.
    pub fn take_render_requests(&mut self) -> Vec<RenderRequest> {
        let visible = self.visible_views();
        let source: Arc<dyn SignalSource> = self.dataset.clone();
        let mut out = Vec::new();
        for v in self.views.iter_mut().filter(|v| v.needs_render && visible.contains(&v.id)) {
            v.needs_render = false;
            out.push(v.render_request(&self.timeline, source.clone(), self.events.clone()));
        }
        out
    }

    // ========================================================================
    // Per-view intents (pointer positions are logical px within the plot)
    // ========================================================================

    pub fn zoom_at(&mut self, id: ViewId, factor: f32, x: f32) {
        let Some(v) = self.view(id) else { return };
        let ratio = (x * v.scale_factor) as f64 / v.canvas_width.max(1) as f64;
        self.timeline.zoom_at(factor as f64, ratio);
        self.mark_all_dirty();
    }

    pub fn pan_pixels(&mut self, id: ViewId, dx: f32) {
        let Some(v) = self.view(id) else { return };
        let dt = -((dx * v.scale_factor) as f64) / v.canvas_width.max(1) as f64 * self.timeline.visible_window_sec;
        self.timeline.pan_time(dt);
        self.mark_all_dirty();
    }

    pub fn zoom_gain(&mut self, id: ViewId, factor: f32) {
        if let Some(v) = self.view_mut(id) {
            v.zoom_gain(factor);
        }
    }

    pub fn scroll_channels(&mut self, id: ViewId, delta: i32) {
        if let Some(v) = self.view_mut(id) {
            v.scroll_by(delta as i64);
        }
    }

    pub fn set_lanes(&mut self, id: ViewId, lanes: usize) {
        if let Some(v) = self.view_mut(id) {
            v.set_lanes(lanes);
        }
    }

    pub fn hover(&mut self, id: ViewId, x: f32, y: f32) {
        let (dataset, timeline) = (self.dataset.clone(), self.timeline.clone());
        if let Some(v) = self.view_mut(id) {
            let s = v.scale_factor;
            v.hover = v.hover_readout(x * s, y * s, &dataset, &timeline);
        }
    }

    /// Double-click: on a heatmap, reveal that channel in a traces view.
    pub fn plot_double_clicked(&mut self, id: ViewId, y: f32) {
        let Some(v) = self.view(id) else { return };
        if v.kind != ViewMode::Heatmap {
            return;
        }
        let Some(ch) = v.channel_at(y * v.scale_factor) else { return };
        let visible = self.visible_views();
        let target = self
            .views
            .iter()
            .filter(|t| t.kind == ViewMode::Traces && t.selection.contains(&ch))
            .max_by_key(|t| visible.contains(&t.id))
            .map(|t| t.id);
        if let Some(t) = target {
            self.view_mut(t).expect("found").reveal(ch);
            self.activate(t);
        }
    }

    // ========================================================================
    // Focused-view intents (keyboard, properties panel, data panel)
    // ========================================================================

    pub fn focused_zoom_gain(&mut self, factor: f32) {
        if let Some(v) = self.focused_view_mut() {
            v.zoom_gain(factor);
        }
    }

    pub fn focused_scroll(&mut self, delta: i32) {
        if let Some(v) = self.focused_view_mut() {
            v.scroll_by(delta as i64);
        }
    }

    pub fn focused_page(&mut self, forward: bool) {
        if let Some(v) = self.focused_view_mut() {
            v.page(forward);
        }
    }

    /// Zoom around the centre of the window (keyboard +/-).
    pub fn zoom_center(&mut self, factor: f32) {
        self.timeline.zoom_at(factor as f64, 0.5);
        self.mark_all_dirty();
    }

    pub fn toggle_channel(&mut self, ch: usize) {
        let total = self.dataset.total_channels;
        if let Some(v) = self.focused_view_mut() {
            v.toggle_channel(ch, total);
        }
    }

    pub fn select_all(&mut self) {
        let all = self.all_channels();
        let total = self.dataset.total_channels;
        if let Some(v) = self.focused_view_mut() {
            v.set_selection(all, total);
        }
    }

    pub fn select_none(&mut self) {
        let total = self.dataset.total_channels;
        if let Some(v) = self.focused_view_mut() {
            v.set_selection([], total);
        }
    }

    pub fn select_invert(&mut self) {
        let total = self.dataset.total_channels;
        if let Some(v) = self.focused_view_mut() {
            let keep: Vec<usize> = (0..total).filter(|c| !v.selection.contains(c)).collect();
            v.set_selection(keep, total);
        }
    }

    /// Applies a range expression to the focused view. Returns an error message to show.
    pub fn select_ranges(&mut self, text: &str) -> Result<(), String> {
        let total = self.dataset.total_channels;
        let channels = parse_channel_ranges(text, total)?;
        if let Some(v) = self.focused_view_mut() {
            v.set_selection(channels, total);
        }
        Ok(())
    }

    /// Channel list rows for the data panel, filtered by `channel_filter`.
    pub fn channel_rows(&self) -> Vec<ChannelRow> {
        let selected = self.focused_view().map(|v| v.selection.as_slice()).unwrap_or(&[]);
        let filter = self.channel_filter.trim();
        (0..self.dataset.total_channels)
            .filter(|c| filter.is_empty() || c.to_string().contains(filter))
            .map(|c| ChannelRow { channel: c, checked: selected.contains(&c), events: self.events.count(c) })
            .collect()
    }

    // ========================================================================
    // Timeline intents (shared by every view)
    // ========================================================================

    pub fn on_toggle_play(&mut self) {
        self.timeline.toggle_play();
        self.last_frame_instant = Instant::now();
    }

    pub fn timeline_changed(&mut self) {
        self.mark_all_dirty();
    }

    pub fn set_window_duration(&mut self, sec: f32) {
        self.timeline.set_window_duration(sec as f64);
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

    /// Spike count per bin across all channels, normalized to [0, 1].
    pub fn overview_density(&self, bins: usize) -> Vec<f32> {
        let dur = self.timeline.total_duration_sec;
        if bins == 0 || dur <= 0.0 {
            return Vec::new();
        }
        let mut counts = vec![0u32; bins];
        for &t in &self.events.times_sec {
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

    // ========================================================================
    // Dataset & session
    // ========================================================================

    /// Replaces the dataset; views keep their layout, selections are clipped to the new
    /// channel count (a view left with nothing selected shows every channel).
    pub fn load_dataset(&mut self, dataset: Dataset, path: Option<PathBuf>) {
        self.events = Arc::new(SpikeEventStore::detect(&dataset));
        self.timeline = TimelineState::new(dataset.total_duration_sec());
        let total = dataset.total_channels;
        self.dataset = Arc::new(dataset);
        for v in &mut self.views {
            let kept: Vec<usize> = v.selection.iter().copied().filter(|&c| c < total).collect();
            let sel = if kept.is_empty() { (0..total).collect() } else { kept };
            v.set_selection(sel, total);
        }
        if let Some(p) = &path {
            self.push_recent(p.clone());
        }
        self.dataset_path = path;
        self.relayout();
    }

    pub fn push_recent(&mut self, path: PathBuf) {
        self.recent.retain(|p| p != &path);
        self.recent.insert(0, path);
        self.recent.truncate(MAX_RECENT);
    }

    pub fn session(&self) -> Session {
        Session {
            version: Session::VERSION,
            dock: self.dock.clone(),
            views: self.views.clone(),
            focused: self.focused,
            panels: self.panels.clone(),
            recent: self.recent.clone(),
            window_sec: self.timeline.visible_window_sec,
        }
    }

    /// Restores recent files always, and the layout if it is valid for the current dataset.
    /// Returns whether the layout was restored.
    pub fn restore(&mut self, session: Session) -> bool {
        self.recent = session.recent.clone();
        self.recent.truncate(MAX_RECENT);
        if !session.is_valid_for(self.dataset.total_channels) {
            return false;
        }
        self.next_id = session.views.iter().map(|v| v.id).max().unwrap_or(0) + 1;
        self.views = session.views;
        for v in &mut self.views {
            v.needs_render = true;
        }
        self.dock = session.dock;
        self.focused = session.focused.filter(|f| self.dock.contains(*f)).or_else(|| self.dock.views().first().copied());
        self.panels = session.panels;
        self.timeline.set_window_duration(session.window_sec);
        self.relayout();
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vm(channels: usize) -> AppViewModel {
        let mut vm = AppViewModel::new(Dataset::generate_synthetic(channels, 10_000.0, 1.0), None);
        vm.workspace_resized(1000.0, 600.0, 1.0);
        vm
    }

    #[test]
    fn test_default_layout_sizes_canvases() {
        let vm = vm(32);
        assert_eq!(vm.views.len(), 2);
        assert_eq!(vm.layout.seats.len(), 2);
        let traces = vm.focused_view().unwrap();
        assert_eq!(traces.kind, ViewMode::Traces);
        // Plot excludes gutter, header, axis
        assert_eq!(traces.canvas_width, 1000 - GUTTER as u32);
        let seat_h = vm.layout.seats[0].height;
        assert_eq!(traces.canvas_height, (seat_h - SEAT_HEADER - AXIS) as u32);
    }

    #[test]
    fn test_add_close_and_render_requests() {
        let mut vm = vm(8);
        let first = vm.focused.unwrap();
        let id = vm.add_view(ViewMode::Traces);
        assert_eq!(vm.focused, Some(id));
        assert_eq!(vm.layout.seats.len(), 3);
        let reqs = vm.take_render_requests();
        assert_eq!(reqs.len(), 3);
        assert!(vm.take_render_requests().is_empty());

        // Timeline changes re-render every visible view
        vm.pan_fraction(0.1);
        assert_eq!(vm.take_render_requests().len(), 3);

        vm.close_view(id);
        assert_eq!(vm.layout.seats.len(), 2);
        assert!(vm.focused.is_some() && vm.focused != Some(id));
        assert!(vm.view(first).is_some());
    }

    #[test]
    fn test_tabbed_views_only_render_active() {
        let mut vm = vm(8);
        let ids = vm.dock.views();
        vm.dock_drop(ids[1], ids[0], DropSide::Centre);
        assert_eq!(vm.layout.seats.len(), 1);
        vm.take_render_requests();
        vm.pan_fraction(0.1);
        let reqs = vm.take_render_requests();
        assert_eq!(reqs.len(), 1);
        assert_eq!(reqs[0].view_id, ids[1]);
    }

    #[test]
    fn test_selection_intents_apply_to_focused_view() {
        let mut vm = vm(16);
        let other = vm.dock.views()[1];
        vm.select_none();
        assert!(vm.focused_view().unwrap().selection.is_empty());
        assert_eq!(vm.view(other).unwrap().selection.len(), 16);
        vm.select_ranges("0-3, 8").unwrap();
        assert_eq!(vm.focused_view().unwrap().selection, vec![0, 1, 2, 3, 8]);
        assert!(vm.select_ranges("99").is_err());
        vm.toggle_channel(8);
        vm.select_invert();
        assert_eq!(vm.focused_view().unwrap().selection.len(), 12);
        vm.channel_filter = "1".into();
        let rows = vm.channel_rows();
        assert_eq!(rows.iter().map(|r| r.channel).collect::<Vec<_>>(), vec![1, 10, 11, 12, 13, 14, 15]);
        assert!(!rows[0].checked && rows[1].checked);
    }

    #[test]
    fn test_divider_drag_and_heatmap_reveal() {
        let mut vm = vm(32);
        let before = vm.layout.seats[0].height;
        vm.divider_pressed(0);
        vm.divider_dragged(0, 50.0);
        assert!((vm.layout.seats[0].height - (before + 50.0)).abs() < 0.5);

        let heat = vm.dock.views()[1];
        let traces = vm.dock.views()[0];
        let v = vm.view(heat).unwrap();
        let row_h = v.canvas_height as f32 / 32.0;
        vm.plot_double_clicked(heat, row_h * 20.5);
        assert_eq!(vm.focused, Some(traces));
        assert!(vm.view(traces).unwrap().drawn_channels().contains(&20));
    }

    #[test]
    fn test_session_round_trip_and_validation() {
        let mut a = vm(16);
        a.add_view(ViewMode::Heatmap);
        a.select_ranges("2-5").unwrap();
        a.push_recent(PathBuf::from("/tmp/x.bin"));
        let json = serde_json::to_string(&a.session()).unwrap();

        let mut b = vm(16);
        assert!(b.restore(serde_json::from_str(&json).unwrap()));
        assert_eq!(b.dock, a.dock);
        assert_eq!(b.focused_view().unwrap().selection, vec![2, 3, 4, 5]);
        assert_eq!(b.recent, vec![PathBuf::from("/tmp/x.bin")]);

        // Too few channels for the saved selection: layout rejected, recents kept
        let mut c = vm(4);
        assert!(!c.restore(serde_json::from_str(&json).unwrap()));
        assert_eq!(c.views.len(), 2);
        assert_eq!(c.recent.len(), 1);
    }
}
