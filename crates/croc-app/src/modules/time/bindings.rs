//! Time module ⇄ Slint: pushes `TimeModule` state into `TimeState`, wires `TimeLogic`.
//!
//! List models are kept alive and updated row by row so repeated elements (a seat being
//! dragged, the channel list scroll position) survive refreshes.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use dsp_core::RecordingSource;
use slint::{ComponentHandle, Image, Model, ModelRc, SharedString, VecModel};

use crate::app::controller::{from_side, rgb, sync_rows, Controller};
use crate::shared::dock::{SeatBox, ViewId};
use crate::ui;

use super::renderer::{render_overview, TimeViewKind, CHANNEL_COLORS};

/// Slint-side models of the Time module.
#[derive(Default)]
pub struct TimeUi {
    pub images: RefCell<HashMap<ViewId, Image>>,
    pub seats: Rc<VecModel<ui::TimeSeat>>,
    pub dividers: Rc<VecModel<ui::DividerData>>,
    pub channels: Rc<VecModel<ui::ChannelRow>>,
    pub overview_size: RefCell<(u32, u32)>,
}

fn to_ui_kind(kind: TimeViewKind) -> ui::TimeViewKind {
    match kind {
        TimeViewKind::Traces => ui::TimeViewKind::Traces,
        TimeViewKind::Heatmap => ui::TimeViewKind::Heatmap,
    }
}

const KINDS: [TimeViewKind; 2] = [TimeViewKind::Traces, TimeViewKind::Heatmap];

impl Controller {
    pub(crate) fn install_time_models(&self, ui: &ui::AppWindow) {
        let state = ui.global::<ui::TimeState>();
        state.set_seats(ModelRc::from(self.time_ui.seats.clone()));
        state.set_dividers(ModelRc::from(self.time_ui.dividers.clone()));
        state.set_channel_rows(ModelRc::from(self.time_ui.channels.clone()));
    }

    /// Pushes everything of the Time module that can change after an intent.
    pub(crate) fn sync_time(&self) {
        self.sync_time_timeline();
        self.sync_time_seats();
        self.sync_time_focused();
        self.sync_time_channels();
        self.sync_time_panels();
    }

    pub(crate) fn sync_time_timeline(&self) {
        let Some(ui) = self.ui.upgrade() else { return };
        let time = self.time.borrow();
        let tl = &time.timeline;
        let state = ui.global::<ui::TimeState>();
        state.set_total_duration_sec(tl.total_duration_sec as f32);
        state.set_current_time_sec(tl.current_time_sec as f32);
        state.set_window_start_sec(tl.window_start_sec as f32);
        state.set_visible_window_sec(tl.visible_window_sec as f32);
        state.set_time_readout(tl.format_time_readout().into());
        state.set_is_playing(tl.is_playing);
        state.set_loop_playback(tl.loop_playback);
        state.set_playback_speed(tl.playback_speed as f32);
    }

    fn time_seat_row(&self, seat: &SeatBox) -> ui::TimeSeat {
        let time = self.time.borrow();
        let tabs: Vec<ui::TabData> = seat
            .views
            .iter()
            .filter_map(|&id| time.ws.view(id))
            .map(|v| ui::TabData { view_id: v.id as i32, title: v.title.clone().into() })
            .collect();
        let mut row = ui::TimeSeat {
            x: seat.x,
            y: seat.y,
            width: seat.width,
            height: seat.height,
            tabs: ModelRc::new(VecModel::from(tabs)),
            active: seat.active as i32,
            view_id: -1,
            empty_message: "Empty workspace — add a view from the View menu (Ctrl+T)".into(),
            ..Default::default()
        };
        let Some(v) = seat.active_view().and_then(|id| time.ws.view(id)) else { return row };
        let lanes: Vec<ui::LaneLabel> = v
            .lane_labels()
            .into_iter()
            .map(|l| ui::LaneLabel { label: l.label.into(), color: rgb(l.color), y_frac: l.y_frac })
            .collect();
        let ticks: Vec<ui::TimeTick> =
            v.time_ticks(&time.timeline).into_iter().map(|t| ui::TimeTick { frac: t.frac, label: t.label.into() }).collect();
        row.view_id = v.id as i32;
        row.kind = to_ui_kind(v.kind);
        row.focused = time.ws.focused == Some(v.id);
        row.image = self.time_ui.images.borrow().get(&v.id).cloned().unwrap_or_default();
        row.lanes = ModelRc::new(VecModel::from(lanes));
        row.ticks = ModelRc::new(VecModel::from(ticks));
        row.scale_bar_label = v.scale_bar_label().into();
        row.scale_bar_frac = v.scale_bar_center_frac();
        row.hover_readout = v.hover.clone().into();
        row.empty_message =
            if v.selection.is_empty() { "No channels selected — pick some in the Channels panel".into() } else { "".into() };
        row
    }

    pub(crate) fn sync_time_seats(&self) {
        let seats = self.time.borrow().ws.layout.seats.clone();
        let rows: Vec<ui::TimeSeat> = seats.iter().map(|s| self.time_seat_row(s)).collect();
        if rows.len() == self.time_ui.seats.row_count() {
            for (i, row) in rows.into_iter().enumerate() {
                self.time_ui.seats.set_row_data(i, row);
            }
        } else {
            self.time_ui.seats.set_vec(rows);
        }
        let dividers = self.time.borrow().ws.layout.dividers.iter().map(crate::app::controller::divider_row).collect();
        sync_rows(&self.time_ui.dividers, dividers);
    }

    fn sync_time_focused(&self) {
        let Some(ui) = self.ui.upgrade() else { return };
        let sources = self.app.borrow().sources.clone();
        let names: Vec<SharedString> = sources.entries().iter().map(|e| e.summary().into()).collect();
        ui.global::<ui::TimeState>().set_source_names(ModelRc::new(VecModel::from(names)));
        let time = self.time.borrow();
        let focused = match time.ws.focused_view() {
            Some(v) => ui::TimeFocused {
                valid: true,
                view_id: v.id as i32,
                title: v.title.clone().into(),
                kind: to_ui_kind(v.kind),
                kind_index: KINDS.iter().position(|&k| k == v.kind).unwrap_or(0) as i32,
                lanes: v.lanes_on_screen() as i32,
                lanes_max: v.selection.len().max(1) as i32,
                gain_label: v.gain_label().into(),
                range_label: v.range_label().into(),
                selected: v.selection.len() as i32,
                source_index: sources.index_of(&v.source) as i32,
                auto_scale: v.auto_scale,
                remove_dc: v.remove_dc,
            },
            None => ui::TimeFocused::default(),
        };
        ui.global::<ui::TimeState>().set_focused(focused);
    }

    fn sync_time_channels(&self) {
        let rows: Vec<ui::ChannelRow> = {
            let app = self.app.borrow();
            let source_id = self.time.borrow().focused_source();
            let ds = app.sources.get(&source_id);
            // Event counts belong to the default source
            let no_events = crate::data::SpikeEventStore::default();
            let events = if app.sources.index_of(&source_id) == app.sources.index_of("") { app.events.as_ref() } else { &no_events };
            let info = ds.info();
            self.time
                .borrow()
                .channel_rows(ds.total_channels, events)
                .into_iter()
                .map(|r| ui::ChannelRow {
                    channel: r.channel as i32,
                    label: info.channels.get(r.channel).map_or_else(|| format!("Ch {}", r.channel), |c| c.name.clone()).into(),
                    color: rgb(CHANNEL_COLORS[r.channel % CHANNEL_COLORS.len()]),
                    checked: r.checked,
                    events: r.events as i32,
                })
                .collect()
        };
        // Same channels in the same order: update in place (keeps the list's scroll position)
        let same = rows.len() == self.time_ui.channels.row_count()
            && rows.iter().enumerate().all(|(i, r)| self.time_ui.channels.row_data(i).is_some_and(|o| o.channel == r.channel));
        if same {
            sync_rows(&self.time_ui.channels, rows);
        } else {
            self.time_ui.channels.set_vec(rows);
        }
    }

    fn sync_time_panels(&self) {
        let Some(ui) = self.ui.upgrade() else { return };
        let time = self.time.borrow();
        let state = ui.global::<ui::TimeState>();
        state.set_show_data(time.panels.data);
        state.set_show_properties(time.panels.properties);
        state.set_show_timeline(time.panels.timeline);
        state.set_show_sorted_spikes(time.show_sorted_spikes);
    }

    pub(crate) fn render_time_overview(&self) {
        let Some(ui) = self.ui.upgrade() else { return };
        let (w, h) = *self.time_ui.overview_size.borrow();
        if w == 0 {
            return;
        }
        let density = self.time.borrow().overview_density(w as usize, &self.app.borrow().events);
        ui.global::<ui::TimeState>().set_overview_image(Image::from_rgba8(render_overview(w, h, &density)));
    }

    /// A time view's frame arrived from the worker.
    pub(crate) fn on_time_frame(&self, view: ViewId, image: Image, scale: Option<f32>) {
        self.time_ui.images.borrow_mut().insert(view, image.clone());
        // The frame was drawn with this amplitude scale: the scale-bar label follows it
        if let Some(s) = scale {
            self.time.borrow_mut().frame_scale(view, s);
        }
        let label: SharedString = self.time.borrow().ws.view(view).map(|v| v.scale_bar_label()).unwrap_or_default().into();
        let frac = self.time.borrow().ws.view(view).map_or(0.0, |v| v.scale_bar_center_frac());
        for i in 0..self.time_ui.seats.row_count() {
            if let Some(mut row) = self.time_ui.seats.row_data(i) {
                if row.view_id == view as i32 {
                    row.image = image.clone();
                    row.scale_bar_label = label.clone();
                    row.scale_bar_frac = frac;
                    self.time_ui.seats.set_row_data(i, row);
                }
            }
        }
    }

    /// Dataset of the focused view's source.
    pub(crate) fn focused_source(&self) -> std::sync::Arc<crate::data::Dataset> {
        let id = self.time.borrow().focused_source();
        let sources = self.app.borrow().sources.clone();
        sources.get(&id)
    }

    pub(crate) fn wire_time(self: &Rc<Self>, ui: &ui::AppWindow) {
        let logic = ui.global::<ui::TimeLogic>();

        // Runs an intent on the Time module, then refreshes the Time UI. The window may have
        // moved, so the spike module's amplitude view is marked stale too.
        macro_rules! intent {
            ($setter:ident, |$m:ident $(, $arg:ident)*| $body:expr) => {{
                let weak = Rc::downgrade(self);
                logic.$setter(move |$($arg),*| {
                    let Some(c) = weak.upgrade() else { return };
                    {
                        #[allow(unused_mut)]
                        let mut $m = c.time.borrow_mut();
                        $body;
                    }
                    c.spikes.borrow_mut().window_changed();
                    c.sync_time();
                });
            }};
        }

        // Timeline
        intent!(on_toggle_play_pause, |m| m.on_toggle_play());
        intent!(on_step_forward, |m| {
            m.timeline.step_forward(0.033);
            m.timeline_changed()
        });
        intent!(on_step_backward, |m| {
            m.timeline.step_backward(0.033);
            m.timeline_changed()
        });
        intent!(on_jump_to_start, |m| {
            m.timeline.scrub_to(0.0);
            m.timeline_changed()
        });
        intent!(on_jump_to_end, |m| {
            let end = m.timeline.total_duration_sec;
            m.timeline.scrub_to(end);
            m.timeline_changed()
        });
        intent!(on_toggle_loop, |m| m.timeline.toggle_loop());
        intent!(on_set_playback_speed, |m, s| m.timeline.playback_speed = s as f64);
        intent!(on_scrub_to_ratio, |m, r| {
            m.timeline.scrub_ratio(r as f64);
            m.timeline_changed()
        });
        intent!(on_pan_overview, |m, r| m.pan_overview(r));
        intent!(on_set_window_edges, |m, a, b| m.set_window_edges(a, b));

        // Keyboard
        intent!(on_pan_fraction, |m, f| m.pan_fraction(f));
        intent!(on_zoom_center, |m, f| m.zoom_center(f));
        intent!(on_focused_gain, |m, f| m.focused_zoom_gain(f));
        intent!(on_focused_scroll, |m, d| m.focused_scroll(d));
        intent!(on_focused_page, |m, fwd| m.focused_page(fwd));

        // Per view
        intent!(on_zoom_at, |m, id, f, x| m.zoom_at(id as ViewId, f, x));
        intent!(on_pan_pixels, |m, id, dx| m.pan_pixels(id as ViewId, dx));
        intent!(on_zoom_gain, |m, id, f| m.zoom_gain(id as ViewId, f));
        intent!(on_scroll_channels, |m, id, d| m.scroll_channels(id as ViewId, d));
        intent!(on_set_lanes, |m, id, n| m.set_lanes(id as ViewId, n.max(1) as usize));
        intent!(on_plot_double_clicked, |m, id, y| m.plot_double_clicked(id as ViewId, y));
        intent!(on_activate_view, |m, id| m.ws.activate(id as ViewId));
        intent!(on_close_view, |m, id| m.ws.close(id as ViewId));
        intent!(on_set_view_kind, |m, id, index| {
            if let Some(&kind) = KINDS.get(index.max(0) as usize) {
                m.set_kind(id as ViewId, kind)
            }
        });
        intent!(on_dock_drop, |m, view, target, side| m.ws.drop_view(view as ViewId, target as ViewId, from_side(side)));
        intent!(on_divider_pressed, |m, i| m.ws.divider_pressed(i as usize));
        intent!(on_divider_dragged, |m, i, delta| m.ws.divider_dragged(i as usize, delta));
        intent!(on_toggle_panel, |m, which| {
            let p = &mut m.panels;
            match which {
                0 => p.data = !p.data,
                1 => p.properties = !p.properties,
                _ => p.timeline = !p.timeline,
            }
        });
        intent!(on_set_show_sorted_spikes, |m, on| {
            m.show_sorted_spikes = on;
            m.mark_traces_dirty()
        });
        intent!(on_set_channel_filter, |m, text| m.channel_filter = text.to_string());

        // Intents that need the shared dataset
        macro_rules! with_data {
            ($setter:ident, |$m:ident, $total:ident $(, $arg:ident)*| $body:expr) => {{
                let weak = Rc::downgrade(self);
                logic.$setter(move |$($arg),*| {
                    let Some(c) = weak.upgrade() else { return };
                    // Channel counts come from the focused view's source
                    let $total = c.focused_source().total_channels;
                    {
                        let mut $m = c.time.borrow_mut();
                        $body;
                    }
                    c.sync_time();
                });
            }};
        }
        {
            let weak = Rc::downgrade(self);
            logic.on_add_view(move |index| {
                let Some(c) = weak.upgrade() else { return };
                let sources = c.app.borrow().sources.clone();
                if let Some(&kind) = KINDS.get(index.max(0) as usize) {
                    c.time.borrow_mut().add_view(kind, &sources);
                }
                c.sync_time();
            });
        }
        {
            let weak = Rc::downgrade(self);
            logic.on_add_view_for_source(move |index| {
                let Some(c) = weak.upgrade() else { return };
                let sources = c.app.borrow().sources.clone();
                c.time.borrow_mut().add_view_for(index.max(0) as usize, &sources);
                c.sync_time();
            });
        }
        {
            let weak = Rc::downgrade(self);
            logic.on_set_view_source(move |index| {
                let Some(c) = weak.upgrade() else { return };
                let sources = c.app.borrow().sources.clone();
                c.time.borrow_mut().set_focused_source(index.max(0) as usize, &sources);
                c.sync_time();
            });
        }
        intent!(on_set_auto_scale, |m, on| m.set_focused_auto_scale(on));
        intent!(on_set_remove_dc, |m, on| m.set_focused_remove_dc(on));
        with_data!(on_toggle_channel, |m, total, ch| m.toggle_channel(ch.max(0) as usize, total));
        with_data!(on_select_all, |m, total| m.select_all(total));
        with_data!(on_select_none, |m, total| m.select_none(total));
        with_data!(on_select_invert, |m, total| m.select_invert(total));
        {
            let weak = Rc::downgrade(self);
            logic.on_select_ranges(move |text| {
                let Some(c) = weak.upgrade() else { return };
                let total = c.focused_source().total_channels;
                let result = c.time.borrow_mut().select_ranges(&text, total);
                if let Some(ui) = c.ui.upgrade() {
                    ui.global::<ui::TimeState>().set_range_error(result.err().unwrap_or_default().into());
                }
                c.sync_time();
            });
        }

        {
            let weak = Rc::downgrade(self);
            logic.on_focus_view(move |id| {
                let Some(c) = weak.upgrade() else { return };
                c.time.borrow_mut().ws.focus(id as ViewId);
                c.sync_time();
                c.request_keyboard_focus();
            });
        }

        // Hover only updates that seat's readout
        {
            let weak = Rc::downgrade(self);
            logic.on_hover_at(move |id, x, y| {
                let Some(c) = weak.upgrade() else { return };
                let sources = c.app.borrow().sources.clone();
                c.time.borrow_mut().hover(id as ViewId, x, y, &sources);
                let text: SharedString = c.time.borrow().ws.view(id as ViewId).map(|v| v.hover.clone()).unwrap_or_default().into();
                for i in 0..c.time_ui.seats.row_count() {
                    if let Some(mut row) = c.time_ui.seats.row_data(i) {
                        if row.view_id == id && row.hover_readout != text {
                            row.hover_readout = text.clone();
                            c.time_ui.seats.set_row_data(i, row);
                        }
                    }
                }
            });
        }

        {
            let weak = Rc::downgrade(self);
            logic.on_workspace_resized(move |w, h| {
                let Some(c) = weak.upgrade() else { return };
                let scale = c.scale();
                c.time.borrow_mut().ws.resized(w, h, scale);
                c.sync_time();
            });
        }

        {
            let weak = Rc::downgrade(self);
            logic.on_overview_resized(move |w, h| {
                let Some(c) = weak.upgrade() else { return };
                let scale = c.scale();
                *c.time_ui.overview_size.borrow_mut() = ((w * scale).max(1.0) as u32, (h * scale).max(1.0) as u32);
                c.render_time_overview();
            });
        }
    }
}
