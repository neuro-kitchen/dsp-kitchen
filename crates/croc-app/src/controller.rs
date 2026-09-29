//! Binds the ViewModel to the Slint window: pushes state into `AppState`, wires `AppLogic`
//! intents, runs the frame timer, and receives frames from the render worker.
//!
//! List models (seats, dividers, channels) are kept alive and updated row by row, so the
//! repeated Slint elements survive refreshes (a seat being dragged is never re-created).

use std::cell::RefCell;
use std::collections::HashMap;
use std::path::PathBuf;
use std::rc::Rc;
use std::time::Duration;

use slint::{Color, ComponentHandle, Image, Model, ModelRc, SharedString, Timer, TimerMode, VecModel};

use crate::model::Dataset;
use crate::view::{render_overview, RenderWorker, ViewMode, CHANNEL_COLORS};
use crate::viewmodel::{self, AppViewModel, Session, ViewId, AXIS, GUTTER, SEAT_HEADER};
use crate::{
    AppLogic, AppState, AppWindow, ChannelRow, DividerData, DropSide, FocusedView, LaneLabel, Metrics,
    SeatData, TabData, TimeTick, ViewKind,
};

thread_local! {
    static CONTROLLER: RefCell<Option<Rc<Controller>>> = const { RefCell::new(None) };
}

/// Runs `f` with the controller (UI thread only; a no-op before startup or after shutdown).
fn with_controller(f: impl FnOnce(&Rc<Controller>)) {
    let c = CONTROLLER.with(|c| c.borrow().clone());
    if let Some(c) = c {
        f(&c);
    }
}

pub struct Controller {
    ui: slint::Weak<AppWindow>,
    vm: RefCell<AppViewModel>,
    worker: RenderWorker,
    images: RefCell<HashMap<ViewId, Image>>,
    seats: Rc<VecModel<SeatData>>,
    dividers: Rc<VecModel<DividerData>>,
    channels: Rc<VecModel<ChannelRow>>,
    recent: Rc<VecModel<SharedString>>,
    overview_size: RefCell<(u32, u32)>,
    timer: Timer,
    save_on_exit: bool,
}

fn to_kind(kind: ViewMode) -> ViewKind {
    match kind {
        ViewMode::Traces => ViewKind::Traces,
        ViewMode::Heatmap => ViewKind::Heatmap,
    }
}

fn from_kind(kind: ViewKind) -> ViewMode {
    match kind {
        ViewKind::Heatmap => ViewMode::Heatmap,
        _ => ViewMode::Traces,
    }
}

fn from_side(side: DropSide) -> viewmodel::DropSide {
    match side {
        DropSide::Left => viewmodel::DropSide::Left,
        DropSide::Right => viewmodel::DropSide::Right,
        DropSide::Top => viewmodel::DropSide::Top,
        DropSide::Bottom => viewmodel::DropSide::Bottom,
        _ => viewmodel::DropSide::Centre,
    }
}

fn rgb(c: slint::Rgba8Pixel) -> Color {
    Color::from_rgb_u8(c.r, c.g, c.b)
}

const DRAG_PREFIX: &str = "croc-view:";

impl Controller {
    /// Builds the controller, restores the session, wires every callback and starts the timer.
    pub fn install(ui: &AppWindow, mut vm: AppViewModel, restore: bool, save_on_exit: bool) -> Rc<Controller> {
        if let Some(session) = Session::load().filter(|_| restore) {
            vm.restore(session);
        }

        let worker = RenderWorker::spawn(|frame, elapsed, req| {
            let (view, per_px) = (req.view_id, req.window_sec * req.source.sample_rate() / req.width.max(1) as f64);
            let _ = slint::invoke_from_event_loop(move || {
                with_controller(|c| c.on_frame(view, frame, elapsed, per_px));
            });
        });

        let c = Rc::new(Controller {
            ui: ui.as_weak(),
            vm: RefCell::new(vm),
            worker,
            images: RefCell::new(HashMap::new()),
            seats: Rc::new(VecModel::default()),
            dividers: Rc::new(VecModel::default()),
            channels: Rc::new(VecModel::default()),
            recent: Rc::new(VecModel::default()),
            overview_size: RefCell::new((0, 0)),
            timer: Timer::default(),
            save_on_exit,
        });
        CONTROLLER.with(|slot| *slot.borrow_mut() = Some(c.clone()));

        let metrics = ui.global::<Metrics>();
        metrics.set_seat_header(SEAT_HEADER);
        metrics.set_gutter(GUTTER);
        metrics.set_axis(AXIS);

        let state = ui.global::<AppState>();
        state.set_seats(ModelRc::from(c.seats.clone()));
        state.set_dividers(ModelRc::from(c.dividers.clone()));
        state.set_channel_rows(ModelRc::from(c.channels.clone()));
        state.set_recent_files(ModelRc::from(c.recent.clone()));

        c.wire(ui);
        c.sync_dataset();
        c.refresh();

        c.timer.start(TimerMode::Repeated, Duration::from_millis(16), || {
            with_controller(|c| c.tick());
        });
        c
    }

    /// Saves the session and drops the controller (call after the event loop ends).
    pub fn shutdown() {
        with_controller(|c| c.save_session());
        CONTROLLER.with(|slot| slot.borrow_mut().take());
    }

    fn save_session(&self) {
        if !self.save_on_exit {
            return;
        }
        if let Err(e) = self.vm.borrow().session().save() {
            tracing::warn!("could not save session: {e}");
        }
    }

    // ------------------------------------------------------------------------
    // Frame loop
    // ------------------------------------------------------------------------

    fn tick(&self) {
        let moved = self.vm.borrow_mut().on_tick();
        if moved {
            self.sync_timeline();
            self.sync_seats();
        }
        let requests = self.vm.borrow_mut().take_render_requests();
        for req in requests {
            self.worker.request(req);
        }
    }

    fn on_frame(&self, view: ViewId, frame: slint::SharedPixelBuffer<slint::Rgba8Pixel>, elapsed: Duration, per_px: f64) {
        let image = Image::from_rgba8(frame);
        self.images.borrow_mut().insert(view, image.clone());
        for i in 0..self.seats.row_count() {
            if let Some(mut row) = self.seats.row_data(i) {
                if row.view_id == view as i32 {
                    row.image = image.clone();
                    self.seats.set_row_data(i, row);
                }
            }
        }
        if let Some(ui) = self.ui.upgrade() {
            if self.vm.borrow().focused == Some(view) {
                let status = format!("Render {:.1} ms  ·  {per_px:.1} samples/px", elapsed.as_secs_f64() * 1000.0);
                ui.global::<AppState>().set_status_message(status.into());
            }
        }
    }

    // ------------------------------------------------------------------------
    // State → UI
    // ------------------------------------------------------------------------

    /// Pushes everything that can change after an intent.
    fn refresh(&self) {
        self.sync_timeline();
        self.sync_seats();
        self.sync_focused();
        self.sync_channels();
        self.sync_panels();
    }

    fn sync_dataset(&self) {
        let Some(ui) = self.ui.upgrade() else { return };
        let vm = self.vm.borrow();
        let ds = &vm.dataset;
        let state = ui.global::<AppState>();
        state.set_dataset_name(ds.name.clone().into());
        state.set_dataset_info(
            format!(
                "{} ch · {:.1} kHz · {:.2} s · {} events",
                ds.total_channels,
                ds.sample_rate / 1000.0,
                ds.total_duration_sec(),
                vm.events.len()
            )
            .into(),
        );
        state.set_channel_count(ds.total_channels as i32);
        state.set_sample_rate(ds.sample_rate as f32);
        state.set_total_duration_sec(vm.timeline.total_duration_sec as f32);
        state.set_total_spikes_detected(vm.events.len() as i32);
        let recent: Vec<SharedString> = vm.recent.iter().map(|p| p.display().to_string().into()).collect();
        self.recent.set_vec(recent);
        drop(vm);
        self.images.borrow_mut().clear();
        let (w, h) = *self.overview_size.borrow();
        if w > 0 {
            self.render_overview(w, h);
        }
    }

    fn sync_timeline(&self) {
        let Some(ui) = self.ui.upgrade() else { return };
        let vm = self.vm.borrow();
        let tl = &vm.timeline;
        let state = ui.global::<AppState>();
        state.set_current_time_sec(tl.current_time_sec as f32);
        state.set_window_start_sec(tl.window_start_sec as f32);
        state.set_visible_window_sec(tl.visible_window_sec as f32);
        state.set_time_readout(tl.format_time_readout().into());
        state.set_is_playing(tl.is_playing);
        state.set_loop_playback(tl.loop_playback);
        state.set_playback_speed(tl.playback_speed as f32);
    }

    fn seat_row(&self, vm: &AppViewModel, seat: &viewmodel::dock::SeatBox) -> SeatData {
        let tabs: Vec<TabData> = seat
            .views
            .iter()
            .filter_map(|&id| vm.view(id))
            .map(|v| TabData { view_id: v.id as i32, title: v.title.clone().into(), kind: to_kind(v.kind) })
            .collect();
        let mut row = SeatData {
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
        if let Some(v) = seat.active_view().and_then(|id| vm.view(id)) {
            let lanes: Vec<LaneLabel> = v
                .lane_labels()
                .into_iter()
                .map(|l| LaneLabel { label: l.label.into(), color: rgb(l.color), y_frac: l.y_frac })
                .collect();
            let ticks: Vec<TimeTick> = v
                .time_ticks(&vm.timeline)
                .into_iter()
                .map(|t| TimeTick { frac: t.frac, label: t.label.into() })
                .collect();
            let bar = v.scale_bar_uv();
            row.view_id = v.id as i32;
            row.kind = to_kind(v.kind);
            row.focused = vm.focused == Some(v.id);
            row.image = self.images.borrow().get(&v.id).cloned().unwrap_or_default();
            row.lanes = ModelRc::new(VecModel::from(lanes));
            row.ticks = ModelRc::new(VecModel::from(ticks));
            row.scale_bar_label = if bar > 0.0 { format!("{bar} µV").into() } else { "".into() };
            row.scale_bar_frac = v.scale_bar_center_frac();
            row.hover_readout = v.hover.clone().into();
            row.empty_message = if v.selection.is_empty() {
                "No channels selected — pick some in the Data panel".into()
            } else {
                "".into()
            };
        }
        row
    }

    fn sync_seats(&self) {
        let rows: Vec<SeatData> = {
            let vm = self.vm.borrow();
            vm.layout.seats.iter().map(|s| self.seat_row(&vm, s)).collect()
        };
        if rows.len() == self.seats.row_count() {
            for (i, row) in rows.into_iter().enumerate() {
                self.seats.set_row_data(i, row);
            }
        } else {
            self.seats.set_vec(rows);
        }

        let dividers: Vec<DividerData> = self
            .vm
            .borrow()
            .layout
            .dividers
            .iter()
            .map(|d| DividerData {
                index: d.index as i32,
                columns: d.columns,
                x: d.x,
                y: d.y,
                width: d.width,
                height: d.height,
            })
            .collect();
        if dividers.len() == self.dividers.row_count() {
            for (i, d) in dividers.into_iter().enumerate() {
                if self.dividers.row_data(i).as_ref() != Some(&d) {
                    self.dividers.set_row_data(i, d);
                }
            }
        } else {
            self.dividers.set_vec(dividers);
        }
        self.vm.borrow_mut().layout_dirty = false;
    }

    fn sync_focused(&self) {
        let Some(ui) = self.ui.upgrade() else { return };
        let vm = self.vm.borrow();
        let focused = match vm.focused_view() {
            Some(v) => FocusedView {
                valid: true,
                view_id: v.id as i32,
                title: v.title.clone().into(),
                kind: to_kind(v.kind),
                lanes: v.lanes_on_screen() as i32,
                lanes_max: v.selection.len().max(1) as i32,
                gain_label: v.gain_label().into(),
                range_label: v.range_label().into(),
                selected: v.selection.len() as i32,
            },
            None => FocusedView::default(),
        };
        ui.global::<AppState>().set_focused(focused);
    }

    fn sync_channels(&self) {
        let rows: Vec<ChannelRow> = self
            .vm
            .borrow()
            .channel_rows()
            .into_iter()
            .map(|r| ChannelRow {
                channel: r.channel as i32,
                label: format!("Ch {}", r.channel).into(),
                color: rgb(CHANNEL_COLORS[r.channel % CHANNEL_COLORS.len()]),
                checked: r.checked,
                events: r.events as i32,
            })
            .collect();
        let same_rows = rows.len() == self.channels.row_count()
            && rows.iter().enumerate().all(|(i, r)| self.channels.row_data(i).is_some_and(|o| o.channel == r.channel));
        if same_rows {
            for (i, r) in rows.into_iter().enumerate() {
                if self.channels.row_data(i).as_ref() != Some(&r) {
                    self.channels.set_row_data(i, r);
                }
            }
        } else {
            self.channels.set_vec(rows);
        }
    }

    fn sync_panels(&self) {
        let Some(ui) = self.ui.upgrade() else { return };
        let p = self.vm.borrow().panels.clone();
        let state = ui.global::<AppState>();
        state.set_show_data(p.data);
        state.set_show_properties(p.properties);
        state.set_show_timeline(p.timeline);
    }

    fn render_overview(&self, w: u32, h: u32) {
        let Some(ui) = self.ui.upgrade() else { return };
        let density = self.vm.borrow().overview_density(w as usize);
        ui.global::<AppState>().set_overview_image(Image::from_rgba8(render_overview(w, h, &density)));
    }

    fn scale(&self) -> f32 {
        self.ui.upgrade().map_or(1.0, |ui| ui.window().scale_factor())
    }

    // ------------------------------------------------------------------------
    // Dataset loading
    // ------------------------------------------------------------------------

    fn open_path(&self, path: PathBuf) {
        match Dataset::load_from_file(&path, None, None) {
            Ok(ds) => {
                self.vm.borrow_mut().load_dataset(ds, Some(path));
                self.sync_dataset();
                self.refresh();
                self.save_session();
            }
            Err(e) => {
                if let Some(ui) = self.ui.upgrade() {
                    ui.global::<AppState>().set_status_message(format!("Could not open: {e:#}").into());
                }
            }
        }
    }

    // ------------------------------------------------------------------------
    // UI → intents
    // ------------------------------------------------------------------------

    fn wire(self: &Rc<Self>, ui: &AppWindow) {
        let logic = ui.global::<AppLogic>();

        // Runs an intent on the ViewModel, then refreshes the whole view state
        macro_rules! intent {
            ($setter:ident, |$vm:ident $(, $arg:ident)*| $body:expr) => {{
                let weak = Rc::downgrade(self);
                logic.$setter(move |$($arg),*| {
                    let Some(c) = weak.upgrade() else { return };
                    {
                        #[allow(unused_mut)]
                        let mut $vm = c.vm.borrow_mut();
                        $body;
                    }
                    c.refresh();
                });
            }};
        }

        // Timeline
        intent!(on_toggle_play_pause, |vm| vm.on_toggle_play());
        intent!(on_step_forward, |vm| {
            vm.timeline.step_forward(0.033);
            vm.timeline_changed()
        });
        intent!(on_step_backward, |vm| {
            vm.timeline.step_backward(0.033);
            vm.timeline_changed()
        });
        intent!(on_jump_to_start, |vm| {
            vm.timeline.scrub_to(0.0);
            vm.timeline_changed()
        });
        intent!(on_jump_to_end, |vm| {
            let end = vm.timeline.total_duration_sec;
            vm.timeline.scrub_to(end);
            vm.timeline_changed()
        });
        intent!(on_toggle_loop, |vm| vm.timeline.toggle_loop());
        intent!(on_set_playback_speed, |vm, s| vm.timeline.playback_speed = s as f64);
        intent!(on_set_window_duration, |vm, d| vm.set_window_duration(d));
        intent!(on_scrub_to_ratio, |vm, r| {
            vm.timeline.scrub_ratio(r as f64);
            vm.timeline_changed()
        });
        intent!(on_pan_overview, |vm, r| vm.pan_overview(r));
        intent!(on_set_window_edges, |vm, a, b| vm.set_window_edges(a, b));

        // Keyboard
        intent!(on_pan_fraction, |vm, f| vm.pan_fraction(f));
        intent!(on_zoom_center, |vm, f| vm.zoom_center(f));
        intent!(on_focused_gain, |vm, f| vm.focused_zoom_gain(f));
        intent!(on_focused_scroll, |vm, d| vm.focused_scroll(d));
        intent!(on_focused_page, |vm, fwd| vm.focused_page(fwd));

        // Per view
        intent!(on_zoom_at, |vm, id, f, x| vm.zoom_at(id as ViewId, f, x));
        intent!(on_pan_pixels, |vm, id, dx| vm.pan_pixels(id as ViewId, dx));
        intent!(on_zoom_gain, |vm, id, f| vm.zoom_gain(id as ViewId, f));
        intent!(on_scroll_channels, |vm, id, d| vm.scroll_channels(id as ViewId, d));
        intent!(on_set_lanes, |vm, id, n| vm.set_lanes(id as ViewId, n.max(1) as usize));
        intent!(on_plot_double_clicked, |vm, id, y| vm.plot_double_clicked(id as ViewId, y));
        intent!(on_activate_view, |vm, id| vm.activate(id as ViewId));
        intent!(on_close_view, |vm, id| vm.close_view(id as ViewId));
        intent!(on_set_view_kind, |vm, id, kind| vm.set_kind(id as ViewId, from_kind(kind)));
        intent!(on_dock_drop, |vm, view, target, side| vm.dock_drop(view as ViewId, target as ViewId, from_side(side)));
        intent!(on_divider_pressed, |vm, i| vm.divider_pressed(i as usize));
        intent!(on_divider_dragged, |vm, i, delta| vm.divider_dragged(i as usize, delta));

        {
            let weak = Rc::downgrade(self);
            logic.on_focus_view(move |id| {
                let Some(c) = weak.upgrade() else { return };
                c.vm.borrow_mut().focus(id as ViewId);
                c.refresh();
                if let Some(ui) = c.ui.upgrade() {
                    let state = ui.global::<AppState>();
                    state.set_focus_request(state.get_focus_request() + 1);
                }
            });
        }

        // Hover only updates that seat's readout
        {
            let weak = Rc::downgrade(self);
            logic.on_hover_at(move |id, x, y| {
                let Some(c) = weak.upgrade() else { return };
                c.vm.borrow_mut().hover(id as ViewId, x, y);
                let text: SharedString = c.vm.borrow().view(id as ViewId).map(|v| v.hover.clone()).unwrap_or_default().into();
                for i in 0..c.seats.row_count() {
                    if let Some(mut row) = c.seats.row_data(i) {
                        if row.view_id == id && row.hover_readout != text {
                            row.hover_readout = text.clone();
                            c.seats.set_row_data(i, row);
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
                c.vm.borrow_mut().workspace_resized(w, h, scale);
                c.refresh();
            });
        }

        {
            let weak = Rc::downgrade(self);
            logic.on_overview_resized(move |w, h| {
                let Some(c) = weak.upgrade() else { return };
                let scale = c.scale();
                let size = ((w * scale).max(1.0) as u32, (h * scale).max(1.0) as u32);
                *c.overview_size.borrow_mut() = size;
                c.render_overview(size.0, size.1);
            });
        }

        // Drag-and-drop payload: a view id as plain text
        logic.on_view_to_transfer(|id| SharedString::from(format!("{DRAG_PREFIX}{id}")).into());
        logic.on_transfer_to_view(|data| {
            data.plain_text()
                .ok()
                .and_then(|t| t.strip_prefix(DRAG_PREFIX).and_then(|n| n.parse::<i32>().ok()))
                .unwrap_or(-1)
        });

        // Data panel
        intent!(on_toggle_channel, |vm, ch| vm.toggle_channel(ch.max(0) as usize));
        intent!(on_select_all, |vm| vm.select_all());
        intent!(on_select_none, |vm| vm.select_none());
        intent!(on_select_invert, |vm| vm.select_invert());
        intent!(on_set_channel_filter, |vm, text| vm.channel_filter = text.to_string());
        {
            let weak = Rc::downgrade(self);
            logic.on_select_ranges(move |text| {
                let Some(c) = weak.upgrade() else { return };
                let result = c.vm.borrow_mut().select_ranges(&text);
                if let Some(ui) = c.ui.upgrade() {
                    ui.global::<AppState>().set_range_error(result.err().unwrap_or_default().into());
                }
                c.refresh();
            });
        }

        // Menu
        intent!(on_add_view, |vm, kind| vm.add_view(from_kind(kind)));
        intent!(on_reset_layout, |vm| vm.reset_layout());
        intent!(on_compact_layout, |vm| vm.compact_layout());
        intent!(on_toggle_panel, |vm, which| {
            let p = &mut vm.panels;
            match which {
                0 => p.data = !p.data,
                1 => p.properties = !p.properties,
                _ => p.timeline = !p.timeline,
            }
        });
        {
            let weak = Rc::downgrade(self);
            logic.on_open_recent(move |i| {
                let Some(c) = weak.upgrade() else { return };
                let path = c.vm.borrow().recent.get(i.max(0) as usize).cloned();
                if let Some(p) = path {
                    c.open_path(p);
                }
            });
        }
        logic.on_open_file(|| {
            // The portal dialog blocks: run it off the UI thread and load on return
            std::thread::spawn(|| {
                let picked = rfd::FileDialog::new()
                    .set_title("Open recording")
                    .add_filter("Raw float32 recording", &["bin"])
                    .add_filter("All files", &["*"])
                    .pick_file();
                if let Some(path) = picked {
                    let _ = slint::invoke_from_event_loop(move || with_controller(|c| c.open_path(path)));
                }
            });
        });
        {
            let weak = Rc::downgrade(self);
            logic.on_quit(move || {
                if let Some(c) = weak.upgrade() {
                    c.save_session();
                }
                let _ = slint::quit_event_loop();
            });
        }
    }
}
