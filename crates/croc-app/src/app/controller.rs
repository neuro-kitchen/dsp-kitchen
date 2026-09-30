//! Parent app controller: owns the shared model and both child modules, hosts the render
//! worker and frame timer, routes frames to their module, and wires the shell (menus, module
//! tabs, file opening, sorting). Each module's own bindings live in `modules/*/bindings.rs`.

use std::cell::RefCell;
use std::path::PathBuf;
use std::rc::Rc;
use std::time::Duration;

use slint::{Color, ComponentHandle, Image, Model, SharedString, Timer, TimerMode, VecModel};

use crate::data::Dataset;
use crate::modules::spikes::bindings::SpikeUi;
use crate::modules::spikes::module::{SpikeModule, MODULE_ID as SPIKES};
use crate::modules::spikes::sorting::{SortParams, Sorting};
use crate::modules::time::bindings::TimeUi;
use crate::modules::time::module::{TimeModule, MODULE_ID as TIME};
use crate::shared::dock::{DividerBox, DropSide};
use crate::shared::render_worker::{Frame, FrameInfo, RenderWorker};
use crate::shared::workspace::{AXIS, GUTTER, SEAT_HEADER};
use crate::ui;

use super::model::{AppModel, Module, SortStatus};
use super::session::Session;

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

pub(crate) fn rgb(c: slint::Rgba8Pixel) -> Color {
    Color::from_rgb_u8(c.r, c.g, c.b)
}

pub(crate) fn from_side(side: ui::DropSide) -> DropSide {
    match side {
        ui::DropSide::Left => DropSide::Left,
        ui::DropSide::Right => DropSide::Right,
        ui::DropSide::Top => DropSide::Top,
        ui::DropSide::Bottom => DropSide::Bottom,
        _ => DropSide::Centre,
    }
}

pub(crate) fn divider_row(d: &DividerBox) -> ui::DividerData {
    ui::DividerData { index: d.index as i32, columns: d.columns, x: d.x, y: d.y, width: d.width, height: d.height }
}

/// Updates `model` to `rows`: row by row when the length matches (repeated elements keep
/// their state), otherwise replaced.
pub(crate) fn sync_rows<T: Clone + PartialEq + 'static>(model: &VecModel<T>, rows: Vec<T>) {
    if rows.len() == model.row_count() {
        for (i, r) in rows.into_iter().enumerate() {
            if model.row_data(i).as_ref() != Some(&r) {
                model.set_row_data(i, r);
            }
        }
    } else {
        model.set_vec(rows);
    }
}

const DRAG_PREFIX: &str = "croc-view:";

pub struct Controller {
    pub(crate) ui: slint::Weak<ui::AppWindow>,
    pub(crate) app: RefCell<AppModel>,
    pub(crate) time: RefCell<TimeModule>,
    pub(crate) spikes: RefCell<SpikeModule>,
    pub(crate) time_ui: TimeUi,
    pub(crate) spike_ui: SpikeUi,
    worker: RenderWorker,
    recent: Rc<VecModel<SharedString>>,
    timer: Timer,
    save_on_exit: bool,
}

impl Controller {
    /// Builds the controller, restores the session, wires every callback and starts the timer.
    pub fn install(ui: &ui::AppWindow, app: AppModel, restore: bool, save_on_exit: bool) -> Rc<Controller> {
        let channels = app.dataset.total_channels;
        let mut time = TimeModule::new(&app.dataset);
        let mut spikes = SpikeModule::new(channels);
        let mut app = app;
        if let Some(session) = Session::load().filter(|_| restore) {
            app.recent = session.recent;
            app.active = session.active;
            if TimeModule::is_valid(&session.time, channels) {
                time.restore(session.time);
            }
            spikes.restore(session.spikes);
        }

        let worker = RenderWorker::spawn(|frame, info| {
            let _ = slint::invoke_from_event_loop(move || with_controller(|c| c.on_frame(frame, info)));
        });

        let c = Rc::new(Controller {
            ui: ui.as_weak(),
            app: RefCell::new(app),
            time: RefCell::new(time),
            spikes: RefCell::new(spikes),
            time_ui: TimeUi::default(),
            spike_ui: SpikeUi::default(),
            worker,
            recent: Rc::new(VecModel::default()),
            timer: Timer::default(),
            save_on_exit,
        });
        CONTROLLER.with(|slot| *slot.borrow_mut() = Some(c.clone()));

        let metrics = ui.global::<ui::Metrics>();
        metrics.set_seat_header(SEAT_HEADER);
        metrics.set_gutter(GUTTER);
        metrics.set_axis(AXIS);
        ui.global::<ui::AppState>().set_recent_files(slint::ModelRc::from(c.recent.clone()));
        c.install_time_models(ui);
        c.install_spike_models(ui);

        c.wire_shell(ui);
        c.wire_time(ui);
        c.wire_spikes(ui);
        c.sync_dataset();
        c.refresh();
        c.start_sorting();

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
        let session = Session {
            version: Session::VERSION,
            active: self.app.borrow().active,
            time: self.time.borrow().session(),
            spikes: self.spikes.borrow().session(),
            recent: self.app.borrow().recent.clone(),
        };
        if let Err(e) = session.save() {
            tracing::warn!("could not save session: {e}");
        }
    }

    pub(crate) fn scale(&self) -> f32 {
        self.ui.upgrade().map_or(1.0, |ui| ui.window().scale_factor())
    }

    /// Returns keyboard focus to the window's shortcut scope.
    pub(crate) fn request_keyboard_focus(&self) {
        if let Some(ui) = self.ui.upgrade() {
            let state = ui.global::<ui::AppState>();
            state.set_focus_request(state.get_focus_request() + 1);
        }
    }

    // ------------------------------------------------------------------------
    // Frame loop
    // ------------------------------------------------------------------------

    fn tick(&self) {
        // Playback runs whichever module is shown; only the visible module renders
        if self.time.borrow_mut().on_tick() {
            self.spikes.borrow_mut().window_changed();
            self.sync_time_timeline();
            self.sync_time_seats();
        }
        let jobs = match self.app.borrow().active {
            Module::Time => {
                let app = self.app.borrow();
                let marks = app.spike_marks(self.time.borrow().window());
                self.time.borrow_mut().take_jobs(&app.dataset, &app.events, &marks)
            }
            Module::Spikes => {
                let mut spikes = self.spikes.borrow_mut();
                self.with_plot_context(|ctx| spikes.take_jobs(ctx))
            }
        };
        for job in jobs {
            self.worker.request(job);
        }
    }

    fn on_frame(&self, frame: Frame, info: FrameInfo) {
        let image = Image::from_rgba8(frame);
        let (module, view) = info.key;
        match module {
            TIME => self.on_time_frame(view, image),
            SPIKES => self.on_spike_frame(view, image),
            _ => return,
        }
        let ms = info.elapsed.as_secs_f64() * 1000.0;
        let status = match info.samples_per_px {
            Some(p) => format!("Render {ms:.1} ms · {p:.1} samples/px"),
            None => format!("Render {ms:.1} ms"),
        };
        if let Some(ui) = self.ui.upgrade() {
            ui.global::<ui::AppState>().set_status_message(status.into());
        }
    }

    // ------------------------------------------------------------------------
    // State → UI
    // ------------------------------------------------------------------------

    /// Pushes the shell and both modules.
    fn refresh(&self) {
        self.sync_shell();
        self.sync_time();
        self.sync_spikes();
    }

    fn sync_shell(&self) {
        let Some(ui) = self.ui.upgrade() else { return };
        let app = self.app.borrow();
        let state = ui.global::<ui::AppState>();
        state.set_active_module(app.active.index() as i32);
        state.set_sort_status(app.sort_status_text().into());
        state.set_sort_running(app.sort_status == SortStatus::Running);
    }

    fn sync_dataset(&self) {
        let Some(ui) = self.ui.upgrade() else { return };
        {
            let app = self.app.borrow();
            let ds = &app.dataset;
            let state = ui.global::<ui::AppState>();
            state.set_dataset_name(ds.name.clone().into());
            state.set_dataset_info(
                format!(
                    "{} ch · {:.1} kHz · {} · {}",
                    ds.total_channels,
                    ds.sample_rate / 1000.0,
                    format_duration(ds.total_duration_sec()),
                    if app.events_detected { format!("{} events", app.events.len()) } else { "events not detected (long recording)".into() }
                )
                .into(),
            );
            state.set_channel_count(ds.total_channels as i32);
            state.set_sample_rate(ds.sample_rate as f32);
            state.set_total_spikes_detected(app.events.len() as i32);
            self.recent.set_vec(app.recent.iter().map(|p| p.display().to_string().into()).collect::<Vec<SharedString>>());
        }
        self.time_ui.images.borrow_mut().clear();
        self.spike_ui.images.borrow_mut().clear();
        self.render_time_overview();
    }

    // ------------------------------------------------------------------------
    // Shared data: dataset and sorting
    // ------------------------------------------------------------------------

    fn open_path(&self, path: PathBuf) {
        match Dataset::open(&path) {
            Ok(ds) => {
                self.app.borrow_mut().load_dataset(ds, Some(path));
                let (ds, channels) = {
                    let app = self.app.borrow();
                    (app.dataset.clone(), app.dataset.total_channels)
                };
                self.time.borrow_mut().dataset_changed(&ds);
                {
                    let mut spikes = self.spikes.borrow_mut();
                    spikes.sort_params.num_clusters = SortParams::for_channels(channels).num_clusters;
                    spikes.mark_all_dirty();
                }
                self.sync_dataset();
                self.refresh();
                self.save_session();
                self.start_sorting();
            }
            Err(e) => {
                if let Some(ui) = self.ui.upgrade() {
                    ui.global::<ui::AppState>().set_status_message(format!("Could not open: {e:#}").into());
                }
            }
        }
    }

    /// Runs the spike sorting on a background thread; results come back to the UI thread.
    fn start_sorting(&self) {
        let Some((dataset, generation)) = self.app.borrow_mut().begin_sorting() else { return };
        let params = self.spikes.borrow().sort_params.clone();
        self.sync_shell();
        std::thread::Builder::new()
            .name("croc-sort".into())
            .spawn(move || {
                let t0 = std::time::Instant::now();
                let sorting = Sorting::run(&dataset, &params);
                let millis = t0.elapsed().as_millis();
                let _ = slint::invoke_from_event_loop(move || {
                    with_controller(|c| {
                        if c.app.borrow_mut().finish_sorting(generation, sorting, millis) {
                            c.selection_changed();
                        }
                        c.refresh();
                    });
                });
            })
            .expect("failed to spawn sorting thread");
    }

    // ------------------------------------------------------------------------
    // Shell intents
    // ------------------------------------------------------------------------

    fn wire_shell(self: &Rc<Self>, ui: &ui::AppWindow) {
        let logic = ui.global::<ui::AppLogic>();

        {
            let weak = Rc::downgrade(self);
            logic.on_set_module(move |i| {
                let Some(c) = weak.upgrade() else { return };
                let Some(&module) = Module::ALL.get(i.max(0) as usize) else { return };
                c.app.borrow_mut().active = module;
                c.refresh();
                c.request_keyboard_focus();
            });
        }
        {
            let weak = Rc::downgrade(self);
            logic.on_run_sorting(move || {
                if let Some(c) = weak.upgrade() {
                    c.start_sorting();
                }
            });
        }
        {
            let weak = Rc::downgrade(self);
            logic.on_reset_layout(move || {
                let Some(c) = weak.upgrade() else { return };
                let channels = c.app.borrow().dataset.total_channels;
                match c.app.borrow().active {
                    Module::Time => c.time.borrow_mut().reset_layout(channels),
                    Module::Spikes => c.spikes.borrow_mut().reset_layout(),
                }
                c.refresh();
            });
        }
        {
            let weak = Rc::downgrade(self);
            logic.on_compact_layout(move || {
                let Some(c) = weak.upgrade() else { return };
                match c.app.borrow().active {
                    Module::Time => c.time.borrow_mut().ws.compact(),
                    Module::Spikes => c.spikes.borrow_mut().ws.compact(),
                }
                c.refresh();
            });
        }

        // Drag-and-drop payload: module and view id as plain text
        logic.on_view_to_transfer(|module, id| SharedString::from(format!("{DRAG_PREFIX}{module}:{id}")).into());
        logic.on_transfer_to_view(|module, data| {
            data.plain_text()
                .ok()
                .and_then(|t| {
                    let (m, id) = t.strip_prefix(DRAG_PREFIX)?.split_once(':')?;
                    (m.parse::<i32>().ok()? == module).then(|| id.parse::<i32>().ok()).flatten()
                })
                .unwrap_or(-1)
        });

        {
            let weak = Rc::downgrade(self);
            logic.on_open_recent(move |i| {
                let Some(c) = weak.upgrade() else { return };
                let path = c.app.borrow().recent.get(i.max(0) as usize).cloned();
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

/// `9.87 s`, `12:34` (m:ss) or `1:42:47` (h:mm:ss).
fn format_duration(sec: f64) -> String {
    if sec < 60.0 {
        return format!("{sec:.2} s");
    }
    let total = sec.round() as u64;
    let (h, m, s) = (total / 3600, total / 60 % 60, total % 60);
    if h > 0 { format!("{h}:{m:02}:{s:02}") } else { format!("{m}:{s:02}") }
}
