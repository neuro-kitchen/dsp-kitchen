//! The app's shared state, as one entity: the open recording, the timeline every view follows,
//! the channel selection of each source, the workspace shown, recent files and the status line.
//! Changes go through its methods, which emit the [`AppEvent`] they caused (never a blanket
//! `notify`); view models subscribe to the events they show.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use gpui_kit::{AppContext as _, Context, EventEmitter, Task};

use dspk_view::{Progress, SignalBackend};
use crate::data::jobs::{self, Cancel, JobMessage};
use crate::data::recording::processed::ProcessedSignal;
use crate::data::recording::{Dataset, SourceSet, SpikeEventStore};
use crate::ui::theme::palette::Palette;
use crate::domain::timeline::TimelineState;
use crate::screens::explore::trace::view::parse_channel_ranges;
use crate::data::preferences::Session;
use crate::domain::command::{apply, Changed, CommandError, SessionCommand};
use crate::domain::session::{RecordingRef, SessionDoc, Show, ViewDoc};
use crate::navigation::route::Workspace;

/// Playback clock period.
const TICK: Duration = Duration::from_millis(16);
/// How often the session document file is checked for edits made outside the app (an agent).
const DOCUMENT_POLL: Duration = Duration::from_millis(500);

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AppEvent {
    /// A recording was opened (or closed): views start over.
    RecordingChanged,
    /// The visible time window moved (views redraw).
    WindowMoved,
    /// The playhead, play state, loop or speed changed (overlays and transport only).
    PlaybackChanged,
    /// The channel selection of a source changed.
    SelectionChanged(String),
    WorkspaceChanged,
    /// Plot colours changed (theme or "dark plots"): views redraw.
    PaletteChanged,
    /// More of a source's min/max pyramid is built (views of it that drew empty columns redraw).
    PyramidProgress(String),
    /// The status line or the recent list changed.
    Status,
    /// The session document changed (by the UI or from its file): what changed.
    Document(Changed),
    /// A channel (or an op of the active view) was picked: the Processing panel follows.
    Picked,
    /// A job started, progressed or ended (status bar, panels).
    JobChanged,
    /// A plugin's own state changed (its panel redraws).
    PluginChanged,
}

/// Work running for the UI (one at a time): what it is, how far, how to stop it.
pub struct RunningJob {
    pub label: String,
    pub fraction: Option<f32>,
    pub line: String,
    pub cancel: Cancel,
}

/// The open recording.
pub struct Recording {
    pub sources: Arc<SourceSet>,
    pub name: String,
    /// Its file (`None`: in memory or synthetic).
    pub path: Option<PathBuf>,
    /// Spike events shown on traces and in the timeline overview (empty until an extraction runs).
    pub events: Arc<SpikeEventStore>,
}

impl Recording {
    pub fn new(sources: SourceSet, path: Option<PathBuf>) -> Self {
        let name = match &path {
            Some(p) => p.file_name().map_or_else(|| p.display().to_string(), |n| n.to_string_lossy().into_owned()),
            None => sources.default_entry().name.clone(),
        };
        Self { sources: Arc::new(sources), name, path, events: Arc::new(SpikeEventStore::default()) }
    }

    /// `32 ch · 30.0 kHz · 5 min 00 s` of the default source.
    pub fn summary(&self) -> String {
        let e = self.sources.default_entry();
        let hz = e.sample_rate.rate_hz();
        let rate = if hz >= 1000.0 { format!("{:.1} kHz", hz / 1000.0) } else { format!("{hz:.1} Hz") };
        format!("{} ch · {rate} · {}", e.channels, duration(self.sources.extent_sec()))
    }
}

pub struct Store {
    pub recording: Option<Recording>,
    pub timeline: TimelineState,
    /// Channels shown, per source id, in display order.
    selection: HashMap<String, Vec<usize>>,
    pub workspace: Workspace,
    pub session: Session,
    session_path: Option<PathBuf>,
    /// What is being opened, while a file opens.
    pub opening: Option<String>,
    pub status: String,
    /// Render time and density of the latest frame (status bar).
    pub render_info: String,
    /// The playback clock while playing.
    clock: Option<Task<()>>,
    /// The app is drawn dark (the user's choice or the system's).
    dark: bool,
    /// Pyramid progress per source id being built: (samples built, of all).
    pub pyramids: HashMap<String, (u64, u64)>,
    /// The session document: views, their op chains, display and linked sortings.
    pub doc: SessionDoc,
    doc_path: Option<PathBuf>,
    /// The text last written to or read from `doc_path` (an edit from outside differs from it).
    doc_text: String,
    /// Watches `doc_path` for edits made outside the app.
    _doc_watch: Option<Task<()>>,
    /// Live preview of each view's op chain (views of the open recording with enabled ops).
    previews: HashMap<String, Arc<ProcessedSignal>>,
    /// The channel the Processing panel is about, and the op selected in it.
    pub picked_channel: Option<usize>,
    pub picked_op: Option<String>,
    pub job: Option<RunningJob>,
    /// The active view and channel subset last applied to the selection.
    applied_view: Option<(String, Option<Vec<usize>>)>,
}

impl EventEmitter<AppEvent> for Store {}

impl Store {
    pub fn new(session: Session, session_path: Option<PathBuf>) -> Self {
        Self {
            recording: None,
            timeline: TimelineState::new(1.0),
            selection: HashMap::new(),
            workspace: session.workspace,
            session,
            session_path,
            opening: None,
            status: String::new(),
            render_info: String::new(),
            clock: None,
            dark: true,
            pyramids: HashMap::new(),
            doc: SessionDoc::default(),
            doc_path: None,
            doc_text: String::new(),
            _doc_watch: None,
            previews: HashMap::new(),
            picked_channel: None,
            picked_op: None,
            job: None,
            applied_view: None,
        }
    }

    // ------------------------------------------------------------------------
    // Session document (one code path for the UI and an agent)
    // ------------------------------------------------------------------------

    /// Loads the session document at `path` (an unreadable one is reported and replaced by an
    /// empty one when first saved) and follows the file: edits made outside the app (an agent)
    /// are applied as they land.
    pub fn attach_document(&mut self, path: PathBuf, cx: &mut Context<Self>) {
        match std::fs::read_to_string(&path) {
            Ok(text) => match SessionDoc::from_json(&text) {
                Ok(doc) => {
                    self.doc = doc;
                    self.doc_text = text;
                }
                Err(e) => tracing::warn!("{}: {e}", path.display()),
            },
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => tracing::warn!("{}: {e}", path.display()),
        }
        self.doc_path = Some(path.clone());
        self._doc_watch = Some(cx.spawn(async move |this, cx| loop {
            cx.background_executor().timer(DOCUMENT_POLL).await;
            let text = std::fs::read_to_string(&path).unwrap_or_default();
            if this.update(cx, |s, cx| s.document_file_changed(text, cx)).is_err() {
                break;
            }
        }));
        cx.emit(AppEvent::Document(Changed::Views));
    }

    /// Applies an edit of the document file made outside the app.
    fn document_file_changed(&mut self, text: String, cx: &mut Context<Self>) {
        if text.is_empty() || text == self.doc_text {
            return;
        }
        self.doc_text = text;
        match SessionDoc::from_json(&self.doc_text) {
            Ok(doc) if doc != self.doc => {
                let before = std::mem::replace(&mut self.doc, doc);
                for change in diff(&before, &self.doc) {
                    self.changed(&change, cx);
                    cx.emit(AppEvent::Document(change));
                }
                self.set_status("Session updated from its file", cx);
            }
            Ok(_) => {}
            Err(e) => self.set_status(format!("Session file not applied: {e}"), cx),
        }
    }

    /// Runs a command on the session document, saves it and emits what changed.
    pub fn execute(&mut self, cmd: SessionCommand, cx: &mut Context<Self>) -> Result<Changed, CommandError> {
        let result = apply(&mut self.doc, cmd);
        match &result {
            Ok(change) => {
                self.save_document();
                self.changed(change, cx);
                cx.emit(AppEvent::Document(change.clone()));
            }
            Err(e) => self.set_status(e.to_string(), cx),
        }
        result
    }

    fn save_document(&mut self) {
        let Some(path) = &self.doc_path else { return };
        let text = self.doc.to_json();
        if text == self.doc_text {
            return;
        }
        match self.doc.save(path) {
            Ok(()) => self.doc_text = text,
            Err(e) => tracing::warn!("could not save the session document: {e}"),
        }
    }

    /// A view of `path`'s `source` (the active one if it shows it; else the first; else a new one).
    fn ensure_view(&mut self, path: &Path, source: &str, cx: &mut Context<Self>) {
        let shows = |v: &ViewDoc| v.recording.path == path && v.recording.source.as_deref().unwrap_or(source) == source;
        if !self.doc.active().is_some_and(shows) {
            if let Some(id) = self.doc.views.iter().find(|v| shows(v)).map(|v| v.id.clone()) {
                let _ = self.execute(SessionCommand::SelectView { view: id }, cx);
            } else {
                let recording = RecordingRef { path: path.to_path_buf(), source: Some(source.to_string()), channels: None };
                let _ = self.execute(SessionCommand::OpenView { recording, name: None }, cx);
            }
        }
        // The previews of every view of this recording, and the active view's channels
        let ids: Vec<String> = self.doc.views.iter().map(|v| v.id.clone()).collect();
        for id in ids {
            self.rebuild_preview(&id, cx);
        }
        self.apply_active_view(cx);
    }

    // ------------------------------------------------------------------------
    // Views of the session document on the open recording
    // ------------------------------------------------------------------------

    /// The active view of the session document.
    pub fn active_view(&self) -> Option<&ViewDoc> {
        self.doc.active()
    }

    /// Whether `view` reads source `source` of the open recording.
    fn on_open_recording(&self, view: &ViewDoc, source: &str) -> bool {
        let Some(rec) = &self.recording else { return false };
        let Some(path) = &rec.path else { return false };
        let default = rec.sources.default_entry().id.clone();
        view.recording.path == *path && view.recording.source.as_deref().unwrap_or(&default) == source
    }

    /// Source id (of the open recording) a view reads.
    pub fn view_source(&self, view: &ViewDoc) -> String {
        let default = self.sources().map(|s| s.default_entry().id.clone()).unwrap_or_default();
        view.recording.source.clone().unwrap_or(default)
    }

    /// What views of `source` draw: the active view's processed preview when it reads `source`,
    /// shows `processed` and has enabled ops; else `None` (the raw recording).
    pub fn signal_for(&self, source: &str) -> Option<Arc<dyn SignalBackend>> {
        let v = self.doc.active()?;
        if v.display.show != Show::Processed || !self.on_open_recording(v, source) {
            return None;
        }
        self.previews.get(&v.id).map(|p| p.clone() as Arc<dyn SignalBackend>)
    }

    /// Whether the visible window of `samples` samples is previewed processed (else raw).
    pub fn preview_covers(&self, source: &str, samples: u64) -> Option<bool> {
        let v = self.doc.active()?;
        if v.display.show != Show::Processed || !self.on_open_recording(v, source) {
            return None;
        }
        self.previews.get(&v.id).map(|p| p.covers(samples))
    }

    /// Rebuilds the preview of view `id` from its enabled ops (none: no preview).
    fn rebuild_preview(&mut self, id: &str, cx: &mut Context<Self>) {
        self.previews.remove(id);
        let Some(v) = self.doc.view(id).cloned() else { return };
        let source = self.view_source(&v);
        if !self.on_open_recording(&v, &source) {
            return;
        }
        let Some(sources) = self.sources().cloned() else { return };
        let ds = sources.get(&source);
        let Some(raw) = ds.source.clone() else { return };
        let mut stages = Vec::new();
        for op in v.ops.iter().filter(|o| o.enabled) {
            match op.node.stage() {
                Ok(Some(stage)) => stages.push(stage),
                Ok(None) => {}
                Err(e) => {
                    self.set_status(format!("{}: op {}: {e}", v.name, op.id), cx);
                    return;
                }
            }
        }
        if stages.is_empty() {
            return;
        }
        let group = v.recording.channels.clone().unwrap_or_else(|| (0..ds.total_channels).collect());
        match ProcessedSignal::new(ds.signal().clone(), raw, group, stages) {
            Ok(p) => {
                self.previews.insert(id.to_string(), Arc::new(p));
            }
            Err(e) => self.set_status(format!("{}: {e}", v.name), cx),
        }
    }

    /// The active view's channel subset becomes the shown selection of its source (once per
    /// change of view or subset, so the user's own selection is kept otherwise).
    fn apply_active_view(&mut self, cx: &mut Context<Self>) {
        let Some(v) = self.doc.active().cloned() else { return };
        let key = (v.id.clone(), v.recording.channels.clone());
        if self.applied_view.as_ref() == Some(&key) {
            return;
        }
        let source = self.view_source(&v);
        if !self.on_open_recording(&v, &source) {
            return;
        }
        self.applied_view = Some(key);
        match &v.recording.channels {
            Some(ch) => self.set_selection(&source, ch.clone(), cx),
            None => self.select_all(&source, cx),
        }
    }

    /// What follows a change of the document: previews, the selection, redraws.
    fn changed(&mut self, change: &Changed, cx: &mut Context<Self>) {
        match change {
            Changed::Views => {
                let ids: Vec<String> = self.doc.views.iter().map(|v| v.id.clone()).collect();
                self.previews.retain(|id, _| ids.contains(id));
                if let Some(id) = self.doc.active_view.clone() {
                    if !self.previews.contains_key(&id) {
                        self.rebuild_preview(&id, cx);
                    }
                }
                self.picked_op = None;
                self.apply_active_view(cx);
                cx.emit(AppEvent::WindowMoved);
            }
            Changed::Ops(id) => {
                self.rebuild_preview(id, cx);
                if self.picked_op.as_ref().is_some_and(|op| self.doc.view(id).is_some_and(|v| v.op(op).is_none())) {
                    self.picked_op = None;
                }
                cx.emit(AppEvent::WindowMoved);
            }
            Changed::Display(_) | Changed::Plugin(..) => cx.emit(AppEvent::WindowMoved),
        }
    }

    /// Views redraw (a plugin's overlay changed).
    pub fn redraw(&mut self, cx: &mut Context<Self>) {
        cx.emit(AppEvent::WindowMoved);
        cx.notify();
    }

    /// The channel the Processing panel is about (`None`: none).
    pub fn pick_channel(&mut self, channel: Option<usize>, cx: &mut Context<Self>) {
        self.picked_channel = channel;
        cx.emit(AppEvent::Picked);
        cx.notify();
    }

    /// The op of the active view selected in the Processing panel.
    pub fn pick_op(&mut self, op: Option<String>, cx: &mut Context<Self>) {
        self.picked_op = op;
        cx.emit(AppEvent::Picked);
        cx.notify();
    }

    // ------------------------------------------------------------------------
    // Jobs (one at a time)
    // ------------------------------------------------------------------------

    /// Starts a job: `start` gets the sender of its messages and returns its cancel handle;
    /// `done` runs on the UI thread with the file it wrote.
    pub fn start_job(
        &mut self,
        label: impl Into<String>,
        start: impl FnOnce(Box<dyn Fn(JobMessage) + Send + Sync>) -> Cancel,
        done: impl FnOnce(&mut Self, PathBuf, &mut Context<Self>) + 'static,
        cx: &mut Context<Self>,
    ) {
        if self.job.is_some() {
            self.set_status("Another job is running: wait for it or cancel it", cx);
            return;
        }
        let label = label.into();
        let (tx, rx) = async_channel::unbounded::<JobMessage>();
        let cancel = start(Box::new(move |m| {
            let _ = tx.try_send(m);
        }));
        self.job = Some(RunningJob { label: label.clone(), fraction: None, line: String::new(), cancel });
        self.set_status(format!("{label}…"), cx);
        cx.emit(AppEvent::JobChanged);
        let mut done = Some(done);
        cx.spawn(async move |this, cx| {
            while let Ok(m) = rx.recv().await {
                let finished = matches!(m, JobMessage::Done(_) | JobMessage::Failed(_));
                let alive = this.update(cx, |s, cx| {
                    match m {
                        JobMessage::Progress(line, fraction) => {
                            if let Some(j) = &mut s.job {
                                j.line = line.clone();
                                j.fraction = fraction.or(j.fraction);
                            }
                            s.status = format!("{label}: {line}");
                            cx.emit(AppEvent::Status);
                        }
                        JobMessage::Done(path) => {
                            s.job = None;
                            s.set_status(format!("{label}: done ({})", file_label(&path)), cx);
                            if let Some(done) = done.take() {
                                done(s, path, cx);
                            }
                        }
                        JobMessage::Failed(e) => {
                            s.job = None;
                            s.set_status(format!("{label}: {e}"), cx);
                        }
                    }
                    cx.emit(AppEvent::JobChanged);
                });
                if alive.is_err() || finished {
                    break;
                }
            }
        })
        .detach();
    }

    pub fn cancel_job(&mut self, cx: &mut Context<Self>) {
        if let Some(j) = &self.job {
            j.cancel.cancel();
            self.set_status(format!("Cancelling {}…", j.label), cx);
        }
    }

    /// Writes the active view (its recording through its enabled ops) to
    /// `<recording>.<view>.processed.zarr` next to the recording.
    pub fn write_active_view(&mut self, cx: &mut Context<Self>) {
        let Some(v) = self.doc.active().cloned() else { return };
        let output = jobs::processed_path(&v.recording.path, &v.id);
        self.start_job(
            format!("Write {}", v.name),
            move |send| jobs::write_view(&v, output, move |m| send(m)),
            |s, path, cx| s.set_status(format!("Written: {}", path.display()), cx),
            cx,
        );
    }

    // ------------------------------------------------------------------------
    // Theme
    // ------------------------------------------------------------------------

    /// Plot colours: dark with the dark theme, or when the user keeps plots dark.
    pub fn palette(&self) -> Palette {
        Palette::of(self.dark || self.session.dark_plots)
    }

    /// The theme the app is drawn with now.
    pub fn set_dark(&mut self, dark: bool, cx: &mut Context<Self>) {
        let before = self.palette();
        self.dark = dark;
        if self.palette() != before {
            cx.emit(AppEvent::PaletteChanged);
        }
    }

    /// The user's theme choice (`None`: follow the system), saved.
    pub fn set_theme_choice(&mut self, dark: Option<bool>, cx: &mut Context<Self>) {
        self.session.dark = dark;
        self.save_session();
        cx.notify();
    }

    pub fn set_dark_plots(&mut self, on: bool, cx: &mut Context<Self>) {
        let before = self.palette();
        self.session.dark_plots = on;
        self.save_session();
        if self.palette() != before {
            cx.emit(AppEvent::PaletteChanged);
        }
        cx.notify();
    }

    pub fn sources(&self) -> Option<&Arc<SourceSet>> {
        self.recording.as_ref().map(|r| &r.sources)
    }

    fn save_session(&mut self) {
        self.session.workspace = self.workspace;
        if let Err(e) = self.session.save(self.session_path.as_deref()) {
            tracing::warn!("could not save the session: {e}");
        }
    }

    pub fn set_explore_docks(&mut self, left: bool, right: bool, bottom: bool) {
        let ex = &mut self.session.explore;
        if (ex.left_open, ex.right_open, ex.bottom_open) != (left, right, bottom) {
            ex.left_open = left;
            ex.right_open = right;
            ex.bottom_open = bottom;
            self.save_session();
        }
    }

    /// The panel shown on the right side bar (saved).
    pub fn set_explore_right(&mut self, key: &str) {
        if self.session.explore.right_panel != key {
            self.session.explore.right_panel = key.to_string();
            self.save_session();
        }
    }

    /// A plugin's own state changed (its panels redraw; views do not).
    pub fn plugin_changed(&mut self, cx: &mut Context<Self>) {
        cx.emit(AppEvent::PluginChanged);
        cx.notify();
    }

    pub fn set_explore_views(&mut self, views: Vec<crate::data::preferences::SavedTimeView>) {
        if self.session.explore.views != views {
            self.session.explore.views = views;
            self.save_session();
        }
    }

    #[allow(dead_code)] // Curation (parked) opens sortings
    pub fn push_recent_sorting(&mut self, path: &Path) {
        self.session.push_recent_sorting(path);
        self.save_session();
    }

    #[allow(dead_code)] // set by spike extraction, which this branch does not run yet
    pub fn set_events(&mut self, events: Arc<SpikeEventStore>, cx: &mut Context<Self>) {
        if let Some(rec) = &mut self.recording {
            rec.events = events;
            cx.emit(AppEvent::WindowMoved);
        }
    }

    pub fn set_status(&mut self, text: impl Into<String>, cx: &mut Context<Self>) {
        self.status = text.into();
        cx.emit(AppEvent::Status);
    }

    /// Latest frame's numbers (no event: the status bar observes the store).
    pub fn set_render_info(&mut self, text: String, cx: &mut Context<Self>) {
        if self.render_info != text {
            self.render_info = text;
            cx.notify();
        }
    }

    // ------------------------------------------------------------------------
    // Recording
    // ------------------------------------------------------------------------

    /// Lists and opens `path`'s sources on a background thread.
    pub fn open(&mut self, path: PathBuf, cx: &mut Context<Self>) {
        let name = path.file_name().map_or_else(|| path.display().to_string(), |n| n.to_string_lossy().into_owned());
        self.opening = Some(name.clone());
        self.set_status(format!("Opening {name}…"), cx);
        let task = cx.background_spawn({
            let path = path.clone();
            async move { SourceSet::open(&path) }
        });
        cx.spawn(async move |this, cx| {
            let result = task.await;
            let _ = this.update(cx, |store, cx| {
                store.opening = None;
                match result {
                    Ok(sources) => {
                        store.session.push_recent(&path);
                        let source = sources.default_entry().id.clone();
                        store.install(Recording::new(sources, Some(path.clone())), cx);
                        store.ensure_view(&path, &source, cx);
                    }
                    Err(e) => store.set_status(format!("Could not open {name}: {e:#}"), cx),
                }
            });
        })
        .detach();
    }

    /// Opens a procedural recording (noise, hum, drifting units) computed on demand.
    pub fn open_synthetic(&mut self, channels: usize, sample_rate: f64, duration_sec: f64, cx: &mut Context<Self>) {
        match Dataset::procedural(channels, sample_rate, duration_sec) {
            Ok(ds) => self.install(Recording::new(SourceSet::single(ds), None), cx),
            Err(e) => self.set_status(format!("Could not make a synthetic recording: {e:#}"), cx),
        }
    }

    /// Installs a recording directly (tests, command line).
    pub fn install(&mut self, recording: Recording, cx: &mut Context<Self>) {
        self.stop_clock();
        self.timeline = TimelineState::new(recording.sources.extent_sec());
        self.selection = recording.sources.entries().iter().map(|e| (e.id.clone(), (0..e.channels).collect())).collect();
        self.pyramids.clear();
        self.previews.clear();
        self.applied_view = None;
        self.picked_channel = None;
        self.picked_op = None;
        self.status = format!("Opened {} ({})", recording.name, recording.summary());
        self.recording = Some(recording);
        self.save_session();
        cx.emit(AppEvent::RecordingChanged);
        cx.emit(AppEvent::Status);
    }

    pub fn set_workspace(&mut self, workspace: Workspace, cx: &mut Context<Self>) {
        if self.workspace != workspace {
            self.workspace = workspace;
            self.save_session();
            cx.emit(AppEvent::WorkspaceChanged);
        }
    }

    /// Views of `source` look at `focus_sec`: its backend builds the min/max pyramid there first
    /// (the build starts on the first call; later calls move the focus). Progress is followed
    /// once per source and emitted as [`AppEvent::PyramidProgress`].
    pub fn focus_pyramid(&mut self, source: &str, focus_sec: f64, cx: &mut Context<Self>) {
        let Some(sources) = self.sources().cloned() else { return };
        let ds = sources.get(source);
        let signal = ds.signal().clone();
        let progress = signal.progress();
        if progress.done >= progress.total {
            return;
        }
        if !self.pyramids.contains_key(source) {
            self.pyramids.insert(source.to_string(), (progress.done, progress.total));
            let (tx, rx) = async_channel::unbounded::<Progress>();
            let id = source.to_string();
            cx.spawn(async move |this, cx| {
                while let Ok(p) = rx.recv().await {
                    let alive = this.update(cx, |s, cx| {
                        if let Some(entry) = s.pyramids.get_mut(&id) {
                            *entry = (p.done, p.total);
                            cx.emit(AppEvent::PyramidProgress(id.clone()));
                        }
                    });
                    if alive.is_err() {
                        break;
                    }
                }
            })
            .detach();
            signal.watch(Arc::new(move |p| {
                let _ = tx.try_send(p);
            }));
        }
        let focus = ((focus_sec - ds.start_time_sec) * ds.sample_rate).max(0.0) as u64;
        signal.focus(focus);
    }

    /// `Building overview 12 / 45 s` while the default source's pyramid builds.
    pub fn pyramid_label(&self) -> Option<String> {
        let sources = self.sources()?;
        let e = sources.default_entry();
        let &(done, total) = self.pyramids.get(&e.id)?;
        let hz = e.sample_rate.rate_hz();
        (done < total && hz > 0.0).then(|| format!("Building overview {:.0} / {:.0} s", done as f64 / hz, total as f64 / hz))
    }

    // ------------------------------------------------------------------------
    // Channel selection (shared by every view of a source)
    // ------------------------------------------------------------------------

    /// Channels of `source` shown, in display order.
    pub fn selection(&self, source: &str) -> &[usize] {
        self.selection.get(source).map_or(&[], Vec::as_slice)
    }

    fn channels_of(&self, source: &str) -> usize {
        self.sources().and_then(|s| s.entries().iter().find(|e| e.id == source)).map_or(0, |e| e.channels)
    }

    /// Replaces the selection (duplicates and out-of-range channels dropped, order kept).
    pub fn set_selection(&mut self, source: &str, channels: impl IntoIterator<Item = usize>, cx: &mut Context<Self>) {
        let total = self.channels_of(source);
        let mut seen = vec![false; total];
        let list: Vec<usize> = channels.into_iter().filter(|&c| c < total && !std::mem::replace(&mut seen[c], true)).collect();
        self.selection.insert(source.to_string(), list);
        cx.emit(AppEvent::SelectionChanged(source.to_string()));
    }

    /// Adds or removes one channel (added channels keep ascending order).
    pub fn toggle_channel(&mut self, source: &str, channel: usize, cx: &mut Context<Self>) {
        let mut list = self.selection(source).to_vec();
        match list.iter().position(|&c| c == channel) {
            Some(i) => {
                list.remove(i);
            }
            None => {
                let at = list.partition_point(|&c| c < channel);
                list.insert(at, channel);
            }
        }
        self.set_selection(source, list, cx);
    }

    pub fn select_all(&mut self, source: &str, cx: &mut Context<Self>) {
        let total = self.channels_of(source);
        self.set_selection(source, 0..total, cx);
    }

    pub fn select_none(&mut self, source: &str, cx: &mut Context<Self>) {
        self.set_selection(source, [], cx);
    }

    pub fn select_invert(&mut self, source: &str, cx: &mut Context<Self>) {
        let total = self.channels_of(source);
        let keep: Vec<usize> = (0..total).filter(|c| !self.selection(source).contains(c)).collect();
        self.set_selection(source, keep, cx);
    }

    /// Applies a range expression (`0-31, 40`); the error message to show otherwise.
    pub fn select_ranges(&mut self, source: &str, text: &str, cx: &mut Context<Self>) -> Result<(), String> {
        let channels = parse_channel_ranges(text, self.channels_of(source))?;
        self.set_selection(source, channels, cx);
        Ok(())
    }

    // ------------------------------------------------------------------------
    // Timeline (one for the whole app)
    // ------------------------------------------------------------------------

    fn window_moved(&mut self, cx: &mut Context<Self>) {
        cx.emit(AppEvent::WindowMoved);
        cx.emit(AppEvent::PlaybackChanged);
    }

    /// Moves the window by `dt` seconds (the playhead stays).
    pub fn pan_time(&mut self, dt: f64, cx: &mut Context<Self>) {
        let before = self.timeline.window_start_sec;
        self.timeline.pan_time(dt);
        if self.timeline.window_start_sec != before {
            cx.emit(AppEvent::WindowMoved);
        }
    }

    /// Moves the window by a fraction of its length.
    pub fn pan_fraction(&mut self, fraction: f64, cx: &mut Context<Self>) {
        self.pan_time(fraction * self.timeline.visible_window_sec, cx);
    }

    /// Scales the window by `factor`, keeping the time at `anchor` (0 = start, 1 = end) in place.
    pub fn zoom_at(&mut self, factor: f64, anchor: f64, cx: &mut Context<Self>) {
        self.timeline.zoom_at(factor, anchor);
        cx.emit(AppEvent::WindowMoved);
    }

    /// Sets the window to `[start, end]` seconds.
    pub fn set_window(&mut self, start: f64, end: f64, cx: &mut Context<Self>) {
        self.timeline.set_window_range(start.min(end), start.max(end));
        cx.emit(AppEvent::WindowMoved);
    }

    /// Puts the playhead at `t` seconds (the window follows it).
    pub fn scrub_to(&mut self, t: f64, cx: &mut Context<Self>) {
        self.timeline.scrub_to(t);
        self.window_moved(cx);
    }

    pub fn jump_to_start(&mut self, cx: &mut Context<Self>) {
        self.scrub_to(0.0, cx);
    }

    pub fn jump_to_end(&mut self, cx: &mut Context<Self>) {
        self.scrub_to(self.timeline.total_duration_sec, cx);
    }

    /// Steps the playhead by a tenth of the window.
    pub fn step(&mut self, forward: bool, cx: &mut Context<Self>) {
        let step = self.timeline.visible_window_sec * 0.1;
        if forward {
            self.timeline.step_forward(step);
        } else {
            self.timeline.step_backward(step);
        }
        self.window_moved(cx);
    }

    pub fn toggle_loop(&mut self, cx: &mut Context<Self>) {
        self.timeline.toggle_loop();
        cx.emit(AppEvent::PlaybackChanged);
    }

    pub fn set_speed(&mut self, speed: f64, cx: &mut Context<Self>) {
        self.timeline.playback_speed = speed;
        cx.emit(AppEvent::PlaybackChanged);
    }

    pub fn toggle_play(&mut self, cx: &mut Context<Self>) {
        if self.recording.is_none() {
            return;
        }
        self.timeline.toggle_play();
        if self.timeline.is_playing {
            self.start_clock(cx);
        } else {
            self.stop_clock();
        }
        cx.emit(AppEvent::PlaybackChanged);
    }

    fn stop_clock(&mut self) {
        self.timeline.is_playing = false;
        self.clock = None;
    }

    /// Advances playback every [`TICK`] while playing; dropped (and so stopped) on pause.
    fn start_clock(&mut self, cx: &mut Context<Self>) {
        self.clock = Some(cx.spawn(async move |this, cx| {
            let mut last = Instant::now();
            loop {
                cx.background_executor().timer(TICK).await;
                let now = Instant::now();
                let dt = now.duration_since(last).as_secs_f64();
                last = now;
                let playing = this.update(cx, |store, cx| {
                    store.timeline.advance(dt);
                    store.window_moved(cx);
                    store.timeline.is_playing
                });
                if !matches!(playing, Ok(true)) {
                    break;
                }
            }
        }));
    }
}

/// What differs between two documents, as the changes a command would have reported.
fn diff(before: &SessionDoc, after: &SessionDoc) -> Vec<Changed> {
    let ids = |d: &SessionDoc| d.views.iter().map(|v| (v.id.clone(), v.name.clone(), v.recording.clone())).collect::<Vec<_>>();
    if ids(before) != ids(after) || before.active_view != after.active_view {
        return vec![Changed::Views];
    }
    let mut out = Vec::new();
    for (a, b) in before.views.iter().zip(&after.views) {
        if a.ops != b.ops {
            out.push(Changed::Ops(b.id.clone()));
        }
        if a.display != b.display {
            out.push(Changed::Display(b.id.clone()));
        }
        for id in a.plugins.keys().chain(b.plugins.keys()) {
            if a.plugins.get(id) != b.plugins.get(id) && !out.contains(&Changed::Plugin(b.id.clone(), id.clone())) {
                out.push(Changed::Plugin(b.id.clone(), id.clone()));
            }
        }
    }
    out
}

/// Seconds as `1 h 02 min`, `3 min 05 s`, `4.20 s` or `33 ms`.
pub fn duration(seconds: f64) -> String {
    let s = seconds.max(0.0);
    if s >= 3600.0 {
        format!("{} h {:02} min", (s / 3600.0) as u64, ((s % 3600.0) / 60.0) as u64)
    } else if s >= 60.0 {
        format!("{} min {:02} s", (s / 60.0) as u64, (s % 60.0) as u64)
    } else if s >= 1.0 {
        format!("{s:.2} s")
    } else {
        format!("{:.0} ms", s * 1e3)
    }
}

/// The file name of `path` for labels.
pub fn file_label(path: &Path) -> String {
    path.file_name().map_or_else(|| path.display().to_string(), |n| n.to_string_lossy().into_owned())
}
