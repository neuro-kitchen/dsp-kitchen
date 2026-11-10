//! The app's shared state, as one entity: the open recording, the timeline every view follows,
//! the channel selection of each source, the workspace shown, recent files and the status line.
//! Changes go through its methods, which emit the [`AppEvent`] they caused (never a blanket
//! `notify`); view models subscribe to the events they show.

use dspk_ops::PipelineStage;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use gpui_kit::{AppContext as _, Context, EventEmitter, Task};

use dspk_view::{Progress, SignalBackend};
use crate::data::jobs::{self, Cancel, JobMessage};
use crate::data::recording::processed::{PreviewSource, ProcessedSignal};
use crate::data::recording::{Dataset, SourceSet, SpikeEventStore};
use crate::ui::theme::palette::Palette;
use crate::domain::timeline::TimelineState;
use crate::screens::explore::trace::view::parse_channel_ranges;
use crate::data::preferences::Session;
use crate::domain::command::{apply, Changed, CommandError, SessionCommand};
use crate::domain::session::{RecordingRef, SessionDoc, Show, ViewDoc};
use crate::navigation::route::Workspace;

/// Previews kept by chain.
const PREVIEW_CACHE: usize = 6;
/// Changes of the session that can be undone.
const UNDO_DEPTH: usize = 200;
/// Playback clock period.
const TICK: Duration = Duration::from_millis(16);
/// How often the session document file is checked for edits made outside the app (an agent).
const DOCUMENT_POLL: Duration = Duration::from_millis(500);

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AppEvent {
    /// A recording was opened (or closed): views start over.
    RecordingChanged,
    /// A recording was loaded beside the open one (its views can be added).
    RecordingLoaded(PathBuf),
    /// Show the Epochs view of session view `id` in Explore.
    ShowEpochs(String),
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
    /// A floating window opened or closed.
    Floating,
}

/// A floating window over the workspace.
#[derive(Debug, Clone, PartialEq)]
pub enum Floating {
    /// What is known about a recording's source.
    Metadata { path: PathBuf, source: String },
    /// An epoch to save (new or edited).
    Epoch(crate::domain::epoch::EpochDoc),
}

/// Work running for the UI (one at a time): what it is, how far, how to stop it.
pub struct RunningJob {
    pub label: String,
    pub fraction: Option<f32>,
    pub line: String,
    pub cancel: Cancel,
}

/// What a graph node's last job is doing (the Pipeline canvas shows it on the node).
#[derive(Debug, Clone, PartialEq)]
pub enum NodeRun {
    Running { line: String, fraction: Option<f32>, since: Instant },
    Done { took: Duration },
    Failed(String),
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
        self.summary_of(&self.sources.default_entry().id)
    }

    /// `32 ch · 30.0 kHz · 5 min 00 s` of source `id`.
    pub fn summary_of(&self, id: &str) -> String {
        let e = self.sources.entry(id);
        let hz = e.sample_rate.rate_hz();
        let rate = if hz >= 1000.0 { format!("{:.1} kHz", hz / 1000.0) } else { format!("{hz:.1} Hz") };
        format!("{} ch · {rate} · {}", e.channels, duration(e.duration_sec()))
    }
}

/// How a view is previewed ([`Store::view_plan`]).
pub struct ViewPlan {
    /// Source channels read, in order.
    pub group: Vec<usize>,
    /// Device steps and their keys (for sharing computed prefixes).
    pub steps: Vec<(String, PipelineStage)>,
    /// Source channels shown, in order.
    pub shown: Vec<usize>,
    /// Whitening steps (by index in `steps`, identities until fitted) and how to fit them.
    pub fits: Vec<(usize, dspk_pipeline::WhitenFit)>,
}

pub struct Store {
    /// The active recording (what Explore draws and Processing edits).
    pub recording: Option<Recording>,
    /// Other recordings loaded this session (each file once): switching to one is instant.
    pub loaded: Vec<Recording>,
    /// A recording's source the Inspector shows (picked under Recordings; a view's focus clears it).
    pub inspected: Option<(PathBuf, String)>,
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
    /// Previews by chain (source, channels, enabled ops), most recent last: views with the same
    /// chain share one, and a branch reads the preview of the ops it shares with another.
    preview_cache: Vec<(Vec<String>, Arc<ProcessedSignal>)>,
    /// Whitenings fitted for previews, by the key of the chain up to them (fitting reads the
    /// recording: once per chain, not per redraw).
    whiten_cache: HashMap<String, dspk_ops::spatial::SpatialWhitening>,
    /// Whitening fits running in the background (their keys).
    whiten_fitting: std::collections::HashSet<String>,
    /// The channel the Processing panel is about, and the op selected in it.
    pub picked_channel: Option<usize>,
    pub picked_op: Option<String>,
    pub job: Option<RunningJob>,
    /// The last job that failed: its label and why (cleared when the next job starts).
    pub last_failed: Option<(String, String)>,
    /// Graph nodes the running job works on, and every node's last run.
    job_nodes: Vec<String>,
    pub node_status: HashMap<String, NodeRun>,
    /// Floating windows, front last.
    pub floating: Vec<Floating>,
    /// Earlier documents (undo) and undone ones (redo).
    history: Vec<SessionDoc>,
    future: Vec<SessionDoc>,
    /// The active view and channel subset last applied to the selection.
    applied_view: Option<(String, Option<Vec<usize>>)>,
    /// Event tables read so far, by path (read on first use; dropped when the set is set again).
    event_tables: HashMap<PathBuf, Arc<Vec<crate::data::events::Event>>>,
}

impl EventEmitter<AppEvent> for Store {}

impl Store {
    pub fn new(session: Session, session_path: Option<PathBuf>) -> Self {
        crate::data::remote::set_profiles(&session.servers);
        Self {
            recording: None,
            loaded: Vec::new(),
            inspected: None,
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
            preview_cache: Vec::new(),
            whiten_cache: HashMap::new(),
            whiten_fitting: Default::default(),
            picked_channel: None,
            picked_op: None,
            job: None,
            last_failed: None,
            job_nodes: Vec::new(),
            node_status: HashMap::new(),
            floating: Vec::new(),
            history: Vec::new(),
            future: Vec::new(),
            applied_view: None,
            event_tables: HashMap::new(),
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
        let before = self.doc.clone();
        let result = apply(&mut self.doc, cmd);
        match &result {
            Ok(_) => {
                if before != self.doc {
                    self.history.push(before.clone());
                    if self.history.len() > UNDO_DEPTH {
                        self.history.remove(0);
                    }
                    self.future.clear();
                }
                self.save_document();
                // What the command changed in the views (a graph edit can change several chains)
                for change in diff(&before, &self.doc) {
                    self.changed(&change, cx);
                    cx.emit(AppEvent::Document(change));
                }
            }
            Err(e) => self.set_status(e.to_string(), cx),
        }
        if result.is_ok() && before.epochs != self.doc.epochs {
            self.write_epochs(&before, cx);
        }
        result
    }

    /// Back to the document before the last change (`redo`: forward again).
    pub fn undo(&mut self, redo: bool, cx: &mut Context<Self>) {
        let next = if redo { self.future.pop() } else { self.history.pop() };
        let Some(doc) = next else {
            self.set_status(if redo { "Nothing to redo" } else { "Nothing to undo" }, cx);
            return;
        };
        let before = std::mem::replace(&mut self.doc, doc);
        if redo { self.history.push(before.clone()) } else { self.future.push(before.clone()) }
        self.save_document();
        for change in diff(&before, &self.doc) {
            self.changed(&change, cx);
            cx.emit(AppEvent::Document(change));
        }
        if before.epochs != self.doc.epochs {
            self.write_epochs(&before, cx);
        }
        self.set_status(if redo { "Redone" } else { "Undone" }, cx);
    }

    /// Writes the epochs file of every recording whose epochs changed.
    fn write_epochs(&mut self, before: &SessionDoc, cx: &mut Context<Self>) {
        let of = |d: &SessionDoc, r: &Path| d.epochs.iter().filter(|e| e.recording == r).cloned().collect::<Vec<_>>();
        let mut paths: Vec<PathBuf> = before.epochs.iter().chain(&self.doc.epochs).map(|e| e.recording.clone()).collect();
        paths.sort();
        paths.dedup();
        for r in paths {
            let now = of(&self.doc, &r);
            if now == of(before, &r) || !r.exists() {
                continue;
            }
            if let Err(e) = crate::data::epochs::save(&r, &now) {
                self.set_status(format!("Epochs of {} not saved: {e}", r.display()), cx);
            }
        }
    }

    /// The events of `set`, once its table is read ([`Self::add_event_set`], or on opening the
    /// recording).
    pub fn events_of(&self, set: &crate::domain::epoch::EventSetDoc) -> Option<Arc<Vec<crate::data::events::Event>>> {
        self.event_tables.get(&set.table).cloned()
    }

    /// Reads `set`'s table (again) and adds the set to the session, or updates it.
    pub fn add_event_set(&mut self, set: crate::domain::epoch::EventSetDoc, cx: &mut Context<Self>) {
        match crate::data::events::read(&set.table) {
            Ok(events) => {
                self.event_tables.insert(set.table.clone(), Arc::new(events));
            }
            Err(e) => self.set_status(format!("Events {}: {e}", set.name), cx),
        }
        let _ = self.execute(SessionCommand::SetEventSet { set }, cx);
    }

    /// Every event of set `id` that has an offset becomes an epoch of type `response`, named after
    /// the set (one change: one undo step).
    pub fn epochs_from_events(&mut self, id: &str, cx: &mut Context<Self>) {
        let Some(set) = self.doc.event_sets.iter().find(|s| s.id == id).cloned() else { return };
        let Some(events) = self.events_of(&set) else {
            self.set_status(format!("Events {}: not read", set.name), cx);
            return;
        };
        let epochs: Vec<crate::domain::epoch::EpochDoc> = events
            .iter()
            .filter_map(|e| Some((e.onset_s, e.offset_s?)))
            .filter(|(a, b)| b > a)
            .enumerate()
            .map(|(i, (a, b))| crate::domain::epoch::EpochDoc { id: String::new(), recording: set.recording.clone(), name: format!("{} {}", set.name, i + 1), kind: "response".into(), start_sec: a, end_sec: b, notes: String::new() })
            .collect();
        let n = epochs.len();
        if self.execute(SessionCommand::SetEpochs { epochs }, cx).is_ok() {
            self.set_status(format!("{n} epochs from {}", set.name), cx);
        }
    }

    /// Moves the window to the first event of set `id` after the window's centre (wrapping to the
    /// first), and returns its onset.
    pub fn next_event(&mut self, id: &str, cx: &mut Context<Self>) -> Option<f64> {
        let set = self.doc.event_sets.iter().find(|s| s.id == id)?.clone();
        let events = self.events_of(&set)?;
        let (a, w) = (self.timeline.window_start_sec, self.timeline.visible_window_sec);
        let mid = a + w / 2.0;
        let e = events.iter().find(|e| e.onset_s > mid + 1e-9).or_else(|| events.first())?;
        let t = e.onset_s;
        self.set_window(t - w / 2.0, t + w / 2.0, cx);
        Some(t)
    }

    /// Reads the tables of the session's event sets not read yet (a session opened from a file).
    pub fn read_event_tables(&mut self) {
        for set in &self.doc.event_sets {
            if !self.event_tables.contains_key(&set.table) {
                if let Ok(events) = crate::data::events::read(&set.table) {
                    self.event_tables.insert(set.table.clone(), Arc::new(events));
                }
            }
        }
    }

    /// The event sets in `path`'s derived store that the session does not list yet.
    fn load_event_sets(&mut self, path: &Path, cx: &mut Context<Self>) {
        let Ok(dir) = std::fs::read_dir(crate::data::derived::events_dir(path)) else { return };
        let mut tables: Vec<PathBuf> = dir.filter_map(Result::ok).map(|e| e.path()).filter(|p| p.join("zarr.json").exists()).collect();
        tables.sort();
        for table in tables {
            if self.doc.event_sets.iter().any(|s| s.table == table) {
                continue;
            }
            let name = table.file_stem().map_or_else(String::new, |s| s.to_string_lossy().into_owned());
            self.add_event_set(crate::domain::epoch::EventSetDoc::new(path.to_path_buf(), name, table), cx);
        }
    }

    /// The epochs saved next to `path` that the session does not have yet.
    fn load_epochs(&mut self, path: &Path, cx: &mut Context<Self>) {
        let mut found = match crate::data::epochs::load(path) {
            Ok(found) => found,
            Err(e) => {
                self.set_status(format!("Epochs of {}: {e}", path.display()), cx);
                Vec::new()
            }
        };
        // Epochs the file holds itself (e.g. NWB intervals), from the recording's start
        let start = self.sources().map_or(0.0, |s| s.get(&s.default_entry().id.clone()).start_time_sec);
        found.extend(crate::plugin::plugins().iter().flat_map(|p| p.recording_epochs(path, start)));
        let same = |a: &crate::domain::epoch::EpochDoc, b: &crate::domain::epoch::EpochDoc| a.recording == b.recording && (a.start_sec - b.start_sec).abs() < 1e-9 && (a.end_sec - b.end_sec).abs() < 1e-9;
        let mut new: Vec<crate::domain::epoch::EpochDoc> = Vec::new();
        for mut e in found {
            if self.doc.epochs.iter().chain(&new).any(|d| same(d, &e)) {
                continue;
            }
            if self.doc.epochs.iter().chain(&new).any(|d| d.id == e.id) {
                e.id.clear();
            }
            new.push(e);
        }
        // Ids for the ones without, unique among the new ones too
        let mut next = 1;
        for e in new.iter_mut().filter(|e| e.id.is_empty()) {
            while self.doc.epochs.iter().any(|d| d.id == format!("e{next}")) {
                next += 1;
            }
            e.id = format!("e{next}");
            next += 1;
        }
        if !new.is_empty() {
            let _ = self.execute(SessionCommand::SetEpochs { epochs: new }, cx);
        }
    }

    /// Opens a floating window (one of each kind and target).
    pub fn open_floating(&mut self, f: Floating, cx: &mut Context<Self>) {
        self.floating.retain(|x| std::mem::discriminant(x) != std::mem::discriminant(&f) || matches!((x, &f), (Floating::Metadata { path: a, .. }, Floating::Metadata { path: b, .. }) if a != b));
        self.floating.push(f);
        cx.emit(AppEvent::Floating);
        cx.notify();
    }

    pub fn close_floating(&mut self, index: usize, cx: &mut Context<Self>) {
        if index < self.floating.len() {
            self.floating.remove(index);
            cx.emit(AppEvent::Floating);
            cx.notify();
        }
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

    /// The session view showing `source` of the open recording with all its channels (made when
    /// none does; it becomes the active view then); `None` for a recording without a file.
    pub fn view_for_source(&mut self, source: &str, cx: &mut Context<Self>) -> Option<String> {
        let rec = self.recording.as_ref()?;
        let path = rec.path.clone()?;
        let default = rec.sources.default_entry().id.clone();
        let shows = |v: &ViewDoc| v.recording.path == path && v.recording.source.as_deref().unwrap_or(&default) == source && v.recording.channels.is_none();
        if let Some(v) = self.doc.views.iter().find(|v| shows(v)) {
            return Some(v.id.clone());
        }
        let recording = RecordingRef { path: path.clone(), source: Some(source.to_string()), channels: None };
        self.execute(SessionCommand::OpenView { recording, name: None }, cx).ok()?;
        let id = self.doc.views.iter().find(|v| shows(v)).map(|v| v.id.clone());
        if let Some(id) = &id {
            self.rebuild_preview(id, cx);
        }
        id
    }

    /// Closes session view `id` when it holds no work: no ops (its source feeds it directly),
    /// all channels, its default name, no plugin data. Views with work stay (the Pipeline shows them).
    pub fn close_view_if_untouched(&mut self, id: &str, cx: &mut Context<Self>) {
        let Some(v) = self.doc.view(id) else { return };
        let direct = self.doc.graph.input(id).and_then(|i| self.doc.graph.node(i)).is_some_and(|n| matches!(n.kind, crate::domain::graph::NodeKind::Source { .. }));
        let untouched = direct && v.ops.is_empty() && v.recording.channels.is_none() && v.name == crate::domain::session::view_name(&v.recording) && v.plugins.is_empty();
        if untouched {
            let _ = self.execute(SessionCommand::CloseView { view: id.to_string() }, cx);
        }
    }

    /// What session view `view` draws when it shows its processed preview (it has enabled ops and
    /// `Show::Processed`); `None`: the recording as stored.
    pub fn preview_of(&self, view: &str) -> Option<Arc<dyn SignalBackend>> {
        let v = self.doc.view(view)?;
        if v.display.show != Show::Processed {
            return None;
        }
        self.previews.get(view).map(|p| p.clone() as Arc<dyn SignalBackend>)
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
        let plan = match self.view_plan(&v, ds.total_channels) {
            Ok(p) => p,
            Err(e) => {
                self.set_status(format!("{}: {e}", v.name), cx);
                return;
            }
        };
        if plan.steps.is_empty() {
            return;
        }
        let group = plan.group;
        let mut stages: Vec<PipelineStage> = plan.steps.iter().map(|(_, st)| st.clone()).collect();
        // The chain's key, step by step: [source and channels, step 1, step 2, …]
        let mut key = vec![format!("{source}|{group:?}")];
        key.extend(plan.steps.into_iter().map(|(k, _)| k));
        // Whitenings fitted on the signal of the steps before them, as the pipeline does. A fit
        // reads the recording: it runs in the background (the view draws the raw signal
        // meanwhile) and the preview is built again when it is done
        for (at, fit) in plan.fits {
            let up_to = key[..at + 2].join("\n");
            let Some(fitted) = self.whiten_cache.get(&up_to).cloned() else {
                if self.whiten_fitting.insert(up_to.clone()) {
                    self.set_status(format!("{}: fitting the whitening…", v.name), cx);
                    let (before, group, raw, view, name) = (stages[..at].to_vec(), group.clone(), raw.clone(), id.to_string(), v.name.clone());
                    let task = cx.background_spawn(async move {
                        let target = dspk_core::ComputeTarget::from_env().map_err(|e| e.to_string())?;
                        dspk_pipeline::fit_whitening(target, raw.as_ref(), &group, &before, &fit).map_err(|e| e.to_string())
                    });
                    cx.spawn(async move |this, cx| {
                        let result = task.await;
                        let _ = this.update(cx, |s, cx| {
                            s.whiten_fitting.remove(&up_to);
                            match result {
                                Ok(w) => {
                                    s.whiten_cache.insert(up_to, w);
                                    s.set_status(format!("{name}: whitening fitted"), cx);
                                    s.rebuild_preview(&view, cx);
                                    cx.emit(AppEvent::WindowMoved);
                                }
                                Err(e) => s.set_status(format!("{name}: whiten: {e}"), cx),
                            }
                        });
                    })
                    .detach();
                }
                return;
            };
            stages[at] = PipelineStage::SpatialWhitening(fitted);
        }
        // The longest computed prefix of this chain
        let found = (2..=key.len()).rev().find_map(|k| self.preview_cache.iter().find(|(c, _)| c[..] == key[..k]).map(|(_, p)| (k, p.clone())));
        let built = match found {
            Some((k, p)) if k == key.len() => Ok(p),
            // A branch reads the shared prefix's rows, in the order that prefix leaves them
            Some((k, p)) => ProcessedSignal::new(ds.signal().clone(), Arc::new(PreviewSource::new(p.clone())), p.out_channels().to_vec(), stages[k - 1..].to_vec()).map(Arc::new),
            None => ProcessedSignal::new(ds.signal().clone(), raw, group, stages).map(Arc::new),
        };
        match built {
            Ok(p) => {
                self.preview_cache.retain(|(c, _)| *c != key);
                self.preview_cache.push((key, p.clone()));
                if self.preview_cache.len() > PREVIEW_CACHE {
                    self.preview_cache.remove(0);
                }
                self.previews.insert(id.to_string(), p);
            }
            Err(e) => self.set_status(format!("{}: {e}", v.name), cx),
        }
    }

    /// How view `v` is previewed, as the pipeline runs it: `group`, the channels it reads (its
    /// subset, or all `total`, through the remaps that open its chain: they fold into the read
    /// and may drop); `steps`, its processing on the device with each step's key (a remap between
    /// processing steps reorders there and keeps every channel); `shown`, the channels it ends
    /// with (through the remaps after the last processing step: they may drop). Resampling is not
    /// previewed.
    pub fn view_plan(&self, v: &ViewDoc, total: usize) -> Result<ViewPlan, String> {
        let mut group: Vec<usize> = v.recording.channels.clone().unwrap_or_else(|| (0..total).collect());
        let base_dir = self.doc_path.as_ref().and_then(|p| p.parent().map(Path::to_path_buf)).unwrap_or_default();
        let ops: Vec<&dspk_pipeline::Node> = v.ops.iter().filter(|o| o.enabled).map(|o| &o.node).collect();
        let is_remap = |n: &dspk_pipeline::Node| matches!(n, dspk_pipeline::Node::Remap { .. });
        let map_of = |n: &dspk_pipeline::Node| n.channel_map(Some(&base_dir)).map_err(|e| e.to_string()).map(|m| m.expect("a remap has a map"));
        let lead = ops.iter().take_while(|n| is_remap(n)).count();
        for n in &ops[..lead] {
            let order = map_of(n)?.resolve(group.len()).map_err(|e| e.to_string())?;
            group = order.into_iter().map(|i| group[i]).collect();
        }
        let last_processing = ops.iter().rposition(|n| !is_remap(n));
        let mut steps = Vec::new();
        let mut fits = Vec::new();
        let mut current = group.clone();
        for (i, n) in ops.iter().enumerate().skip(lead) {
            let key = serde_json::to_string(n).unwrap_or_default();
            if is_remap(n) {
                let order = map_of(n)?.resolve(current.len()).map_err(|e| e.to_string())?;
                if last_processing.is_some_and(|last| i > last) {
                    // After the last processing step: the channels shown (may drop)
                    current = order.into_iter().map(|j| current[j]).collect();
                    continue;
                }
                let mut seen = order.clone();
                seen.sort_unstable();
                if seen != (0..current.len()).collect::<Vec<_>>() {
                    return Err(format!("a remap between processing steps keeps every channel ({}; it reorders them): drop channels in a remap first or last", current.len()));
                }
                current = order.iter().map(|&j| current[j]).collect();
                steps.push((key, PipelineStage::ReorderChannels { order }));
            } else if let Some(fit) = n.whiten_fit() {
                n.stage().map_err(|e| e.to_string())?;
                fits.push((steps.len(), fit));
                steps.push((key, PipelineStage::SpatialWhitening(dspk_pipeline::unfitted_whitening(current.len()))));
            } else if let Some(stage) = n.stage().map_err(|e| e.to_string())? {
                // A grid's operator is made for its electrodes
                if let PipelineStage::SurfaceLaplacian(lap) = &stage
                    && lap.num_channels != current.len()
                {
                    return Err(format!("the Laplacian's grid has {} electrodes; the view has {} channels", lap.num_channels, current.len()));
                }
                steps.push((key, stage));
            }
        }
        Ok(ViewPlan { group, steps, shown: current, fits })
    }

    /// The channels view `v` shows, in order (see [`Self::view_plan`]).
    pub fn view_channels(&self, v: &ViewDoc, total: usize) -> Result<Vec<usize>, String> {
        self.view_plan(v, total).map(|p| p.shown)
    }

    /// The active view's channels (its subset, through its remaps) become the shown
    /// selection of its source (once per change of them, so the user's own selection is kept
    /// otherwise).
    fn apply_active_view(&mut self, cx: &mut Context<Self>) {
        let Some(v) = self.doc.active().cloned() else { return };
        let source = self.view_source(&v);
        if !self.on_open_recording(&v, &source) {
            return;
        }
        let total = self.sources().map_or(0, |ss| ss.entry(&source).channels);
        let channels = match self.view_channels(&v, total) {
            Ok(c) => c,
            Err(e) => {
                self.set_status(format!("{}: {e}", v.name), cx);
                v.recording.channels.clone().unwrap_or_else(|| (0..total).collect())
            }
        };
        let subset = (channels.len() != total || channels.iter().enumerate().any(|(i, &c)| i != c)).then_some(channels);
        let key = (v.id.clone(), subset.clone());
        if self.applied_view.as_ref() == Some(&key) {
            return;
        }
        self.applied_view = Some(key);
        match subset {
            Some(ch) => self.set_selection(&source, ch, cx),
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
                // A remap at the head of the chain changes the channels shown
                if self.doc.active_view.as_deref() == Some(id.as_str()) {
                    self.apply_active_view(cx);
                }
                if self.picked_op.as_ref().is_some_and(|op| self.doc.view(id).is_some_and(|v| v.op(op).is_none())) {
                    self.picked_op = None;
                }
                cx.emit(AppEvent::WindowMoved);
            }
            Changed::Display(_) | Changed::Plugin(..) => cx.emit(AppEvent::WindowMoved),
            Changed::Graph | Changed::Timeline => {}
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
    /// Starts a job ([`Self::start_job`]) that works on graph nodes `nodes`: they show running on
    /// the Pipeline canvas until it ends, then done or failed. `false` (nothing marked) when
    /// another job is running.
    pub fn start_job_on(
        &mut self,
        nodes: impl IntoIterator<Item = String>,
        label: impl Into<String>,
        start: impl FnOnce(Box<dyn Fn(JobMessage) + Send + Sync>) -> Cancel,
        done: impl FnOnce(&mut Self, PathBuf, &mut Context<Self>) + 'static,
        cx: &mut Context<Self>,
    ) -> bool {
        if !self.start_job(label, start, done, cx) {
            return false;
        }
        self.mark_job_nodes(nodes, cx);
        true
    }

    /// The running job's graph nodes (only from [`Self::start_job_on`], right after it starts).
    fn mark_job_nodes(&mut self, ids: impl IntoIterator<Item = String>, cx: &mut Context<Self>) {
        let since = Instant::now();
        self.job_nodes = ids.into_iter().collect();
        for id in &self.job_nodes {
            self.node_status.insert(id.clone(), NodeRun::Running { line: String::new(), fraction: None, since });
        }
        cx.emit(AppEvent::JobChanged);
    }

    /// The job's nodes: done (with how long it took) or failed.
    fn finish_job_nodes(&mut self, failed: Option<String>) {
        for id in std::mem::take(&mut self.job_nodes) {
            let status = match (&failed, self.node_status.get(&id)) {
                (Some(e), _) => NodeRun::Failed(e.clone()),
                (None, Some(NodeRun::Running { since, .. })) => NodeRun::Done { took: since.elapsed() },
                (None, _) => NodeRun::Done { took: Duration::ZERO },
            };
            self.node_status.insert(id, status);
        }
    }

    /// The nodes a write of `id` runs: `id` and everything upstream of it.
    fn chain_nodes(&self, id: &str) -> Vec<String> {
        let mut ids: Vec<String> = self.doc.graph.ancestors(id).into_iter().collect();
        ids.push(id.to_string());
        ids
    }

    pub fn start_job(
        &mut self,
        label: impl Into<String>,
        start: impl FnOnce(Box<dyn Fn(JobMessage) + Send + Sync>) -> Cancel,
        done: impl FnOnce(&mut Self, PathBuf, &mut Context<Self>) + 'static,
        cx: &mut Context<Self>,
    ) -> bool {
        if self.job.is_some() {
            self.set_status("Another job is running: wait for it or cancel it", cx);
            return false;
        }
        let label = label.into();
        self.last_failed = None;
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
                            for id in s.job_nodes.clone() {
                                if let Some(NodeRun::Running { line: l, fraction: f, .. }) = s.node_status.get_mut(&id) {
                                    *l = line.clone();
                                    *f = fraction.or(*f);
                                }
                            }
                            s.status = format!("{label}: {line}");
                            cx.emit(AppEvent::Status);
                        }
                        JobMessage::Done(path) => {
                            s.job = None;
                            s.finish_job_nodes(None);
                            s.set_status(format!("{label}: done ({})", file_label(&path)), cx);
                            if let Some(done) = done.take() {
                                done(s, path, cx);
                            }
                        }
                        JobMessage::Failed(e) => {
                            s.job = None;
                            s.last_failed = Some((label.clone(), e.clone()));
                            s.finish_job_nodes(Some(e.clone()));
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
        true
    }

    pub fn cancel_job(&mut self, cx: &mut Context<Self>) {
        if let Some(j) = &self.job {
            j.cancel.cancel();
            self.set_status(format!("Cancelling {}…", j.label), cx);
        }
    }

    /// Writes the chain of graph node `id` (a write or view node) to
    /// `<recording>.<id>.processed.zarr` next to its recording.
    pub fn write_node(&mut self, id: &str, cx: &mut Context<Self>) {
        let Some((recording, ops)) = self.doc.graph.chain(id, &self.doc.recipes) else {
            self.set_status(format!("{id} is not connected to a recording"), cx);
            return;
        };
        let v = ViewDoc { id: id.to_string(), name: id.to_string(), recording, ops, display: Default::default(), plugins: Default::default() };
        self.write_view(v, cx);
    }

    /// The write nodes fed by source node `id` (with their chains).
    pub fn writes_of(&self, id: &str) -> Vec<String> {
        let g = &self.doc.graph;
        g.descendants(id).into_iter().filter(|d| matches!(g.node(d).map(|n| &n.kind), Some(crate::domain::graph::NodeKind::Write {}))).collect()
    }

    /// Writes every output of source node `id` in one read of its recording.
    pub fn write_source(&mut self, id: &str, cx: &mut Context<Self>) {
        let writes = self.writes_of(id);
        if writes.len() < 2 {
            if let Some(w) = writes.first().cloned() {
                self.write_node(&w, cx);
            }
            return;
        }
        let mut chains = Vec::new();
        let mut outputs = Vec::new();
        for w in &writes {
            let Some((recording, ops)) = self.doc.graph.chain(w, &self.doc.recipes) else { continue };
            let v = ViewDoc { id: w.clone(), name: w.clone(), recording: recording.clone(), ops: ops.clone(), display: Default::default(), plugins: Default::default() };
            if jobs::gives_events(&v) {
                // Events are found on their own (a branched run writes signals)
                self.set_status(format!("{w} gives events: run it on its own (double-click it)"), cx);
                continue;
            }
            outputs.push(jobs::processed_path(&recording.path, w));
            chains.push(ViewDoc { id: w.clone(), name: w.clone(), recording, ops, display: Default::default(), plugins: Default::default() });
        }
        let n = chains.len();
        let nodes: std::collections::BTreeSet<String> = writes.iter().flat_map(|w| self.chain_nodes(w)).collect();
        self.start_job_on(
            nodes,
            format!("Write {n} outputs"),
            move |send| jobs::write_branches(chains, outputs, move |m| send(m)),
            move |s, _, cx| s.set_status(format!("Written: {n} outputs in one read"), cx),
            cx,
        );
    }

    /// Shows view `id` in Explore, opening its recording when another one is open.
    pub fn show_view(&mut self, id: &str, cx: &mut Context<Self>) {
        let Some(v) = self.doc.view(id).cloned() else {
            self.set_status(format!("{id} is not connected to a recording yet"), cx);
            return;
        };
        let _ = self.execute(SessionCommand::SelectView { view: id.to_string() }, cx);
        let open = self.recording.as_ref().and_then(|r| r.path.clone());
        if open.as_deref() != Some(v.recording.path.as_path()) {
            self.open(v.recording.path.clone(), cx);
        }
        self.set_workspace(Workspace::Explore, cx);
        if v.display.layout == crate::domain::session::ViewLayout::Epochs {
            cx.emit(AppEvent::ShowEpochs(id.to_string()));
        }
    }

    /// The event set a Write of `view`'s chain saved (the same recording and ops), if any.
    pub fn saved_events_of(&self, view: &ViewDoc) -> Option<&crate::domain::epoch::EventSetDoc> {
        let g = &self.doc.graph;
        g.nodes.iter().filter(|n| matches!(n.kind, crate::domain::graph::NodeKind::Write {})).find_map(|w| {
            let (recording, ops) = g.chain(&w.id, &self.doc.recipes)?;
            if recording.path != view.recording.path || ops != view.ops {
                return None;
            }
            let table = crate::data::derived::events_group(&recording.path, &w.id);
            self.doc.event_sets.iter().find(|s| s.table == table)
        })
    }

    /// Writes the active view (its recording through its enabled ops) to
    /// `<recording>.<view>.processed.zarr` next to the recording.
    pub fn write_active_view(&mut self, cx: &mut Context<Self>) {
        let Some(v) = self.doc.active().cloned() else { return };
        self.write_view(v, cx);
    }

    fn write_view(&mut self, v: ViewDoc, cx: &mut Context<Self>) {
        if !v.ops.iter().any(|o| o.enabled) {
            self.set_status(format!("{}: no enabled ops to write", v.name), cx);
            return;
        }
        if jobs::gives_events(&v) {
            return self.write_events(v, cx);
        }
        let output = jobs::processed_path(&v.recording.path, &v.id);
        let nodes = self.chain_nodes(&v.id);
        self.start_job_on(
            nodes,
            format!("Write {}", v.name),
            move |send| jobs::write_view(&v, output, move |m| send(m)),
            |s, path, cx| s.set_status(format!("Written: {}", path.display()), cx),
            cx,
        );
    }

    /// Finds the events of `v` (a chain ending with Crossings) into the recording's derived store,
    /// `events/<write id>`, and adds them to the session as an event set (ticks on its traces).
    fn write_events(&mut self, v: ViewDoc, cx: &mut Context<Self>) {
        let recording = v.recording.path.clone();
        if let Err(e) = crate::data::derived::ensure(&recording, &["/events"]) {
            self.set_status(format!("Events of {}: {e}", recording.display()), cx);
            return;
        }
        let output = crate::data::derived::events_group(&recording, &v.id);
        let nodes = self.chain_nodes(&v.id);
        let name = v.name.clone();
        self.start_job_on(
            nodes,
            format!("Events {}", v.name),
            move |send| jobs::write_events(&v, output, move |m| send(m)),
            move |s, path, cx| {
                // The set keeps its look when it is found again
                let set = match s.doc.event_sets.iter().find(|e| e.table == path) {
                    Some(e) => e.clone(),
                    None => crate::domain::epoch::EventSetDoc::new(recording.clone(), name.clone(), path.clone()),
                };
                s.add_event_set(set, cx);
                let n = s.doc.event_sets.iter().find(|e| e.table == path).and_then(|e| s.events_of(e)).map_or(0, |e| e.len());
                s.set_status(format!("{n} events: {}", path.display()), cx);
            },
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

    /// The servers the user works with (Settings), saved.
    pub fn set_servers(&mut self, servers: Vec<crate::data::remote::ServerProfile>, cx: &mut Context<Self>) {
        crate::data::remote::set_profiles(&servers);
        self.session.servers = servers;
        self.save_session();
        cx.notify();
    }

    /// Where results fetched from servers go (`None`: the default), saved.
    pub fn set_results_dir(&mut self, dir: Option<std::path::PathBuf>, cx: &mut Context<Self>) {
        self.session.results_dir = dir;
        self.save_session();
        cx.notify();
    }

    /// Whether the last recording opens at start, saved.
    pub fn set_reopen_last(&mut self, on: bool, cx: &mut Context<Self>) {
        self.session.reopen_last = on;
        self.save_session();
        cx.notify();
    }

    /// Forgets `path` from the recent lists (`None`: all of them), saved. The files stay.
    pub fn forget_recent(&mut self, path: Option<&std::path::Path>, cx: &mut Context<Self>) {
        match path {
            Some(p) => self.session.forget_recent(p),
            None => self.session.clear_recent(),
        }
        self.save_session();
        cx.notify();
    }

    /// The compute runtime for the next start (`None`: the first compiled in), saved.
    pub fn set_runtime(&mut self, runtime: Option<String>, cx: &mut Context<Self>) {
        self.session.runtime = runtime;
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

    /// At the first launch, says how to avoid the slow first run.
    pub fn suggest_tuning_once(&mut self, cx: &mut Context<Self>) {
        if !self.session.tune_suggested {
            self.session.tune_suggested = true;
            self.save_session();
            self.set_status("First run on this device? File → Tune this device makes first runs and previews faster", cx);
        }
    }

    fn save_session(&mut self) {
        self.session.workspace = self.workspace;
        if let Err(e) = self.session.save(self.session_path.as_deref()) {
            tracing::warn!("could not save the session: {e}");
        }
    }

    /// Remembers whether Explore shows the Pipeline docked.
    pub fn set_pipeline_dock(&mut self, on: bool) {
        if self.session.explore.pipeline_dock != on {
            self.session.explore.pipeline_dock = on;
            self.save_session();
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
    /// A resizable panel's size (`default` until the user drags it).
    pub fn panel_width(&self, key: &str, default: f32) -> f32 {
        self.session.panel_widths.get(key).copied().unwrap_or(default)
    }

    /// Whether a side panel shows (`true` until the user hides it).
    pub fn panel_open(&self, key: &str) -> bool {
        self.session.panel_open.get(key).copied().unwrap_or(true)
    }

    pub fn set_panel_open(&mut self, key: &str, open: bool, cx: &mut Context<Self>) {
        self.session.panel_open.insert(key.to_string(), open);
        self.save_session();
        cx.notify();
    }

    pub fn set_panel_width(&mut self, key: &str, width: f32, cx: &mut Context<Self>) {
        if self.session.panel_widths.get(key) != Some(&width) {
            self.session.panel_widths.insert(key.to_string(), width);
            self.save_session();
            cx.notify();
        }
    }

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
        if self.recording.as_ref().is_some_and(|r| r.path.as_ref() == Some(&path)) {
            return;
        }
        if let Some(at) = self.loaded.iter().position(|r| r.path.as_ref() == Some(&path)) {
            let recording = self.loaded.remove(at);
            self.install(recording, cx);
            return;
        }
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
                        store.load_epochs(&path, cx);
                        store.load_event_sets(&path, cx);
                        store.read_event_tables();
                    }
                    Err(e) => store.set_status(format!("Could not open {name}: {e:#}"), cx),
                }
            });
        })
        .detach();
    }

    /// Loads the recording at `path` beside the open one (it stays the active one), for views of
    /// both; [`AppEvent::RecordingLoaded`] when it is ready (at once when already loaded).
    pub fn load_beside(&mut self, path: PathBuf, cx: &mut Context<Self>) {
        if self.recordings().any(|r| r.path.as_ref() == Some(&path)) {
            cx.emit(AppEvent::RecordingLoaded(path));
            return;
        }
        let name = path.file_name().map_or_else(|| path.display().to_string(), |n| n.to_string_lossy().into_owned());
        self.set_status(format!("Loading {name}…"), cx);
        let task = cx.background_spawn({
            let path = path.clone();
            async move { SourceSet::open(&path) }
        });
        cx.spawn(async move |this, cx| {
            let result = task.await;
            let _ = this.update(cx, |store, cx| match result {
                Ok(sources) => {
                    store.loaded.retain(|r| r.path.as_ref() != Some(&path));
                    store.loaded.push(Recording::new(sources, Some(path.clone())));
                    store.set_status(format!("Loaded {name} beside the open recording"), cx);
                    cx.emit(AppEvent::RecordingLoaded(path));
                }
                Err(e) => store.set_status(format!("Could not load {name}: {e:#}"), cx),
            });
        })
        .detach();
    }

    /// The sources of the loaded recording at `path` (the open one, or one loaded beside it).
    pub fn sources_of(&self, path: &Path) -> Option<&Arc<SourceSet>> {
        self.recordings().find(|r| r.path.as_deref() == Some(path)).map(|r| &r.sources)
    }

    /// Seconds to add to the open recording's time to get `path`'s: the difference of their
    /// placements on the session's time (0 when either is not placed).
    pub fn time_offset_of(&self, path: &Path) -> f64 {
        let open = self.recording.as_ref().and_then(|r| r.path.clone());
        let at = |p: &Path| self.doc.placements.iter().find(|pl| pl.path == p).map(|pl| pl.start_sec);
        match (open.as_deref().and_then(at), at(path)) {
            (Some(a), Some(b)) => a - b,
            _ => 0.0,
        }
    }

    /// Opens the recording at `path` on a server (a `dspk://` path: drawn from the server's
    /// pyramid, read there, nothing copied whole), like any recording: a view, recent files.
    pub fn open_remote(&mut self, profile: crate::data::remote::ServerProfile, path: String, cx: &mut Context<Self>) {
        crate::data::remote::set_profiles(&self.session.servers);
        self.open(crate::data::remote::uri(&profile.name, &path), cx);
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
        self.preview_cache.clear();
        self.whiten_cache.clear();
        self.applied_view = None;
        self.picked_channel = None;
        self.picked_op = None;
        self.status = format!("Opened {} ({})", recording.name, recording.summary());
        // The one it replaces stays loaded (files only: a synthetic one is gone once replaced)
        if let Some(old) = self.recording.take().filter(|r| r.path.is_some() && r.path != recording.path) {
            self.loaded.retain(|r| r.path != old.path);
            self.loaded.push(old);
        }
        self.loaded.retain(|r| r.path.is_none() || r.path != recording.path);
        self.recording = Some(recording);
        self.save_session();
        cx.emit(AppEvent::RecordingChanged);
        cx.emit(AppEvent::Status);
    }

    /// Shows `source` of the recording at `path` in the Inspector (`None`: back to the view).
    pub fn inspect(&mut self, what: Option<(PathBuf, String)>, cx: &mut Context<Self>) {
        if self.inspected != what {
            let show = what.is_some();
            self.inspected = what;
            cx.emit(if show { AppEvent::Picked } else { AppEvent::Status });
            cx.notify();
        }
    }

    /// Every loaded recording, the active one first.
    pub fn recordings(&self) -> impl Iterator<Item = &Recording> {
        self.recording.iter().chain(self.loaded.iter())
    }

    /// Unloads the recording at `path` (`None`: the active one). Its views stay in the session
    /// document; closing the active one switches to the last loaded, or to no recording.
    pub fn close_recording(&mut self, path: Option<&Path>, cx: &mut Context<Self>) {
        let active = self.recording.as_ref().is_some_and(|r| path.is_none() || r.path.as_deref() == path);
        if !active {
            self.loaded.retain(|r| r.path.as_deref() != path);
            cx.emit(AppEvent::RecordingChanged);
            return;
        }
        let closed = self.recording.take().map(|r| r.name).unwrap_or_default();
        if let Some(next) = self.loaded.pop() {
            self.install(next, cx);
        } else {
            self.stop_clock();
            self.timeline = TimelineState::new(1.0);
            self.selection.clear();
            self.pyramids.clear();
            self.previews.clear();
            self.preview_cache.clear();
        self.whiten_cache.clear();
            self.applied_view = None;
            self.picked_channel = None;
            self.picked_op = None;
            cx.emit(AppEvent::RecordingChanged);
        }
        self.set_status(format!("Closed {closed}"), cx);
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
    let mut out = diff_views(before, after);
    if before.graph != after.graph || before.recipes != after.recipes {
        out.push(Changed::Graph);
    }
    if before.epochs != after.epochs || before.placements != after.placements {
        out.push(Changed::Timeline);
    }
    out
}

fn diff_views(before: &SessionDoc, after: &SessionDoc) -> Vec<Changed> {
    let ids = |d: &SessionDoc| d.views.iter().map(|v| (v.id.clone(), v.name.clone(), v.recording.clone())).collect::<Vec<_>>();
    if ids(before) != ids(after) || before.active_view != after.active_view {
        // Chains of the views kept may have changed too
        let mut out = vec![Changed::Views];
        for b in &after.views {
            if before.view(&b.id).is_some_and(|a| a.ops != b.ops) {
                out.push(Changed::Ops(b.id.clone()));
            }
        }
        return out;
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
