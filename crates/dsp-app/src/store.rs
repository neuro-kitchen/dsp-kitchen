//! The app's shared state, as one entity: the open recording, the timeline every view follows,
//! the channel selection of each source, the workspace shown, recent files and the status line.
//! Changes go through its methods, which emit the [`AppEvent`] they caused (never a blanket
//! `notify`); view models subscribe to the events they show.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use gpui_kit::{AppContext as _, Context, EventEmitter, Task};

use crate::engine::data::summarize::Progress;
use crate::engine::data::{Dataset, SourceSet, SpikeEventStore};
use crate::engine::palette::Palette;
use crate::engine::time::timeline::TimelineState;
use crate::engine::time::view::parse_channel_ranges;
use crate::session::Session;
use crate::workspace::Workspace;

/// Playback clock period.
const TICK: Duration = Duration::from_millis(16);

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
    /// More of a source's min/max summary is ready (zoomed-out views of it redraw).
    SummaryProgress(String),
    /// The status line or the recent list changed.
    Status,
}

/// The open recording.
pub struct Recording {
    pub sources: Arc<SourceSet>,
    pub name: String,
    /// Spike events shown on traces and in the timeline overview (empty until an extraction runs).
    pub events: Arc<SpikeEventStore>,
}

impl Recording {
    pub fn new(sources: SourceSet, path: Option<PathBuf>) -> Self {
        let name = match &path {
            Some(p) => p.file_name().map_or_else(|| p.display().to_string(), |n| n.to_string_lossy().into_owned()),
            None => sources.default_entry().name.clone(),
        };
        Self { sources: Arc::new(sources), name, events: Arc::new(SpikeEventStore::default()) }
    }

    /// `32 ch · 30.0 kHz · 5 min 00 s` of the default source.
    pub fn summary(&self) -> String {
        let e = self.sources.default_entry();
        let rate = if e.sample_rate >= 1000.0 { format!("{:.1} kHz", e.sample_rate / 1000.0) } else { format!("{:.1} Hz", e.sample_rate) };
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
    /// Summary progress per source id: (samples summarized, of all).
    pub summaries: HashMap<String, (u64, u64)>,
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
            summaries: HashMap::new(),
        }
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

    pub fn set_explore_views(&mut self, views: Vec<crate::session::SavedTimeView>) {
        if self.session.explore.views != views {
            self.session.explore.views = views;
            self.save_session();
        }
    }

    pub fn push_recent_sorting(&mut self, path: &Path) {
        self.session.push_recent_sorting(path);
        self.save_session();
    }

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
                        store.install(Recording::new(sources, Some(path)), cx);
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
        self.summaries.clear();
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

    /// Summarizes `source` in the background (once per recording), nearest `focus_sec` first;
    /// later calls move the focus to where the views look now.
    pub fn summarize(&mut self, source: &str, focus_sec: f64, cx: &mut Context<Self>) {
        let Some(sources) = self.sources().cloned() else { return };
        let ds = sources.get(source);
        let focus = ((focus_sec - ds.start_time_sec) * ds.sample_rate).max(0.0) as u64;
        if self.summaries.contains_key(source) {
            ds.summarize(focus, Arc::new(|_| {}));
            return;
        }
        if ds.lod().is_some() {
            return;
        }
        self.summaries.insert(source.to_string(), (0, ds.total_samples as u64));
        let (tx, rx) = async_channel::unbounded::<Progress>();
        let id = source.to_string();
        cx.spawn(async move |this, cx| {
            while let Ok(p) = rx.recv().await {
                let alive = this.update(cx, |s, cx| {
                    if let Some(entry) = s.summaries.get_mut(&id) {
                        *entry = (p.done, p.total);
                        cx.emit(AppEvent::SummaryProgress(id.clone()));
                    }
                });
                if alive.is_err() {
                    break;
                }
            }
        })
        .detach();
        ds.summarize(
            focus,
            Arc::new(move |p| {
                let _ = tx.try_send(p);
            }),
        );
    }

    /// `Summarizing 12 / 45 s` while the default source's summary fills.
    pub fn summary_label(&self) -> Option<String> {
        let sources = self.sources()?;
        let e = sources.default_entry();
        let &(done, total) = self.summaries.get(&e.id)?;
        (done < total && e.sample_rate > 0.0).then(|| format!("Summarizing {:.0} / {:.0} s", done as f64 / e.sample_rate, total as f64 / e.sample_rate))
    }

    /// Builds the min/max cache files of the recording's sources on background threads; views
    /// use them from their next frame on (zoomed-out windows then read a few levels, not every
    /// sample).
    pub fn build_caches(&mut self, cx: &mut Context<Self>) {
        let Some(r) = &self.recording else { return };
        r.sources.build_caches();
        self.set_status("Building min/max caches in the background…", cx);
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
