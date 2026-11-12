//! One time view (traces or heatmap): its settings ([`TimeView`]), the frame on screen and the
//! window it was drawn for, and the hover readout.
//!
//! Rendering is asked for on change, never polled: a timeline, selection, size or setting change
//! schedules one request, sent once the current update ends (so a burst of changes costs one
//! request), and only while the view is displayed. Frames come back over this view's own channel;
//! the image they replace is released one frame later (it may still be on screen).

use std::sync::Arc;
use std::time::{Duration, Instant};

use gpui_kit::{App, Context, Entity, EventEmitter, RenderImage, Subscription, Task};

use crate::data::recording::SourceSet;
use crate::data::work_pool::JobKey;
use crate::screens::explore::trace::hover::{HoverReply, HoverRequest};
use crate::screens::explore::trace::renderer::{render_on_worker, TimeViewKind};
use crate::screens::explore::trace::view::{HoverTarget, TimeView, ViewId};
use crate::navigation::route::Workspace;
use crate::data::store::{AppEvent, Store};

use crate::data::services::{Services, RENDER_SLOT};

/// The frame on screen and what it shows.
#[derive(Clone)]
pub struct Shown {
    pub image: Arc<RenderImage>,
    /// Time window it was drawn for (seconds).
    pub start: f64,
    pub window: f64,
}

struct Delivered {
    shown: Shown,
    scale: Option<f32>,
    /// Every column was available (else redraw as the pyramid builds).
    complete: bool,
    elapsed: Duration,
    samples_per_px: f64,
}

pub enum TraceEvent {
    /// The view was pressed: it becomes the focused view.
    Pressed,
    /// Kind, source or title changed (tab title, settings panel).
    Changed,
    /// The view no longer shows this session view (its source changed).
    Released(String),
}

pub struct TraceVm {
    store: Entity<Store>,
    pub view: TimeView,
    pub shown: Option<Shown>,
    /// Replaced image, released when the next frame arrives.
    retired: Option<Arc<RenderImage>>,
    /// Displayed in its tab group (hidden views do not render).
    active: bool,
    scheduled: bool,
    /// The frame on screen has columns the pyramid had not built yet.
    incomplete: bool,
    pub hover: String,
    frames: async_channel::Sender<Delivered>,
    hover_reply: HoverReply,
    _tasks: Vec<Task<()>>,
    _store: Subscription,
    home: Workspace,
    /// Live events of the window (the chain ends with Crossings), what they were found for, and a
    /// detection on its way.
    pub live: Option<(crate::data::live::LiveKey, Arc<Vec<crate::data::events::Event>>)>,
    live_pending: Option<crate::data::live::LiveKey>,
    /// Why no live events show (e.g. the window is too wide).
    pub live_note: Option<String>,
}

impl EventEmitter<TraceEvent> for TraceVm {}

impl TraceVm {
    /// A view of `kind` on `source` (its id in the open file).
    pub fn new(store: Entity<Store>, id: ViewId, kind: TimeViewKind, source: &str, cx: &mut Context<Self>) -> Self {
        let mut view = TimeView::new(id, kind, Vec::new());
        Self::attach(&mut view, &store, source, cx);
        let (frames, frames_rx) = async_channel::unbounded::<Delivered>();
        let (hover_tx, hover_rx) = async_channel::unbounded::<(u64, String)>();
        let hover_reply: HoverReply = Arc::new(move |seq, text| {
            let _ = hover_tx.try_send((seq, text));
        });
        let frame_task = cx.spawn(async move |this, cx| {
            while let Ok(d) = frames_rx.recv().await {
                if this.update(cx, |vm, cx| vm.on_frame(d, cx)).is_err() {
                    break;
                }
            }
        });
        let hover_task = cx.spawn(async move |this, cx| {
            while let Ok((seq, text)) = hover_rx.recv().await {
                let alive = this.update(cx, |vm, cx| {
                    if vm.view.hover_seq == seq {
                        vm.hover = text;
                        cx.notify();
                    }
                });
                if alive.is_err() {
                    break;
                }
            }
        });
        let sub = cx.subscribe(&store, |vm, store, event, cx| match event {
            AppEvent::WindowMoved | AppEvent::PaletteChanged => vm.request_render(cx),
            // Back on its workspace: draw what changed while another one was shown
            AppEvent::WorkspaceChanged if vm.view.needs_render => vm.request_render(cx),
            AppEvent::RecordingChanged if vm.home != Workspace::Explore => {
                let source = store.read(cx).sources().map(|s| s.default_entry().id.clone()).unwrap_or_default();
                vm.set_source(&source, cx);
            }
            AppEvent::PlaybackChanged => cx.notify(),
            AppEvent::PyramidProgress(source) if *source == vm.view.source && vm.incomplete => vm.request_render(cx),
            AppEvent::SelectionChanged(source) if *source == vm.view.source && !vm.view.pinned_selection => {
                let channels = store.read(cx).selection(source).to_vec();
                let total = store.read(cx).sources().map_or(0, |s| s.entry(source).channels);
                vm.view.set_selection(channels, total);
                vm.request_render(cx);
            }
            _ => {}
        });
        let this = cx.weak_entity();
        cx.defer(move |cx| {
            let _ = this.update(cx, |vm, cx| vm.bind(cx));
        });
        Self {
            store,
            view,
            shown: None,
            retired: None,
            active: true,
            scheduled: false,
            incomplete: false,
            hover: String::new(),
            frames,
            hover_reply,
            _tasks: vec![frame_task, hover_task],
            _store: sub,
            home: Workspace::Explore,
            live: None,
            live_pending: None,
            live_note: None,
        }
    }

    /// Finds the window's events again when the view's chain ends with Crossings and the window,
    /// the channels shown or the chain changed (in the background; the ticks follow).
    fn update_live(&mut self, signal: &Arc<dyn dspk_view::SignalBackend>, window: (f64, f64), cx: &mut Context<Self>) {
        let store = self.store.read(cx);
        let node = if self.view.kind == TimeViewKind::Traces { self.shown_view(store).and_then(|v| crate::data::live::crossings_of(v).map(|n| (v.clone(), n.clone()))) } else { None };
        let Some((doc, node)) = node else {
            if self.live.is_some() || self.live_note.is_some() {
                (self.live, self.live_note) = (None, None);
                cx.notify();
            }
            return;
        };
        let rate = signal.info().sample_rate_hz();
        let samples = (window.0.max(0.0) * rate) as u64..((window.0 + window.1).max(0.0) * rate).ceil() as u64;
        let channels = self.view.drawn_channels().to_vec();
        let key = crate::data::live::LiveKey::new(&doc, &channels, samples.clone(), signal);
        if self.live.as_ref().is_some_and(|(k, _)| *k == key) || self.live_pending.as_ref() == Some(&key) {
            return;
        }
        self.live_pending = Some(key.clone());
        let signal = signal.clone();
        let task = gpui_kit::AppContext::background_spawn(cx, async move { crate::data::live::detect(&signal, &channels, samples, &node) });
        cx.spawn(async move |this, cx| {
            let result = task.await;
            let _ = this.update(cx, |vm, cx| {
                if vm.live_pending.as_ref() != Some(&key) {
                    return;
                }
                vm.live_pending = None;
                match result {
                    Ok(events) => {
                        vm.live = Some((key, Arc::new(events)));
                        vm.live_note = None;
                    }
                    Err(e) => {
                        vm.live = None;
                        vm.live_note = Some(e.to_string().trim_start_matches("invalid configuration: ").to_string());
                    }
                }
                cx.notify();
            });
        })
        .detach();
    }

    /// The workspace showing this view (Explore's views by default; a plugin's strip its own):
    /// it renders only while that workspace is in front.
    pub fn set_home(&mut self, home: Workspace, cx: &mut Context<Self>) {
        self.home = home;
        self.request_render(cx);
    }

    /// Points `view` at `source` of the open recording, with that source's shared selection.
    fn attach(view: &mut TimeView, store: &Entity<Store>, source: &str, cx: &App) {
        let s = store.read(cx);
        let Some(sources) = s.sources() else { return };
        let e = sources.entry(source);
        view.set_source(&e.id, &e.name, &e.unit, e.channels);
        view.set_selection(s.selection(&e.id).to_vec(), e.channels);
    }

    pub fn id(&self) -> ViewId {
        self.view.id
    }

    pub fn store(&self) -> &Entity<Store> {
        &self.store
    }

    /// The sources this view reads: its own recording's when it shows one beside the open one.
    fn sources(&self, cx: &App) -> Option<Arc<SourceSet>> {
        let s = self.store.read(cx);
        match self.foreign(cx) {
            Some(path) => s.sources_of(&path).cloned(),
            None => s.sources().cloned(),
        }
    }

    /// The recording this view reads when it is not the open one.
    pub fn foreign(&self, cx: &App) -> Option<std::path::PathBuf> {
        let path = self.view.recording.as_ref()?;
        let open = self.store.read(cx).recording.as_ref().and_then(|r| r.path.clone());
        (open.as_ref() != Some(path)).then(|| path.clone())
    }

    /// The timeline in this view's recording's time (shifted by the placements for a view of
    /// another recording).
    fn timeline(&self, cx: &App) -> crate::domain::timeline::TimelineState {
        let s = self.store.read(cx);
        let mut tl = s.timeline.clone();
        if let Some(path) = self.foreign(cx) {
            let offset = s.time_offset_of(&path);
            tl.window_start_sec += offset;
            tl.current_time_sec += offset;
        }
        tl
    }

    /// Points this view at `source` of the loaded recording `path` (beside the open one): its own
    /// channel selection (all channels), titled with the file. False when it isn't loaded.
    pub fn pin_recording(&mut self, path: &std::path::Path, source: &str, cx: &mut Context<Self>) -> bool {
        let Some(sources) = self.store.read(cx).sources_of(path).cloned() else { return false };
        let e = sources.entry(source);
        self.view.recording = Some(path.to_path_buf());
        self.view.set_source(&e.id, &e.name, &e.unit, e.channels);
        if self.view.selection.is_empty() || self.view.selection.iter().any(|&c| c >= e.channels) {
            self.view.set_selection(0..e.channels, e.channels);
        }
        self.view.pinned_selection = true;
        self.view.doc_view.clear();
        let file = path.file_name().map_or_else(String::new, |n| n.to_string_lossy().into_owned());
        self.view.source_name = format!("{file} · {}", e.name);
        cx.emit(TraceEvent::Changed);
        self.request_render(cx);
        true
    }

    // ------------------------------------------------------------------------
    // Rendering
    // ------------------------------------------------------------------------

    /// Asks for a frame once the current update ends.
    pub fn request_render(&mut self, cx: &mut Context<Self>) {
        self.view.needs_render = true;
        cx.notify();
        if self.scheduled {
            return;
        }
        self.scheduled = true;
        let this = cx.weak_entity();
        cx.defer(move |cx| {
            let _ = this.update(cx, |vm, cx| {
                vm.scheduled = false;
                vm.submit(cx);
            });
        });
    }

    fn submit(&mut self, cx: &mut Context<Self>) {
        if !self.active || !self.view.needs_render || self.view.canvas_width == 0 || self.view.canvas_height == 0 {
            return;
        }
        // Hidden behind another workspace: draw when its workspace shows again (the work pool
        // and the device stay free for the workspace in front)
        if self.store.read(cx).workspace != self.home {
            return;
        }
        let Some(sources) = self.sources(cx) else { return };
        self.view.needs_render = false;
        let foreign = self.foreign(cx).is_some();
        let timeline = self.timeline(cx);
        // The pyramid builds in the background, nearest what this view shows first (a view of
        // another recording asks its signal directly)
        let (source_id, focus) = (self.view.source.clone(), timeline.window_start_sec);
        if foreign {
            let ds = sources.get(&self.view.source);
            ds.signal().focus((focus.max(0.0) * ds.sample_rate) as u64);
        } else {
            self.store.update(cx, |s, cx| s.focus_pyramid(&source_id, focus, cx));
        }
        let store = self.store.read(cx);
        let dataset = sources.get(&self.view.source);
        let on_default = sources.index_of(&self.view.source) == sources.index_of("");
        let events = match &store.recording {
            Some(r) if on_default && !foreign => r.events.clone(),
            _ => Default::default(),
        };
        // This view's session view: its processed preview, else the recording as stored (a view
        // of another recording: as stored)
        let shown = if foreign { None } else { self.shown_view(store) };
        let signal = shown.and_then(|v| store.preview_of(&v.id)).unwrap_or_else(|| dataset.signal().clone());
        let mut req = self.view.render_request(&timeline, signal, events, store.palette());
        let window = req.window_start_sec..req.window_start_sec + req.window_sec;
        match req.mode {
            TimeViewKind::Traces if self.view.spikes => {
                req.overlay = shown.and_then(|v| crate::plugin::plugins().iter().find_map(|p| p.trace_overlay(store, v, window.clone(), cx))).map(Arc::new);
            }
            TimeViewKind::Raster => {
                req.raster = shown.and_then(|v| crate::plugin::plugins().iter().find_map(|p| p.raster(store, v, window.clone(), cx))).map(Arc::new);
            }
            _ => {}
        }
        let live_signal = (!foreign).then(|| req.signal.clone());
        let live_window = (req.window_start_sec, req.window_sec);
        let samples_per_px = req.window_sec * dataset.sample_rate / req.width.max(1) as f64;
        let (start, window) = (req.window_start_sec, req.window_sec);
        let tx = self.frames.clone();
        let work = Box::new(move |cancel: &std::sync::atomic::AtomicBool| {
            let t0 = Instant::now();
            let Some((frame, scale, complete)) = render_on_worker(&req, cancel) else { return };
            let (w, h) = (frame.width, frame.height);
            let Some(buffer) = image::RgbaImage::from_raw(w, h, frame.into_bgra()) else { return };
            // Converted here, on the work thread: the UI thread only shows it
            let image = Arc::new(RenderImage::new(vec![image::Frame::new(buffer)]));
            let shown = Shown { image, start, window };
            let _ = tx.try_send(Delivered { shown, scale: Some(scale), complete, elapsed: t0.elapsed(), samples_per_px });
        });
        cx.global::<Services>().pool.request(JobKey::new(self.view.id, RENDER_SLOT), work);
        if let Some(signal) = live_signal {
            self.update_live(&signal, live_window, cx);
        }
    }

    fn on_frame(&mut self, d: Delivered, cx: &mut Context<Self>) {
        if let Some(scale) = d.scale {
            // The scale bar follows the scale the frame was drawn with
            self.view.amp_scale = scale;
        }
        self.incomplete = !d.complete;
        if let Some(old) = self.retired.take() {
            cx.drop_image(old, None);
        }
        self.retired = self.shown.replace(d.shown).map(|s| s.image);
        let info = format!("Render {:.1} ms · {:.1} samples/px", d.elapsed.as_secs_f64() * 1e3, d.samples_per_px);
        self.store.update(cx, |s, cx| s.set_render_info(info, cx));
        cx.notify();
    }

    /// Plot area size in physical pixels.
    pub fn set_plot_size(&mut self, width: u32, height: u32, scale: f32, cx: &mut Context<Self>) {
        if (width, height, scale) != (self.view.canvas_width, self.view.canvas_height, self.view.scale_factor) {
            self.view.set_canvas(width, height, scale);
            self.request_render(cx);
        }
    }

    /// Whether it is the tab shown in its tab group.
    pub fn is_active(&self) -> bool {
        self.active
    }

    /// Shown or hidden in its tab group (or its dock opened / closed).
    pub fn set_active(&mut self, active: bool, cx: &mut Context<Self>) {
        self.active = active;
        if active && self.view.needs_render {
            self.request_render(cx);
        }
    }

    /// The view left the dock: stop its jobs and release its images.
    pub fn close(&mut self, cx: &mut Context<Self>) {
        self.active = false;
        cx.global::<Services>().pool.forget(self.view.id);
        for image in self.shown.take().map(|s| s.image).into_iter().chain(self.retired.take()) {
            cx.drop_image(image, None);
        }
    }

    // ------------------------------------------------------------------------
    // Settings (this view only)
    // ------------------------------------------------------------------------

    pub fn set_kind(&mut self, kind: TimeViewKind, cx: &mut Context<Self>) {
        if self.view.kind != kind {
            self.view.kind = kind;
            self.view.retitle();
            cx.emit(TraceEvent::Changed);
            self.request_render(cx);
        }
    }

    /// Traces with the spikes coloured (`spikes`), or alone.
    pub fn set_spikes(&mut self, spikes: bool, cx: &mut Context<Self>) {
        if self.view.spikes != spikes {
            self.view.spikes = spikes;
            cx.emit(TraceEvent::Changed);
            self.request_render(cx);
        }
    }

    /// Events as ticks on (`events`), or off.
    pub fn set_events(&mut self, events: bool, cx: &mut Context<Self>) {
        if self.view.events != events {
            self.view.events = events;
            cx.emit(TraceEvent::Changed);
            cx.notify();
        }
    }

    /// The ticks alone (the signal hidden), or over the traces.
    pub fn set_events_only(&mut self, only: bool, cx: &mut Context<Self>) {
        if self.view.events_only != only {
            self.view.events_only = only;
            if only {
                self.view.events = true;
            }
            cx.emit(TraceEvent::Changed);
            self.request_render(cx);
        }
    }

    pub fn set_color_by(&mut self, color_by: crate::screens::explore::trace::renderer::ColorBy, cx: &mut Context<Self>) {
        if self.view.color_by != color_by {
            self.view.color_by = color_by;
            cx.emit(TraceEvent::Changed);
            self.request_render(cx);
        }
    }

    /// Draws session view `id` (a view of this view's source): the chain picked in the Pipeline
    /// or by an agent. The one it drew before is released (it leaves the session if it holds no
    /// work).
    pub fn show_doc_view(&mut self, id: &str, cx: &mut Context<Self>) {
        if self.view.doc_view == id {
            return;
        }
        let old = std::mem::replace(&mut self.view.doc_view, id.to_string());
        if !old.is_empty() {
            cx.emit(TraceEvent::Released(old));
        }
        self.shown = None;
        cx.emit(TraceEvent::Changed);
        self.request_render(cx);
    }

    /// Shows another source of the open file (Explore's views: with that source's session view
    /// and its processing).
    pub fn set_source(&mut self, source: &str, cx: &mut Context<Self>) {
        if self.view.source != source {
            Self::attach(&mut self.view, &self.store, source, cx);
            let old = std::mem::take(&mut self.view.doc_view);
            if !old.is_empty() {
                cx.emit(TraceEvent::Released(old));
            }
            self.shown = None;
            self.bind(cx);
            cx.emit(TraceEvent::Changed);
            self.request_render(cx);
        }
    }

    /// Explore's views show a session view of their own: the one of their source with all its
    /// channels (made when none). The workspaces' strips follow the active session view instead.
    fn bind(&mut self, cx: &mut Context<Self>) {
        if self.home != Workspace::Explore || !self.view.doc_view.is_empty() {
            return;
        }
        let (store, source) = (self.store.clone(), self.view.source.clone());
        let this = cx.weak_entity();
        // After this update: making the session view updates the store
        cx.defer(move |cx| {
            let id = store.update(cx, |s, cx| s.view_for_source(&source, cx));
            let _ = this.update(cx, |vm, cx| {
                if let Some(id) = id
                    && vm.view.source == source
                {
                    vm.view.doc_view = id;
                    cx.emit(TraceEvent::Changed);
                }
                vm.request_render(cx);
            });
        });
    }

    /// The session view this view draws: its own (Explore), else the active one when it reads this
    /// view's source (the strips).
    pub fn shown_view<'a>(&self, store: &'a Store) -> Option<&'a crate::domain::session::ViewDoc> {
        if !self.view.doc_view.is_empty() {
            return store.doc.view(&self.view.doc_view);
        }
        store.active_view().filter(|v| store.view_source(v) == self.view.source)
    }

    pub fn set_lanes(&mut self, lanes: usize, cx: &mut Context<Self>) {
        self.view.set_lanes(lanes);
        cx.emit(TraceEvent::Changed);
        self.request_render(cx);
    }

    pub fn scroll_by(&mut self, delta: i64, cx: &mut Context<Self>) {
        let before = self.view.scroll;
        self.view.scroll_by(delta);
        if self.view.scroll != before {
            self.request_render(cx);
        }
    }

    pub fn page(&mut self, forward: bool, cx: &mut Context<Self>) {
        let before = self.view.scroll;
        self.view.page(forward);
        if self.view.scroll != before {
            self.request_render(cx);
        }
    }

    pub fn zoom_gain(&mut self, factor: f32, cx: &mut Context<Self>) {
        self.view.zoom_gain(factor);
        cx.emit(TraceEvent::Changed);
        self.request_render(cx);
    }

    pub fn set_auto_scale(&mut self, on: bool, cx: &mut Context<Self>) {
        self.view.auto_scale = on;
        cx.emit(TraceEvent::Changed);
        self.request_render(cx);
    }

    pub fn set_remove_dc(&mut self, on: bool, cx: &mut Context<Self>) {
        self.view.remove_dc = on;
        cx.emit(TraceEvent::Changed);
        self.request_render(cx);
    }

    /// Pins (or unpins) this view's channel selection from the source's shared selection.
    pub fn set_pinned_selection(&mut self, pinned: bool, cx: &mut Context<Self>) {
        if self.view.pinned_selection == pinned {
            return;
        }
        self.view.pinned_selection = pinned;
        if !pinned {
            let s = self.store.read(cx);
            let channels = s.selection(&self.view.source).to_vec();
            let total = self.sources(cx).map_or(0, |ss| ss.entry(&self.view.source).channels);
            self.view.set_selection(channels, total);
        }
        cx.emit(TraceEvent::Changed);
        self.request_render(cx);
    }

    fn total_channels(&self, cx: &App) -> usize {
        self.sources(cx).map_or(0, |s| s.entry(&self.view.source).channels)
    }

    pub fn set_selection(&mut self, channels: impl IntoIterator<Item = usize>, cx: &mut Context<Self>) {
        let total = self.total_channels(cx);
        self.view.set_selection(channels, total);
        cx.emit(TraceEvent::Changed);
        self.request_render(cx);
    }

    pub fn toggle_channel(&mut self, channel: usize, cx: &mut Context<Self>) {
        let total = self.total_channels(cx);
        self.view.toggle_channel(channel, total);
        cx.emit(TraceEvent::Changed);
        self.request_render(cx);
    }

    pub fn select_all(&mut self, cx: &mut Context<Self>) {
        let total = self.total_channels(cx);
        self.set_selection(0..total, cx);
    }

    pub fn select_none(&mut self, cx: &mut Context<Self>) {
        self.set_selection([], cx);
    }

    pub fn select_invert(&mut self, cx: &mut Context<Self>) {
        let total = self.total_channels(cx);
        let keep: Vec<usize> = (0..total).filter(|c| !self.view.selection.contains(c)).collect();
        self.set_selection(keep, cx);
    }

    pub fn select_ranges(&mut self, text: &str, cx: &mut Context<Self>) -> Result<(), String> {
        let channels = crate::screens::explore::trace::view::parse_channel_ranges(text, self.total_channels(cx))?;
        self.set_selection(channels, cx);
        Ok(())
    }

    /// Double-click on a channel: show it alone, or go back when one is already shown alone.
    pub fn toggle_solo(&mut self, channel: usize, cx: &mut Context<Self>) {
        if !self.view.unsolo() {
            self.view.solo(channel);
        }
        if !self.view.pinned_selection {
            // Back from solo on a view that follows the shared selection: catch up with it
            let s = self.store.read(cx);
            let channels = s.selection(&self.view.source).to_vec();
            let total = self.sources(cx).map_or(0, |ss| ss.entry(&self.view.source).channels);
            self.view.set_selection(channels, total);
        }
        cx.emit(TraceEvent::Changed);
        self.request_render(cx);
    }

    /// Esc: back from a channel shown alone (nothing otherwise).
    pub fn leave_solo(&mut self, cx: &mut Context<Self>) {
        if let Some(ch) = self.view.solo.as_ref().map(|s| s.channel) {
            self.toggle_solo(ch, cx);
        }
    }

    /// Scrolls so `channel` is on screen; false when this view does not show it.
    pub fn reveal(&mut self, channel: usize, cx: &mut Context<Self>) -> bool {
        let found = self.view.reveal(channel);
        if found {
            self.request_render(cx);
        }
        found
    }

    pub fn pressed(&mut self, cx: &mut Context<Self>) {
        cx.emit(TraceEvent::Pressed);
    }

    // ------------------------------------------------------------------------
    // Pointer (positions as fractions of the plot area)
    // ------------------------------------------------------------------------

    /// The pointer at (`fx`, `fy`) of the plot: a readout now, or once its sample is read.
    pub fn hover_at(&mut self, fx: f32, fy: f32, cx: &mut Context<Self>) {
        let Some(sources) = self.sources(cx) else { return };
        let dataset = sources.get(&self.view.source);
        let (w, h) = (self.view.canvas_width as f32, self.view.canvas_height as f32);
        self.view.hover_seq += 1;
        let timeline = self.timeline(cx);
        let target = self.view.hover_target(fx * w, fy * h, &dataset, &timeline);
        match target {
            HoverTarget::Text(text) => {
                self.hover = text;
                cx.notify();
            }
            HoverTarget::Sample { channel, sample, prefix, unit } => {
                let req = HoverRequest { seq: self.view.hover_seq, dataset, channel, sample, prefix, unit, reply: self.hover_reply.clone() };
                cx.global::<Services>().hover.request(req);
            }
        }
    }

    pub fn hover_left(&mut self, cx: &mut Context<Self>) {
        self.view.hover_seq += 1;
        if !self.hover.is_empty() {
            self.hover.clear();
            cx.notify();
        }
    }

    /// Channel at a height (fraction of the plot).
    pub fn channel_at(&self, fy: f32) -> Option<usize> {
        self.view.channel_at(fy * self.view.canvas_height as f32)
    }
}
