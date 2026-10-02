//! One time view (traces or heatmap): its settings ([`TimeView`]), the frame on screen and the
//! window it was drawn for, and the hover readout.
//!
//! Rendering is asked for on change, never polled: a timeline, selection, size or setting change
//! schedules one request, sent once the current update ends (so a burst of changes costs one
//! request), and only while the view is displayed. Frames come back over this view's own channel;
//! the image they replace is released one frame later (it may still be on screen).

use std::sync::Arc;
use std::time::Duration;

use gpui_kit::{App, Context, Entity, EventEmitter, Global, RenderImage, Subscription, Task};

use crate::engine::data::SourceSet;
use crate::engine::render_pool::{FrameInfo, RenderJob, RenderPool, Rendered};
use crate::engine::time::hover::{HoverReader, HoverReply, HoverRequest};
use crate::engine::time::renderer::{render_on_worker, TimeViewKind};
use crate::engine::time::view::{HoverTarget, TimeView, ViewId};
use crate::store::{AppEvent, Store};
use dsp_base::resampler::summary::BASE as SUMMARY_BASE;

/// The background threads every view shares.
pub struct Services {
    pub pool: RenderPool,
    pub hover: HoverReader,
}

impl Global for Services {}

impl Services {
    pub fn install(cx: &mut App) {
        cx.set_global(Services { pool: RenderPool::new(RenderPool::default_threads()), hover: HoverReader::spawn() });
    }
}

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
    elapsed: Duration,
    samples_per_px: f64,
}

pub enum TraceEvent {
    /// The view was pressed: it becomes the focused view.
    Pressed,
    /// Kind, source or title changed (tab title, settings panel).
    Changed,
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
    pub hover: String,
    frames: async_channel::Sender<Delivered>,
    hover_reply: HoverReply,
    _tasks: Vec<Task<()>>,
    _store: Subscription,
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
            AppEvent::PlaybackChanged => cx.notify(),
            AppEvent::SummaryProgress(source) if *source == vm.view.source && vm.zoomed_out(store.read(cx).timeline.visible_window_sec, cx) => vm.request_render(cx),
            AppEvent::SelectionChanged(source) if *source == vm.view.source => {
                let channels = store.read(cx).selection(source).to_vec();
                let total = store.read(cx).sources().map_or(0, |s| s.entry(source).channels);
                vm.view.set_selection(channels, total);
                vm.request_render(cx);
            }
            _ => {}
        });
        Self {
            store,
            view,
            shown: None,
            retired: None,
            active: true,
            scheduled: false,
            hover: String::new(),
            frames,
            hover_reply,
            _tasks: vec![frame_task, hover_task],
            _store: sub,
        }
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

    fn sources(&self, cx: &App) -> Option<Arc<SourceSet>> {
        self.store.read(cx).sources().cloned()
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

    /// Whether a window of `window_sec` draws from the min/max summary (more samples per pixel
    /// column than its finest bucket) rather than raw samples.
    fn zoomed_out(&self, window_sec: f64, cx: &App) -> bool {
        let Some(sources) = self.sources(cx) else { return false };
        let rate = sources.entry(&self.view.source).sample_rate;
        window_sec * rate >= SUMMARY_BASE as f64 * self.view.canvas_width.max(1) as f64
    }

    fn submit(&mut self, cx: &mut Context<Self>) {
        if !self.active || !self.view.needs_render || self.view.canvas_width == 0 || self.view.canvas_height == 0 {
            return;
        }
        let Some(sources) = self.sources(cx) else { return };
        self.view.needs_render = false;
        // The summary fills in the background, nearest what this view shows first
        let (source_id, focus) = (self.view.source.clone(), self.store.read(cx).timeline.window_start_sec);
        self.store.update(cx, |s, cx| s.summarize(&source_id, focus, cx));
        let store = self.store.read(cx);
        let dataset = sources.get(&self.view.source);
        let lod = dataset.lod();
        self.view.has_lod = lod.is_some();
        let on_default = sources.index_of(&self.view.source) == sources.index_of("");
        let events = match &store.recording {
            Some(r) if on_default => r.events.clone(),
            _ => Default::default(),
        };
        let source: Arc<dyn dsp_core::RecordingSource> = dataset.clone();
        let req = self.view.render_request(&store.timeline, source, lod, Some(dataset.summary()), events, store.palette());
        let samples_per_px = req.window_sec * dataset.sample_rate / req.width.max(1) as f64;
        let (start, window) = (req.window_start_sec, req.window_sec);
        let tx = self.frames.clone();
        let deliver = Arc::new(move |out: Rendered, info: FrameInfo| {
            let (w, h) = (out.frame.width, out.frame.height);
            let Some(buffer) = image::RgbaImage::from_raw(w, h, out.frame.into_bgra()) else { return };
            let image = Arc::new(RenderImage::new(vec![image::Frame::new(buffer)]));
            let shown = Shown { image, start, window };
            let _ = tx.try_send(Delivered { shown, scale: out.scale, elapsed: info.elapsed, samples_per_px });
        });
        cx.global::<Services>().pool.request(RenderJob { key: self.view.id, render: Box::new(move |ctx| render_on_worker(&req, ctx)), deliver });
    }

    fn on_frame(&mut self, d: Delivered, cx: &mut Context<Self>) {
        if let Some(scale) = d.scale {
            // The scale bar follows the scale the frame was drawn with
            self.view.amp_scale = scale;
        }
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

    /// Shows another source of the open file.
    pub fn set_source(&mut self, source: &str, cx: &mut Context<Self>) {
        if self.view.source != source {
            Self::attach(&mut self.view, &self.store, source, cx);
            self.shown = None;
            cx.emit(TraceEvent::Changed);
            self.request_render(cx);
        }
    }

    pub fn set_lanes(&mut self, lanes: usize, cx: &mut Context<Self>) {
        self.view.set_lanes(lanes);
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
        self.request_render(cx);
    }

    pub fn set_auto_scale(&mut self, on: bool, cx: &mut Context<Self>) {
        self.view.auto_scale = on;
        self.request_render(cx);
    }

    pub fn set_remove_dc(&mut self, on: bool, cx: &mut Context<Self>) {
        self.view.remove_dc = on;
        self.request_render(cx);
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
        let target = self.view.hover_target(fx * w, fy * h, &dataset, &self.store.read(cx).timeline);
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
