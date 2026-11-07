//! What a domain plugin adds to the core app (electrophysiology: `dspk-app-synapse`). The core
//! works on any multichannel signal; a plugin brings its domain:
//!
//! | Extension | The core asks | e.g. electrophysiology |
//! |---|---|---|
//! | [`SourceOpener`] | open this file (before dspk-io's generic formats) | SpikeGLX, NWB, with probes |
//! | [`ChannelLayout`] | how channels group, what to call them | shanks, site positions |
//! | [`AppPlugin::synthetic`] | a recording made up on demand | drifting units, noise, hum |
//! | [`AppPlugin::trace_overlay`] | what to draw over the traces of a window | spikes coloured by unit |
//! | [`AppPlugin::workspace`] | the body of a workspace tab | Sorting |
//! | [`AppPlugin::panel`], [`AppPlugin::view_section`] | a panel on the right side bar, for the active view | Sorting: the linked sorting, units, Sort… |
//! | per-view data in the session document | `views[i].plugins.<id>` (any JSON) | the linked `.sorting.zarr` |
//!
//! Plugins are compiled in (Rust has no stable plugin ABI): the `dspk-app` binary registers the
//! ones it is built with ([`register`]) before the app starts. Programs under another licence
//! (the GPL sorters) stay separate processes a plugin runs; they are never linked.

use std::ops::Range;
use std::path::Path;
use std::sync::{Arc, OnceLock};

use dspk_core::{DspResult, RecordingSource};
use dspk_io::SourceEntry;
use gpui_kit::{AnyElement, AnyView, App, Entity, Window};

use crate::data::store::Store;
use crate::domain::session::ViewDoc;
use crate::navigation::route::Workspace;
use crate::ui::plot::canvas::Pixel;

/// What a view's data holds beyond its continuous signal (traces and heatmaps need only the
/// signal): spike times (the Spikes layer) and units (the Raster layer).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Layers {
    pub spikes: bool,
    pub units: bool,
}

impl Layers {
    fn or(self, other: Layers) -> Layers {
        Layers { spikes: self.spikes || other.spikes, units: self.units || other.units }
    }
}

/// What the installed plugins can draw for session view `view`, together.
pub fn layers(store: &Store, view: &ViewDoc, cx: &App) -> Layers {
    plugins().iter().fold(Layers::default(), |acc, p| acc.or(p.layers(store, view, cx)))
}

/// Channels in groups (e.g. the shanks of a probe) and a label per channel (e.g. its site).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ChannelLayout {
    /// What a group is called in lists (`Shank`).
    pub group_kind: String,
    pub groups: Vec<ChannelGroup>,
    /// Per channel (index = channel), shown next to it.
    pub labels: Vec<Option<String>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ChannelGroup {
    pub name: String,
    /// Channels in display order.
    pub channels: Vec<usize>,
}

impl ChannelLayout {
    pub fn label(&self, channel: usize) -> Option<&str> {
        self.labels.get(channel)?.as_deref()
    }
}

/// Opens the files of a domain's formats.
pub trait SourceOpener: Send + Sync {
    /// Whether `path` is one of its formats (cheap: the name or a header).
    fn reads(&self, path: &Path) -> bool;
    fn sources(&self, path: &Path) -> DspResult<Vec<SourceEntry>>;
    fn open(&self, path: &Path, id: &str) -> DspResult<Box<dyn RecordingSource>>;
    /// The channel layout of source `id` of `path`, when the file stores one.
    fn layout(&self, _path: &Path, _id: &str) -> Option<ChannelLayout> {
        None
    }
}

/// What to draw over a window of traces: per unit its colour and channels, and marks (spans of
/// session time) of those units. With `dim`, the traces themselves are drawn faint so the marks
/// stand out.
#[derive(Debug, Clone, Default)]
pub struct TraceOverlay {
    pub dim: bool,
    pub units: Vec<OverlayUnit>,
    pub marks: Vec<OverlayMark>,
}

#[derive(Debug, Clone)]
pub struct OverlayUnit {
    pub color: Pixel,
    /// Channels the unit's marks cover, sorted.
    pub channels: Vec<usize>,
}

#[derive(Debug, Clone, Copy)]
pub struct OverlayMark {
    /// Session seconds.
    pub start_sec: f64,
    pub end_sec: f64,
    /// Index into [`TraceOverlay::units`].
    pub unit: u32,
}

/// Spike rows for a raster view: per unit its colour, label and spike times (session seconds)
/// in the window.
#[derive(Debug, Clone, Default)]
pub struct Raster {
    pub rows: Vec<RasterRow>,
}

#[derive(Debug, Clone)]
pub struct RasterRow {
    pub color: Pixel,
    pub label: String,
    pub times: Vec<f64>,
}

/// A domain plugin. Every method has a default: a plugin implements what it adds.
pub trait AppPlugin: Send + Sync + 'static {
    /// Its key in the session document (`views[i].plugins.<id>`) and in logs.
    fn id(&self) -> &'static str;
    fn name(&self) -> &'static str;

    fn opener(&self) -> Option<&dyn SourceOpener> {
        None
    }

    /// A recording computed on demand (`channels` × `duration_sec` at `sample_rate`).
    fn synthetic(&self, _channels: usize, _sample_rate: f64, _duration_sec: f64) -> Option<anyhow::Result<Arc<dyn RecordingSource>>> {
        None
    }

    /// Called once the store exists (subscribe to its events, create the plugin's state).
    fn install(&self, _store: &Entity<Store>, _cx: &mut App) {}

    /// Marks over the traces of session view `view` for the session-time window `window_sec`.
    fn trace_overlay(&self, _store: &Store, _view: &ViewDoc, _window_sec: Range<f64>, _cx: &App) -> Option<TraceOverlay> {
        None
    }

    /// Raster rows of session view `view` for the session-time window `window_sec`.
    fn raster(&self, _store: &Store, _view: &ViewDoc, _window_sec: Range<f64>, _cx: &App) -> Option<Raster> {
        None
    }

    /// What this plugin can draw for session view `view` (e.g. a linked sorting: spikes, units).
    fn layers(&self, _store: &Store, _view: &ViewDoc, _cx: &App) -> Layers {
        Layers::default()
    }

    /// The body of workspace `ws` when this plugin provides it.
    fn workspace(&self, _ws: Workspace, _store: &Entity<Store>, _window: &mut Window, _cx: &mut App) -> Option<AnyView> {
        None
    }

    /// Whether this plugin provides workspace `ws` (its tab is shown).
    fn provides(&self, _ws: Workspace) -> bool {
        false
    }

    /// Its panel on the right side bar: title and icon (its body is [`Self::view_section`]).
    fn panel(&self) -> Option<(&'static str, std::rc::Rc<dyn Fn() -> gpui_kit::component::Icon>)> {
        None
    }

    /// The body of its panel for `view`.
    fn view_section(&self, _store: &Entity<Store>, _view: &ViewDoc, _window: &mut Window, _cx: &mut App) -> Option<AnyElement> {
        None
    }
}

static PLUGINS: OnceLock<Vec<Arc<dyn AppPlugin>>> = OnceLock::new();

/// Registers the plugins the app is built with (once, before it starts).
pub fn register(plugins: Vec<Arc<dyn AppPlugin>>) {
    if PLUGINS.set(plugins).is_err() {
        tracing::warn!("plugins are registered once");
    }
}

/// The registered plugins (none when [`register`] was not called: tests, a core-only build).
pub fn plugins() -> &'static [Arc<dyn AppPlugin>] {
    PLUGINS.get().map_or(&[], Vec::as_slice)
}

/// `path`'s sources: the first plugin that reads the file, else dspk-io's generic formats.
pub fn sources(path: &Path) -> DspResult<Vec<SourceEntry>> {
    match opener_of(path) {
        Some(o) => o.sources(path),
        None => dspk_io::sources(path),
    }
}

/// The plugin opener that reads `path`, if any.
fn opener_of(path: &Path) -> Option<&'static dyn SourceOpener> {
    plugins().iter().filter_map(|p| p.opener()).find(|o| o.reads(path))
}

/// Source `id` of `path`, by the opener that listed it.
pub fn open_source(path: &Path, id: &str) -> DspResult<Box<dyn RecordingSource>> {
    match opener_of(path) {
        Some(o) => o.open(path, id),
        None => dspk_io::open_source(path, id),
    }
}

/// Source `id` of `path` (`None`: the file's default source).
pub fn open(path: &Path, id: Option<&str>) -> DspResult<Box<dyn RecordingSource>> {
    match id {
        Some(id) => open_source(path, id),
        None => {
            let list = sources(path)?;
            let id = dspk_io::default_source(&list).map(|e| e.id.clone()).unwrap_or_default();
            open_source(path, &id)
        }
    }
}

/// The channel layout of source `id` of `path`, from the opener that reads it.
pub fn layout(path: &Path, id: &str) -> Option<ChannelLayout> {
    opener_of(path)?.layout(path, id)
}

/// A synthetic recording from the first plugin that makes one.
pub fn synthetic(channels: usize, sample_rate: f64, duration_sec: f64) -> Option<anyhow::Result<Arc<dyn RecordingSource>>> {
    plugins().iter().find_map(|p| p.synthetic(channels, sample_rate, duration_sec))
}
