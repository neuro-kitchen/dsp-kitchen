//! DSP App: an electrophysiology workbench (GPUI + gpui-kit). Workspaces (Explore · Sorting ·
//! Pipeline · Curation) of docked views on one timeline; see `.tasks/slint-gpui-migration/`.
//!
//! Layers: `engine` (no UI type: data, timeline, renderers, render threads) → `store` (shared state
//! and its events) → `viewmodels` → `views` / `app` (GPUI). `--snapshot` uses the engine alone.

mod actions;
mod app;
mod assets;
mod engine;
mod session;
mod store;
mod viewmodels;
mod views;
mod widgets;
mod workspace;

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Result;
use clap::Parser;
use gpui_kit::component::TitleBar;
use gpui_kit::{px, size, App, AppContext as _, Bounds, TitlebarOptions, WindowBounds, WindowOptions};

use engine::data::{Dataset, SourceSet, SpikeEventStore};
use engine::time::renderer::{TimeViewKind, WaveformRenderer};
use engine::time::timeline::TimelineState;
use engine::time::view::TimeView;
use session::Session;
use store::Store;

#[derive(Parser, Debug)]
#[command(name = "dsp-app")]
#[command(about = "DSP App: electrophysiology workbench (GPUI): docked time views on one timeline")]
struct Args {
    /// Recording to open: SpikeGLX .bin/.cbin, raw .bin with a JSON .meta, or a Zarr store
    /// (`dsp-app <path>`, or `--file <path>`). Without it (and without --synthetic) the last
    /// opened recording is reopened.
    #[arg(value_name = "PATH", conflicts_with = "file")]
    path: Option<PathBuf>,

    /// Same as PATH.
    #[arg(short, long)]
    file: Option<PathBuf>,

    /// Open a procedural recording of this length instead, e.g. 90s, 10m, 2h
    #[arg(long)]
    synthetic: Option<String>,

    /// Channels of the --synthetic recording
    #[arg(short, long, default_value_t = 32)]
    channels: usize,

    /// Sample rate of the --synthetic recording in Hz
    #[arg(short = 'r', long, default_value_t = 30_000.0)]
    sample_rate: f64,

    /// Render one view to this PNG (no window) and exit
    #[arg(long)]
    snapshot: Option<PathBuf>,

    /// Render the snapshot as a heatmap
    #[arg(long)]
    heatmap: bool,

    /// Source (signal) of the file for --snapshot, by name or id (e.g. EMG)
    #[arg(long)]
    source: Option<String>,

    /// Where the --snapshot window is centred (seconds)
    #[arg(long, default_value_t = 2.45)]
    at: f64,

    /// Neither read nor write the saved session (recent files, workspace)
    #[arg(long)]
    no_session: bool,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let mut args = Args::parse();
    args.file = args.file.take().or(args.path.take());

    if let Some(path) = &args.snapshot {
        return snapshot(&args, path);
    }

    let session_path = if args.no_session { None } else { Session::path() };
    let session = Session::load(session_path.as_deref());
    // What to open at start: --synthetic, --file, else the last recording
    let synthetic = args.synthetic.as_deref().map(parse_duration).transpose()?;
    let file = args.file.clone().map(|p| std::fs::canonicalize(&p).unwrap_or(p)).or_else(|| session.recent.first().cloned().filter(|p| p.exists()));
    let (channels, rate) = (args.channels, args.sample_rate);

    gpui_kit::application().with_assets(assets::AppAssets).run(move |cx: &mut App| {
        gpui_kit::init(cx);
        actions::bind_keys(cx);
        viewmodels::Services::install(cx);
        cx.on_action(|_: &actions::Quit, cx| cx.quit());
        // One main window: closing it ends the app (Linux / Windows convention)
        cx.on_window_closed(|cx, _| cx.quit()).detach();

        let store = cx.new(|_| Store::new(session, session_path));
        let options = WindowOptions {
            titlebar: Some(TitlebarOptions { title: Some("DSP App".into()), ..TitleBar::title_bar_options() }),
            window_bounds: Some(WindowBounds::Windowed(Bounds::centered(None, size(px(1400.), px(880.)), cx))),
            window_min_size: Some(size(px(760.), px(520.))),
            app_id: Some("org.dsp-kitchen.dsp-app".into()),
            ..TitleBar::window_options()
        };
        let s = store.clone();
        let (_, root) = gpui_kit::open_window(options, cx, |window, cx| cx.new(|cx| app::DspApp::new(s, window, cx))).expect("failed to open the main window");
        cx.activate(true);

        match (synthetic, file) {
            (Some(d), _) => store.update(cx, |s, cx| s.open_synthetic(channels, rate, d, cx)),
            (None, Some(p)) => root.update(cx, |a, cx| a.open_any(p, cx)),
            (None, None) => {}
        }
    });
    Ok(())
}

/// Renders the default traces (or heatmap) view of the recording around `--at` to a PNG, exact
/// at any zoom: the pyramid is built over the visible window first.
fn snapshot(args: &Args, path: &std::path::Path) -> Result<()> {
    let sources = match (&args.synthetic, &args.file) {
        (Some(d), _) => SourceSet::single(Dataset::procedural(args.channels, args.sample_rate, parse_duration(d)?)?),
        (None, Some(p)) => SourceSet::open(p)?,
        (None, None) => anyhow::bail!("--snapshot needs --file or --synthetic"),
    };
    let entry = match &args.source {
        Some(name) => sources.entries().iter().find(|e| &e.name == name || &e.id == name).ok_or_else(|| anyhow::anyhow!("no source {name}"))?.clone(),
        None => sources.default_entry().clone(),
    };
    let (w, h) = (1200u32, 600u32);
    let mut timeline = TimelineState::new(sources.extent_sec());
    timeline.scrub_to(args.at);
    let kind = if args.heatmap { TimeViewKind::Heatmap } else { TimeViewKind::Traces };
    let mut view = TimeView::new(1, kind, Vec::new());
    view.set_source(&entry.id, &entry.name, &entry.unit, entry.channels);
    view.set_canvas(w, h, 1.0);
    let dataset = sources.get(&entry.id);
    let window = |t: f64| ((t - dataset.start_time_sec) * dataset.sample_rate).max(0.0) as u64;
    let (start, end) = (window(timeline.window_start_sec), window(timeline.window_start_sec + timeline.visible_window_sec));
    dataset.signal().prepare(start, end)?;
    let req = view.render_request(&timeline, dataset.signal().clone(), Arc::new(SpikeEventStore::default()), engine::palette::Palette::DARK);
    let (frame, scale) = WaveformRenderer::default().render_scaled(&req);
    view.amp_scale = scale;
    println!("Source {} · scale bar {}", dataset.name, view.scale_bar_label());
    image::save_buffer(path, &frame.to_rgba(), w, h, image::ExtendedColorType::Rgba8)?;
    println!("Plot snapshot saved to {} ({w}x{h} px).", path.display());
    Ok(())
}

/// Parses `90`, `90s`, `10m`, `2h`, `1.5h` into seconds.
fn parse_duration(s: &str) -> Result<f64> {
    let s = s.trim();
    let (num, mult) = match s.chars().last() {
        Some('h') => (&s[..s.len() - 1], 3600.0),
        Some('m') => (&s[..s.len() - 1], 60.0),
        Some('s') => (&s[..s.len() - 1], 1.0),
        _ => (s, 1.0),
    };
    Ok(num.parse::<f64>()? * mult)
}
