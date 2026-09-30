mod app;
mod data;
mod modules;
mod shared;

use std::path::PathBuf;
use std::sync::Arc;
use anyhow::Result;
use clap::Parser;
use slint::ComponentHandle;

use app::controller::Controller;
use app::model::AppModel;
use data::Dataset;
use dsp_core::RecordingSource;
use modules::time::module::TimeModule;
use modules::time::renderer::{TimeViewKind, WaveformRenderer};

/// Generated Slint UI (window, globals, structs).
mod ui {
    slint::include_modules!();
}

#[derive(Parser, Debug)]
#[command(name = "croc-app")]
#[command(about = "Croc: electrophysiology signal workbench — Time and Spikes modules (Slint, MVVM)")]
struct Args {
    /// Recording to open: SpikeGLX .bin/.cbin, raw .bin with a JSON .meta, or a Zarr store.
    /// Without it (and without --synthetic) the last opened recording is reopened.
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

    /// Optional path to export a rendered plot snapshot PNG and exit
    #[arg(long)]
    snapshot: Option<PathBuf>,

    /// Render the snapshot in heatmap mode
    #[arg(long)]
    heatmap: bool,

    /// Open the window, capture it to this PNG once views have rendered, and exit
    #[arg(long)]
    screenshot: Option<PathBuf>,

    /// Window size for --screenshot, e.g. 1280x800 (logical px)
    #[arg(long, default_value = "1280x800")]
    size: String,

    /// Start in this module (0 = Time, 1 = Spikes)
    #[arg(long)]
    module: Option<usize>,

    /// Neither restore nor save the workspace layout (--screenshot restores but never saves)
    #[arg(long)]
    no_session: bool,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let args = Args::parse();

    // 1. Load Model (Dataset) and initialize ViewModel
    let (dataset, path) = initial_dataset(&args)?;
    println!(
        "Loaded dataset: {} ({} channels, {} samples, {:.2}s @ {:.1} kHz)",
        dataset.name,
        dataset.total_channels,
        dataset.total_samples,
        dataset.total_duration_sec(),
        dataset.sample_rate / 1000.0
    );
    let mut app = AppModel::new(dataset, path.clone());
    if let Some(p) = path {
        app.push_recent(p);
    }
    println!("Detected {} timeline events across {} channels.", app.events.len(), app.dataset.total_channels);

    // 2. Optional headless snapshot of the default traces (or heatmap) view
    if let Some(snap_path) = args.snapshot {
        let (w, h) = (1200u32, 600u32);
        let mut time = TimeModule::new(&app.dataset);
        time.timeline.scrub_to(2.45);
        let kind = if args.heatmap { TimeViewKind::Heatmap } else { TimeViewKind::Traces };
        let mut view = time.ws.views.iter().find(|v| v.kind == kind).expect("default layout has both kinds").clone();
        view.set_canvas(w, h, 1.0);
        let source: Arc<dyn RecordingSource> = app.dataset.clone();
        let req = view.render_request(&time.timeline, source, app.events.clone(), Vec::new());
        let pixel_buf = WaveformRenderer::default().render(&req);
        image::save_buffer(&snap_path, pixel_buf.as_bytes(), w, h, image::ExtendedColorType::Rgba8)?;
        println!("Plot snapshot saved to {} ({w}x{h} px).", snap_path.display());
        return Ok(());
    }

    // 3. Window + controller (restores the saved layout, wires intents, starts the frame timer)
    let ui = ui::AppWindow::new()?;
    // A screenshot restores the saved layout but never overwrites it
    let restore = !args.no_session;
    let save = restore && args.screenshot.is_none();
    let _controller = Controller::install(&ui, app, restore, save);
    if let Some(module) = args.module {
        ui.global::<ui::AppLogic>().invoke_set_module(module as i32);
    }

    let _capture = args.screenshot.map(|path| {
        let (w, h) = args.size.split_once('x').and_then(|(w, h)| Some((w.parse().ok()?, h.parse().ok()?))).unwrap_or((1280.0, 800.0));
        ui.window().set_size(slint::LogicalSize::new(w, h));
        let weak = ui.as_weak();
        let timer = slint::Timer::default();
        // Enough time for the first layout pass and the worker's first frames
        timer.start(slint::TimerMode::SingleShot, std::time::Duration::from_millis(1500), move || {
            if let Some(ui) = weak.upgrade() {
                match ui.window().take_snapshot() {
                    Ok(buf) => {
                        let saved = image::save_buffer(&path, buf.as_bytes(), buf.width(), buf.height(), image::ExtendedColorType::Rgba8);
                        println!("Screenshot {} ({}x{} px): {saved:?}", path.display(), buf.width(), buf.height());
                    }
                    Err(e) => eprintln!("Screenshot failed: {e}"),
                }
            }
            let _ = slint::quit_event_loop();
        });
        timer
    });

    ui.run()?;
    Controller::shutdown();
    Ok(())
}

/// `--synthetic`, else `--file`, else the last opened recording, the bundled 10 s sample, or
/// a minute of procedural data — whichever opens first.
fn initial_dataset(args: &Args) -> Result<(Dataset, Option<PathBuf>)> {
    if let Some(d) = &args.synthetic {
        return Ok((Dataset::procedural(args.channels, args.sample_rate, parse_duration(d)?)?, None));
    }
    if let Some(p) = &args.file {
        return Ok((Dataset::open(p)?, Some(p.clone())));
    }
    let last = app::session::Session::load().filter(|_| !args.no_session).and_then(|s| s.recent.into_iter().next());
    let bundled = PathBuf::from("playground/data/mearec_32ch_10s.bin");
    for p in last.into_iter().chain([bundled]) {
        match Dataset::open(&p) {
            Ok(ds) => return Ok((ds, Some(p))),
            Err(e) => tracing::warn!("{e:#}"),
        }
    }
    Ok((Dataset::procedural(32, 30_000.0, 60.0)?, None))
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
