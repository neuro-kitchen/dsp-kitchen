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
use data::{Dataset, SignalSource};
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
    /// Path to recording dataset file (.bin)
    #[arg(short, long)]
    file: Option<PathBuf>,

    /// Number of signal channels
    #[arg(short, long)]
    channels: Option<usize>,

    /// Acquisition sample rate in Hz
    #[arg(short = 'r', long)]
    sample_rate: Option<f64>,

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
    let dataset = Dataset::load_or_synthetic(args.file.clone(), args.channels, args.sample_rate)?;
    println!(
        "Loaded dataset: {} ({} channels, {} samples, {:.2}s @ {:.1} kHz)",
        dataset.name,
        dataset.total_channels,
        dataset.total_samples,
        dataset.total_duration_sec(),
        dataset.sample_rate / 1000.0
    );
    let path = args.file.clone().filter(|p| p.exists());
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
        let source: Arc<dyn SignalSource> = app.dataset.clone();
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
