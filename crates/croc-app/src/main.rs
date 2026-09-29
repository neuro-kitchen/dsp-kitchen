mod model;
mod view;
mod viewmodel;

use std::cell::RefCell;
use std::path::PathBuf;
use std::rc::Rc;
use anyhow::Result;
use clap::Parser;
use slint::{ComponentHandle, Image, SharedString, Timer, TimerMode};

use model::Dataset;
use viewmodel::AppViewModel;

slint::include_modules!();

#[derive(Parser, Debug)]
#[command(name = "croc-app")]
#[command(about = "High-Performance Multi-Channel Signal & Electrophysiology Viewer in Slint (MVVM Architecture)")]
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

    /// Optional path to export a rendered waveform snapshot PNG and exit
    #[arg(long)]
    snapshot: Option<PathBuf>,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let args = Args::parse();

    // 1. Load Model (Dataset) and initialize ViewModel
    let dataset = Dataset::load_or_synthetic(args.file, args.channels, args.sample_rate)?;
    println!(
        "Loaded dataset: {} ({} channels, {} samples, {:.2}s @ {:.1} kHz)",
        dataset.name,
        dataset.total_channels,
        dataset.total_samples,
        dataset.total_duration_sec(),
        dataset.sample_rate / 1000.0
    );

    let mut initial_vm = AppViewModel::new(dataset);
    println!(
        "Detected {} timeline events across {} channels.",
        initial_vm.events.len(),
        initial_vm.dataset.total_channels
    );

    // 2. Optional Headless Snapshot Export
    if let Some(snap_path) = args.snapshot {
        println!("Exporting headless waveform snapshot to: {}", snap_path.display());
        initial_vm.timeline.scrub_to(2.45);
        let pixel_buf = initial_vm.render_waveform_buffer(1200, 600);
        image::save_buffer(
            &snap_path,
            pixel_buf.as_bytes(),
            1200,
            600,
            image::ExtendedColorType::Rgba8,
        )?;
        println!("Waveform snapshot successfully saved (1200x600 px).");
        return Ok(());
    }

    // 3. Initialize Slint View and Bind ViewModel State
    let ui = AppWindow::new()?;
    let state = ui.global::<AppState>();
    let logic = ui.global::<AppLogic>();

    state.set_dataset_name(SharedString::from(&initial_vm.dataset.name));
    state.set_channel_count(initial_vm.dataset.total_channels as i32);
    state.set_sample_rate(initial_vm.dataset.sample_rate as f32);
    state.set_total_duration_sec(initial_vm.timeline.total_duration_sec as f32);
    state.set_total_spikes_detected(initial_vm.events.len() as i32);
    state.set_visible_channels_count(initial_vm.visible_channels as i32);

    let vm = Rc::new(RefCell::new(initial_vm));

    // Synchronizes ViewModel state with Slint AppState
    let sync_ui = {
        let ui_weak = ui.as_weak();
        let vm = Rc::clone(&vm);
        move || {
            let Some(ui) = ui_weak.upgrade() else { return };
            let v = vm.borrow();

            let pixel_buf = v.render_waveform_buffer(v.canvas_width, v.canvas_height);
            let image = Image::from_rgba8(pixel_buf);

            let state = ui.global::<AppState>();
            state.set_waveform_image(image);
            state.set_current_time_sec(v.timeline.current_time_sec as f32);
            state.set_window_start_sec(v.timeline.window_start_sec as f32);
            state.set_visible_window_sec(v.timeline.visible_window_sec as f32);
            state.set_time_readout(SharedString::from(v.time_readout()));
            state.set_is_playing(v.timeline.is_playing);
            state.set_loop_playback(v.timeline.loop_playback);
            state.set_playback_speed(v.timeline.playback_speed as f32);
            state.set_amplitude_scale(v.amplitude_scale);
            state.set_channel_label_range(SharedString::from(v.channel_label_range()));
        }
    };

    // Initial render
    sync_ui();

    // 4. Bind View Intents (AppLogic Callbacks) -> ViewModel
    {
        let vm = Rc::clone(&vm);
        let sync_ui = sync_ui.clone();
        logic.on_toggle_play_pause(move || {
            vm.borrow_mut().on_toggle_play();
            sync_ui();
        });
    }

    {
        let vm = Rc::clone(&vm);
        let sync_ui = sync_ui.clone();
        logic.on_step_forward(move || {
            vm.borrow_mut().on_step_forward();
            sync_ui();
        });
    }

    {
        let vm = Rc::clone(&vm);
        let sync_ui = sync_ui.clone();
        logic.on_step_backward(move || {
            vm.borrow_mut().on_step_backward();
            sync_ui();
        });
    }

    {
        let vm = Rc::clone(&vm);
        let sync_ui = sync_ui.clone();
        logic.on_jump_to_start(move || {
            vm.borrow_mut().on_jump_to_start();
            sync_ui();
        });
    }

    {
        let vm = Rc::clone(&vm);
        let sync_ui = sync_ui.clone();
        logic.on_jump_to_end(move || {
            vm.borrow_mut().on_jump_to_end();
            sync_ui();
        });
    }

    {
        let vm = Rc::clone(&vm);
        let sync_ui = sync_ui.clone();
        logic.on_toggle_loop(move || {
            vm.borrow_mut().on_toggle_loop();
            sync_ui();
        });
    }

    {
        let vm = Rc::clone(&vm);
        let sync_ui = sync_ui.clone();
        logic.on_set_playback_speed(move |spd| {
            vm.borrow_mut().on_set_playback_speed(spd);
            sync_ui();
        });
    }

    {
        let vm = Rc::clone(&vm);
        let sync_ui = sync_ui.clone();
        logic.on_set_window_duration(move |dur| {
            vm.borrow_mut().on_set_window_duration(dur);
            sync_ui();
        });
    }

    {
        let vm = Rc::clone(&vm);
        let sync_ui = sync_ui.clone();
        logic.on_scrub_to_ratio(move |ratio| {
            vm.borrow_mut().on_scrub_to_ratio(ratio);
            sync_ui();
        });
    }

    {
        let vm = Rc::clone(&vm);
        let sync_ui = sync_ui.clone();
        logic.on_zoom_time(move |factor| {
            vm.borrow_mut().on_zoom_time(factor);
            sync_ui();
        });
    }

    {
        let vm = Rc::clone(&vm);
        let sync_ui = sync_ui.clone();
        logic.on_pan_time(move |dt| {
            vm.borrow_mut().on_pan_time(dt);
            sync_ui();
        });
    }

    {
        let vm = Rc::clone(&vm);
        let sync_ui = sync_ui.clone();
        logic.on_zoom_amplitude(move |factor| {
            vm.borrow_mut().on_zoom_amplitude(factor);
            sync_ui();
        });
    }

    {
        let vm = Rc::clone(&vm);
        let sync_ui = sync_ui.clone();
        logic.on_next_channel_page(move || {
            vm.borrow_mut().on_next_channel_page();
            sync_ui();
        });
    }

    {
        let vm = Rc::clone(&vm);
        let sync_ui = sync_ui.clone();
        logic.on_prev_channel_page(move || {
            vm.borrow_mut().on_prev_channel_page();
            sync_ui();
        });
    }

    {
        let vm = Rc::clone(&vm);
        let sync_ui = sync_ui.clone();
        logic.on_canvas_resized(move |w, h| {
            vm.borrow_mut().on_canvas_resized(w, h);
            sync_ui();
        });
    }

    // 5. 60 FPS Animation & Playback Timer
    let timer = Timer::default();
    {
        let vm = Rc::clone(&vm);
        let sync_ui = sync_ui.clone();
        timer.start(
            TimerMode::Repeated,
            std::time::Duration::from_millis(16), // ~60 Hz
            move || {
                let changed = vm.borrow_mut().on_tick();
                if changed {
                    sync_ui();
                }
            },
        );
    }

    ui.run()?;
    Ok(())
}
