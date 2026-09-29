mod renderer;
mod timeline;

use std::cell::RefCell;
use std::fs::File;
use std::path::PathBuf;
use std::rc::Rc;
use std::time::Instant;
use anyhow::{Context, Result};
use clap::Parser;
use memmap2::Mmap;
use serde::Deserialize;
use slint::{ComponentHandle, Image, SharedString, Timer, TimerMode};

use renderer::{render_waveforms_lod, RenderConfig};
use timeline::TimelineState;

slint::include_modules!();

#[derive(Parser, Debug)]
#[command(name = "croc-app")]
#[command(about = "High-Performance Multi-Channel Signal & Electrophysiology Viewer in Slint")]
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


#[derive(Deserialize, Debug)]
struct SidecarMetadata {
    channels: Option<usize>,
    samples: Option<usize>,
    sample_rate_hz: Option<f64>,
}

struct AppModel {
    raw_data: Vec<f32>,
    total_samples: usize,
    total_channels: usize,
    sample_rate: f64,
    timeline: TimelineState,
    amplitude_scale: f32,
    channel_offset: usize,
    visible_channels: usize,
    spike_times: Vec<f64>,
    spike_channels: Vec<usize>,
    canvas_width: u32,
    canvas_height: u32,
    last_frame_instant: Instant,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let args = Args::parse();

    // 1. Resolve dataset or generate synthetic continuous recording
    let default_bin = PathBuf::from("playground/data/mearec_32ch_10s.bin");
    let target_file = args.file.or_else(|| {
        if default_bin.exists() {
            Some(default_bin)
        } else {
            None
        }
    });

    let (raw_data, total_channels, total_samples, sample_rate, dataset_name) = match target_file {
        Some(path) => {
            let meta_path = path.with_extension("meta");
            let mut detected_ch = args.channels;
            let mut detected_s = None;
            let mut detected_sr = args.sample_rate;

            if meta_path.exists() {
                if let Ok(content) = std::fs::read_to_string(&meta_path) {
                    if let Ok(meta) = serde_json::from_str::<SidecarMetadata>(&content) {
                        if detected_ch.is_none() { detected_ch = meta.channels; }
                        if detected_s.is_none() { detected_s = meta.samples; }
                        if detected_sr.is_none() { detected_sr = meta.sample_rate_hz; }
                    }
                }
            }

            let channels = detected_ch.unwrap_or(32);
            let sample_rate = detected_sr.unwrap_or(32000.0);

            let file = File::open(&path)
                .with_context(|| format!("Failed to open dataset: {}", path.display()))?;
            let mmap = unsafe { Mmap::map(&file)? };

            let total_floats = mmap.len() / std::mem::size_of::<f32>();
            let samples = detected_s.unwrap_or(total_floats / channels);

            let slice = unsafe {
                std::slice::from_raw_parts(mmap.as_ptr() as *const f32, channels * samples)
            };

            let name = path.file_name()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_else(|| "dataset.bin".to_string());

            (slice.to_vec(), channels, samples, sample_rate, name)
        }
        None => {
            // Generate synthetic 32-channel electrophysiology test signal
            let channels = args.channels.unwrap_or(32);
            let sample_rate = args.sample_rate.unwrap_or(30000.0);
            let samples = (sample_rate * 5.0) as usize; // 5.0 seconds
            let mut buffer = vec![0.0f32; channels * samples];

            let dt = 1.0 / sample_rate;
            let omega_60 = 2.0 * std::f64::consts::PI * 60.0;
            for ch in 0..channels {
                let ch_offset = ch * samples;
                let phase = (ch as f64 * 0.1).fract() * 2.0 * std::f64::consts::PI;
                for s in 0..samples {
                    let t = s as f64 * dt;
                    let hum = (25.0 * (omega_60 * t + phase).sin()) as f32;
                    let noise = (((s * 37 + ch * 101) % 1000) as f32 / 1000.0 - 0.5) * 20.0;
                    buffer[ch_offset + s] = hum + noise;
                }

                // Inject action potentials every ~300ms
                let spike_interval = (sample_rate * 0.3) as usize;
                for sp in 1..(samples / spike_interval) {
                    let center = sp * spike_interval + (ch * 17) % 500;
                    if center + 40 < samples {
                        for i in 0..40 {
                            let t_rel = (i as f32 - 15.0) / 6.0;
                            let shape = -90.0 * (-0.5 * t_rel * t_rel).exp();
                            buffer[ch_offset + center + i] += shape;
                        }
                    }
                }
            }

            (buffer, channels, samples, sample_rate, "synthetic_32ch.bin".to_string())
        }
    };

    let total_duration = total_samples as f64 / sample_rate;
    println!("Loaded dataset: {} ({} channels, {} samples, {:.2}s @ {:.1} kHz)",
        dataset_name, total_channels, total_samples, total_duration, sample_rate / 1000.0
    );

    // 2. Pre-detect action potentials for the timeline event track
    println!("Extracting action potentials for timeline event track...");
    let mut spike_times = Vec::new();
    let mut spike_channels = Vec::new();

    for ch in 0..total_channels {
        let ch_offset = ch * total_samples;
        let channel_data = &raw_data[ch_offset..ch_offset + total_samples];

        // Simple threshold crossing detection on bandpass-like signal
        let mut noise_std = 15.0f32;
        if channel_data.len() > 1000 {
            let mut sum_sq = 0.0f64;
            for &val in &channel_data[0..1000] {
                sum_sq += (val as f64) * (val as f64);
            }
            noise_std = ((sum_sq / 1000.0).sqrt() as f32).max(5.0);
        }

        let threshold = -4.5 * noise_std;
        let mut in_refractory = 0usize;
        for (i, &sample) in channel_data.iter().enumerate() {
            if in_refractory > 0 {
                in_refractory -= 1;
                continue;
            }
            if sample < threshold {
                let t_sec = i as f64 / sample_rate;
                spike_times.push(t_sec);
                spike_channels.push(ch);
                in_refractory = (sample_rate * 0.002) as usize; // 2.0 ms refractory
            }
        }
    }
    println!("Detected {} timeline events across {} channels.", spike_times.len(), total_channels);

    // If --snapshot was requested, render waveform image to disk and exit cleanly
    if let Some(snap_path) = args.snapshot {
        println!("Exporting headless waveform snapshot to: {}", snap_path.display());
        let config = RenderConfig {
            raw_data: &raw_data,
            total_samples,
            total_channels,
            channel_offset: 0,
            visible_channels: 8,
            window_start_sec: 2.40,
            visible_window_sec: 0.100,
            sample_rate,
            amplitude_scale: 1.0,
            spike_times: &spike_times,
            spike_channels: &spike_channels,
        };
        let pixel_buf = render_waveforms_lod(1200, 600, &config);
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

    // 3. Initialize Slint Window

    let ui = AppWindow::new()?;
    let state = ui.global::<AppState>();
    let logic = ui.global::<AppLogic>();

    state.set_dataset_name(SharedString::from(dataset_name));
    state.set_channel_count(total_channels as i32);
    state.set_sample_rate(sample_rate as f32);
    state.set_total_duration_sec(total_duration as f32);
    state.set_total_spikes_detected(spike_times.len() as i32);
    state.set_visible_channels_count(8);
    state.set_amplitude_scale(1.0);

    let initial_timeline = TimelineState::new(total_duration);
    state.set_current_time_sec(initial_timeline.current_time_sec as f32);
    state.set_visible_window_sec(initial_timeline.visible_window_sec as f32);
    state.set_window_start_sec(initial_timeline.window_start_sec as f32);
    state.set_time_readout(SharedString::from(initial_timeline.format_time_readout()));
    state.set_channel_label_range(SharedString::from(format!("Ch 0 - Ch 7 (of {})", total_channels)));

    let model = Rc::new(RefCell::new(AppModel {
        raw_data,
        total_samples,
        total_channels,
        sample_rate,
        timeline: initial_timeline,
        amplitude_scale: 1.0,
        channel_offset: 0,
        visible_channels: 8,
        spike_times,
        spike_channels,
        canvas_width: 1200,
        canvas_height: 550,
        last_frame_instant: Instant::now(),
    }));

    // Helper to refresh waveform image and UI properties
    let render_update = {
        let ui_weak = ui.as_weak();
        let model = Rc::clone(&model);
        move || {
            let Some(ui) = ui_weak.upgrade() else { return };
            let m = model.borrow();


            let config = RenderConfig {
                raw_data: &m.raw_data,
                total_samples: m.total_samples,
                total_channels: m.total_channels,
                channel_offset: m.channel_offset,
                visible_channels: m.visible_channels,
                window_start_sec: m.timeline.window_start_sec,
                visible_window_sec: m.timeline.visible_window_sec,
                sample_rate: m.sample_rate,
                amplitude_scale: m.amplitude_scale,
                spike_times: &m.spike_times,
                spike_channels: &m.spike_channels,
            };

            let pixel_buf = render_waveforms_lod(m.canvas_width, m.canvas_height, &config);
            let image = Image::from_rgba8(pixel_buf);

            let state = ui.global::<AppState>();
            state.set_waveform_image(image);
            state.set_current_time_sec(m.timeline.current_time_sec as f32);
            state.set_window_start_sec(m.timeline.window_start_sec as f32);
            state.set_visible_window_sec(m.timeline.visible_window_sec as f32);
            state.set_time_readout(SharedString::from(m.timeline.format_time_readout()));
            state.set_is_playing(m.timeline.is_playing);
            state.set_playback_speed(m.timeline.playback_speed as f32);
            state.set_amplitude_scale(m.amplitude_scale);

            let end_ch = (m.channel_offset + m.visible_channels).min(m.total_channels);
            state.set_channel_label_range(SharedString::from(format!(
                "Ch {} - Ch {} (of {})",
                m.channel_offset,
                end_ch.saturating_sub(1),
                m.total_channels
            )));
        }
    };

    // Initial render
    render_update();

    // 4. Hook up AppLogic Callbacks
    {
        let model = Rc::clone(&model);
        let render_update = render_update.clone();
        logic.on_toggle_play_pause(move || {
            {
                let mut m = model.borrow_mut();
                m.timeline.toggle_play();
                m.last_frame_instant = Instant::now();
            }
            render_update();
        });
    }

    {
        let model = Rc::clone(&model);
        let render_update = render_update.clone();
        logic.on_step_forward(move || {
            {
                let mut m = model.borrow_mut();
                m.timeline.step_forward(0.033);
            }
            render_update();
        });
    }

    {
        let model = Rc::clone(&model);
        let render_update = render_update.clone();
        logic.on_step_backward(move || {
            {
                let mut m = model.borrow_mut();
                m.timeline.step_backward(0.033);
            }
            render_update();
        });
    }

    {
        let model = Rc::clone(&model);
        let render_update = render_update.clone();
        logic.on_jump_to_start(move || {
            {
                let mut m = model.borrow_mut();
                m.timeline.scrub_to(0.0);
            }
            render_update();
        });
    }

    {
        let model = Rc::clone(&model);
        let render_update = render_update.clone();
        logic.on_jump_to_end(move || {
            {
                let mut m = model.borrow_mut();
                let dur = m.timeline.total_duration_sec;
                m.timeline.scrub_to(dur);
            }
            render_update();
        });
    }

    {
        let model = Rc::clone(&model);
        let render_update = render_update.clone();
        logic.on_toggle_loop(move || {
            {
                let mut m = model.borrow_mut();
                m.timeline.toggle_loop();
            }
            render_update();
        });
    }

    {
        let model = Rc::clone(&model);
        let render_update = render_update.clone();
        logic.on_set_playback_speed(move |spd| {
            {
                let mut m = model.borrow_mut();
                m.timeline.playback_speed = spd as f64;
            }
            render_update();
        });
    }

    {
        let model = Rc::clone(&model);
        let render_update = render_update.clone();
        logic.on_set_window_duration(move |dur| {
            {
                let mut m = model.borrow_mut();
                m.timeline.set_window_duration(dur as f64);
            }
            render_update();
        });
    }

    {
        let model = Rc::clone(&model);
        let render_update = render_update.clone();
        logic.on_scrub_to_ratio(move |ratio| {
            {
                let mut m = model.borrow_mut();
                m.timeline.scrub_ratio(ratio as f64);
            }
            render_update();
        });
    }

    {
        let model = Rc::clone(&model);
        let render_update = render_update.clone();
        logic.on_zoom_time(move |factor| {
            {
                let mut m = model.borrow_mut();
                m.timeline.zoom_time(factor as f64);
            }
            render_update();
        });
    }

    {
        let model = Rc::clone(&model);
        let render_update = render_update.clone();
        logic.on_pan_time(move |dt| {
            {
                let mut m = model.borrow_mut();
                m.timeline.pan_time(dt as f64);
            }
            render_update();
        });
    }

    {
        let model = Rc::clone(&model);
        let render_update = render_update.clone();
        logic.on_zoom_amplitude(move |factor| {
            {
                let mut m = model.borrow_mut();
                m.amplitude_scale = (m.amplitude_scale * factor).clamp(0.1, 20.0);
            }
            render_update();
        });
    }

    {
        let model = Rc::clone(&model);
        let render_update = render_update.clone();
        logic.on_next_channel_page(move || {
            {
                let mut m = model.borrow_mut();
                if m.channel_offset + m.visible_channels < m.total_channels {
                    m.channel_offset += m.visible_channels;
                }
            }
            render_update();
        });
    }

    {
        let model = Rc::clone(&model);
        let render_update = render_update.clone();
        logic.on_prev_channel_page(move || {
            {
                let mut m = model.borrow_mut();
                m.channel_offset = m.channel_offset.saturating_sub(m.visible_channels);
            }
            render_update();
        });
    }

    {
        let model = Rc::clone(&model);
        let render_update = render_update.clone();
        logic.on_canvas_resized(move |w, h| {
            {
                let mut m = model.borrow_mut();
                m.canvas_width = (w / 1.0).max(100.0) as u32;
                m.canvas_height = (h / 1.0).max(100.0) as u32;
            }
            render_update();
        });
    }

    // 5. 60 FPS Animation & Playback Timer
    let timer = Timer::default();
    {
        let model = Rc::clone(&model);
        let render_update = render_update.clone();
        timer.start(
            TimerMode::Repeated,
            std::time::Duration::from_millis(16), // ~60 Hz
            move || {
                let is_playing = {
                    let mut m = model.borrow_mut();
                    let now = Instant::now();
                    let dt = now.duration_since(m.last_frame_instant).as_secs_f64();
                    m.last_frame_instant = now;

                    if m.timeline.is_playing {
                        m.timeline.advance(dt);
                        true
                    } else {
                        false
                    }
                };

                if is_playing {
                    render_update();
                }
            },
        );
    }

    ui.run()?;
    Ok(())
}
