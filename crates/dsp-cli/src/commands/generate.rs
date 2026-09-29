use memmap2::Mmap;
use serde::Serialize;
use std::fs::{self, File};
use std::io::Write;
use std::path::Path;

#[derive(Serialize)]
struct RecordingMetadata {
    channels: usize,
    samples: usize,
    sample_rate_hz: f64,
    duration_seconds: f64,
    noise_rms_uv: f32,
    line_noise_uv: f32,
    spikes_injected: usize,
    format: &'static str,
    byte_order: &'static str,
    bytes_per_sample: usize,
    total_data_bytes: usize,
}

pub fn run_generate(
    channels: usize,
    samples: usize,
    sample_rate: f64,
    noise_uv: f32,
    line_noise_uv: f32,
    spikes: bool,
    format: &str,
    output_path: &Path,
) -> anyhow::Result<()> {
    println!("=== Synthetic Multi-Channel Signal Generator ===");
    println!("Channels:            {}", channels);
    println!("Samples per channel: {}", samples);
    println!("Sample Rate:         {:.1} Hz", sample_rate);
    println!("Duration:            {:.3} seconds", samples as f64 / sample_rate);
    println!("Gaussian Noise RMS:  {:.1} uV", noise_uv);
    println!("60 Hz Line Noise:    {:.1} uV", line_noise_uv);
    println!("Injecting Spikes:    {}", spikes);
    println!("Storage Format:      {}", format);
    println!("Destination:         {}", output_path.display());

    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent)?;
    }

    let total_elements = channels * samples;
    let mut buffer = vec![0.0f32; total_elements];

    // Deterministic pseudo-random generator (Xorshift64 + Box-Muller transform)
    let mut rng_state: u64 = 0x853c49e6748fea9b;
    let mut next_uniform = || -> f32 {
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 7;
        rng_state ^= rng_state << 17;
        (rng_state as f64 / u64::MAX as f64) as f32
    };

    let dt = 1.0 / sample_rate;
    let omega_60hz = 2.0 * std::f64::consts::PI * 60.0;

    // 1. Generate baseline + 60Hz line interference + Gaussian noise
    for ch in 0..channels {
        let channel_offset = ch * samples;
        let ch_phase = (ch as f64 * 0.05).fract() * 2.0 * std::f64::consts::PI;

        for s in 0..samples {
            let t = s as f64 * dt;
            let line_val = (line_noise_uv as f64 * (omega_60hz * t + ch_phase).sin()) as f32;

            let u1 = next_uniform().max(1e-7);
            let u2 = next_uniform();
            let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
            let noise_val = z0 * noise_uv;

            buffer[channel_offset + s] = line_val + noise_val;
        }
    }

    // 2. Inject action potentials (spikes) if requested
    let mut spikes_injected = 0usize;
    if spikes && samples > 100 {
        let spike_len = 60usize;
        let mut spike_shape = vec![0.0f32; spike_len];
        for i in 0..spike_len {
            let t_rel = (i as f32 - 15.0) / 8.0;
            spike_shape[i] =
                -120.0 * (-0.5 * t_rel * t_rel).exp() + 35.0 * (-0.5 * (t_rel - 1.8).powi(2)).exp();
        }

        let num_spike_times = (samples / 600).max(1);
        for st in 1..num_spike_times {
            let center_sample = st * 500;
            if center_sample + spike_len >= samples {
                break;
            }

            let span = channels.min(5);
            let primary_ch = if channels > span {
                (st * 37) % (channels - span)
            } else {
                0
            };

            for d_ch in 0..span {
                let ch = primary_ch + d_ch;
                let ch_offset = ch * samples;
                let attenuation = 1.0 / (1.0 + (d_ch as f32 * 0.8).powi(2));

                for i in 0..spike_len {
                    buffer[ch_offset + center_sample + i] += spike_shape[i] * attenuation;
                }
            }
            spikes_injected += 1;
        }
    }

    // Storage formatting
    if format == "zarr" {
        let zarr_path = if output_path.extension().map_or(false, |ext| ext == "bin") {
            output_path.with_extension("zarr")
        } else {
            output_path.to_path_buf()
        };

        println!("\n[Storing via zarrs Engine (Zarr v3 Format)]");
        dsp_stream::create_zarr_recording(
            &zarr_path,
            channels,
            samples,
            sample_rate,
            &buffer,
        )?;

        let total_bytes = channels * samples * std::mem::size_of::<f32>();
        println!("\n[Zarr Dataset Generation Complete]");
        println!("  Store Directory: {}", zarr_path.display());
        println!("  Array Path:      /traces");
        println!("  Array Shape:     [{} channels, {} samples]", channels, samples);
        println!("  Chunking Grid:   [{} channels, {} samples]", channels, 5000.min(samples));
        println!(
            "  Total Size:      {:.2} MB ({} floats)",
            total_bytes as f64 / (1024.0 * 1024.0),
            total_elements
        );
        println!("  Spikes Injected: {}", spikes_injected);

        // Verification Readback
        println!("\n[zarrs Verification Read]");
        let (read_data, r_ch, r_s, r_sr) = dsp_stream::read_zarr_recording(&zarr_path)?;
        println!(
            "  Zarr Readback:   SUCCESS (retrieved {} channels, {} samples @ {:.1} Hz)",
            r_ch, r_s, r_sr
        );
        println!("  Sample (ch0, s0): {:.2} uV", read_data[0]);
        return Ok(());
    }

    // Write flat binary data
    let mut file = File::create(output_path)?;
    let byte_slice = unsafe {
        std::slice::from_raw_parts(
            buffer.as_ptr() as *const u8,
            buffer.len() * std::mem::size_of::<f32>(),
        )
    };
    file.write_all(byte_slice)?;
    file.flush()?;

    let total_bytes = byte_slice.len();

    // Write JSON metadata sidecar
    let meta_path = output_path.with_extension("meta");
    let metadata = RecordingMetadata {
        channels,
        samples,
        sample_rate_hz: sample_rate,
        duration_seconds: samples as f64 / sample_rate,
        noise_rms_uv: noise_uv,
        line_noise_uv: line_noise_uv,
        spikes_injected,
        format: "float32-le",
        byte_order: "little-endian",
        bytes_per_sample: std::mem::size_of::<f32>(),
        total_data_bytes: total_bytes,
    };
    let meta_json = serde_json::to_string_pretty(&metadata)?;
    fs::write(&meta_path, meta_json)?;

    println!("\n[Generation Complete]");
    println!(
        "  Data File:      {} ({:.2} MB)",
        output_path.display(),
        total_bytes as f64 / (1024.0 * 1024.0)
    );
    println!("  Metadata File:  {}", meta_path.display());
    println!("  Spikes Injected: {}", spikes_injected);

    // Verify mmap
    println!("\n[Zero-Copy mmap2 Verification]");
    let verify_file = File::open(output_path)?;
    let mmap = unsafe { Mmap::map(&verify_file)? };
    println!(
        "  Memory Mapped:  SUCCESS ({} bytes mapped to host address {:p})",
        mmap.len(),
        mmap.as_ptr()
    );

    Ok(())
}
