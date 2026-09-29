use cubecl::prelude::*;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use dsp_base::filter::{design_notch_coeffs, execute_notch, execute_teager_kaiser};
use dsp_base::math::execute_scaling;
use std::fs;
use std::path::Path;
use std::time::Instant;

pub fn run_benchmark_pipeline(
    channels: usize,
    samples: usize,
    iterations: usize,
    save: bool,
    format: &str,
    output_path: &Path,
) -> anyhow::Result<()> {
    println!("============================================================");
    println!("       dsp-kitchen Real-Time Pipeline Benchmark             ");
    println!("============================================================");
    let total_elements = channels * samples;
    let data_bytes = total_elements * std::mem::size_of::<f32>();
    println!("Workload:           {} channels × {} samples", channels, samples);
    println!(
        "Buffer Size:        {:.2} MB ({} float32 samples)",
        data_bytes as f64 / (1024.0 * 1024.0),
        total_elements
    );
    println!(
        "Persistence Mode:   {}",
        if save {
            format!("ENABLED ({format})")
        } else {
            "DISABLED (Pure In-VRAM Chaining)".into()
        }
    );
    println!("Iterations:         {}", iterations);

    let device = WgpuDevice::default();
    let client = WgpuRuntime::client(&device);
    println!("Compute Runtime:    {}", WgpuRuntime::name(&client));

    let input_host = vec![10.0f32; total_elements];
    let input_bytes = unsafe {
        std::slice::from_raw_parts(
            input_host.as_ptr() as *const u8,
            input_host.len() * std::mem::size_of::<f32>(),
        )
    };

    let buf_a = client.create_from_slice(input_bytes);
    let buf_b = client.empty(data_bytes);
    let notch_coeffs = design_notch_coeffs(60.0, 30000.0, 30.0);

    // Warmup JIT compilation
    execute_scaling::<WgpuRuntime>(&client, &buf_a, &buf_b, total_elements, 1.0, 0.0, false);
    execute_notch::<WgpuRuntime>(&client, &buf_b, &buf_a, notch_coeffs, channels, samples, false);
    execute_teager_kaiser::<WgpuRuntime>(&client, &buf_a, &buf_b, channels, samples, false);

    let mut kernel_durations = Vec::with_capacity(iterations);
    let mut total_durations = Vec::with_capacity(iterations);
    let mut upload_durations = Vec::with_capacity(iterations);
    let mut download_durations = Vec::with_capacity(iterations);
    let mut save_durations = Vec::with_capacity(iterations);

    for _ in 0..iterations {
        let total_start = Instant::now();

        // Upload
        let t_upload_start = Instant::now();
        let in_buf = client.create_from_slice(input_bytes);
        upload_durations.push(t_upload_start.elapsed());

        // Chained In-VRAM Kernels (Zero Host Roundtrips)
        let t_kernel_start = Instant::now();
        execute_scaling::<WgpuRuntime>(&client, &in_buf, &buf_b, total_elements, 0.195, 0.0, false);
        execute_notch::<WgpuRuntime>(&client, &buf_b, &in_buf, notch_coeffs, channels, samples, false);
        execute_teager_kaiser::<WgpuRuntime>(&client, &in_buf, &buf_b, channels, samples, false);
        kernel_durations.push(t_kernel_start.elapsed());

        // Download
        let t_down_start = Instant::now();
        let out_bytes = client.read_one_unchecked(buf_b.clone());
        download_durations.push(t_down_start.elapsed());

        // Optional Persistence
        let mut save_elapsed = std::time::Duration::ZERO;
        if save {
            let t_save_start = Instant::now();
            if format == "zarr" {
                let zarr_path = if output_path.extension().map_or(false, |ext| ext == "bin") {
                    output_path.with_extension("zarr")
                } else {
                    output_path.to_path_buf()
                };
                let out_slice = unsafe {
                    std::slice::from_raw_parts(
                        out_bytes.as_ptr() as *const f32,
                        out_bytes.len() / std::mem::size_of::<f32>(),
                    )
                };
                dsp_stream::create_zarr_recording(&zarr_path, channels, samples, 30000.0, out_slice)?;
            } else {
                if let Some(parent) = output_path.parent() {
                    fs::create_dir_all(parent)?;
                }
                fs::write(output_path, &*out_bytes)?;
            }
            save_elapsed = t_save_start.elapsed();
        }
        save_durations.push(save_elapsed);
        total_durations.push(total_start.elapsed());
    }

    let avg_kernel_ms = kernel_durations.iter().map(|d| d.as_secs_f64() * 1000.0).sum::<f64>() / iterations as f64;
    let avg_upload_ms = upload_durations.iter().map(|d| d.as_secs_f64() * 1000.0).sum::<f64>() / iterations as f64;
    let avg_download_ms = download_durations.iter().map(|d| d.as_secs_f64() * 1000.0).sum::<f64>() / iterations as f64;
    let avg_save_ms = save_durations.iter().map(|d| d.as_secs_f64() * 1000.0).sum::<f64>() / iterations as f64;
    let avg_total_ms = total_durations.iter().map(|d| d.as_secs_f64() * 1000.0).sum::<f64>() / iterations as f64;

    let compute_throughput_msamples = (total_elements as f64 / 1_000_000.0) / (avg_kernel_ms / 1000.0);
    let sample_duration_sec = samples as f64 / 30000.0;
    let real_time_factor = (sample_duration_sec * 1000.0) / avg_kernel_ms;

    println!("\n[Benchmark Results Summary]");
    println!("  In-VRAM Kernel Time:   {:.3} ms  (Scaling -> Notch -> TKEO)", avg_kernel_ms);
    println!(
        "  Compute Throughput:    {:.1} MSamples/sec ({:.2} GB/s)",
        compute_throughput_msamples,
        compute_throughput_msamples * 4.0 / 1024.0
    );
    println!("  Real-Time Multiplier:  {:.1}x realtime factor", real_time_factor);

    if save {
        let bus_io_time = avg_upload_ms + avg_download_ms + avg_save_ms;
        let bus_overhead_pct = (bus_io_time / avg_total_ms) * 100.0;

        println!("\n[Data Movement & Persistence Overhead]");
        println!("  Host -> Device Upload: {:.3} ms", avg_upload_ms);
        println!("  Device -> Host Down:   {:.3} ms", avg_download_ms);
        println!("  Storage Write Time:    {:.3} ms", avg_save_ms);
        println!("  Total End-to-End:      {:.3} ms", avg_total_ms);
        println!("  I/O & Bus Penalty:     {:.1}% of runtime spent moving data!", bus_overhead_pct);
    } else {
        println!("\n[Pipeline Efficiency]");
        println!("  Host Roundtrips:       0 (Zero PCIe downloads between stages)");
        println!("  VRAM Data Movement:    Chained directly in persistent device buffers");
    }

    println!("============================================================");
    Ok(())
}

pub fn run_benchmark_sweep(samples_per_channel: usize, iterations: usize) -> anyhow::Result<()> {
    println!("=========================================================================================");
    println!("             dsp-kitchen Dynamic Channel Scaling Sweep Benchmark                         ");
    println!("=========================================================================================");
    println!("Samples per channel: {}", samples_per_channel);
    println!("Benchmark iterations: {}", iterations);
    println!("Pipeline: Scale (ADC) -> Notch (60 Hz) -> Teager-Kaiser Energy Operator (TKEO)");
    println!("-----------------------------------------------------------------------------------------");
    println!(
        "{:<10} | {:<12} | {:<14} | {:<18} | {:<14} | {:<10}",
        "Channels", "Elements", "Kernel Time", "Throughput", "Bandwidth", "Realtime"
    );
    println!("-----------------------------------------------------------------------------------------");

    let device = WgpuDevice::default();
    let client = WgpuRuntime::client(&device);
    let notch_coeffs = design_notch_coeffs(60.0, 30000.0, 30.0);

    let channel_counts = [1, 4, 16, 32, 64, 128, 384, 1024];

    for &channels in &channel_counts {
        let total_elements = channels * samples_per_channel;
        let data_bytes = total_elements * std::mem::size_of::<f32>();

        let input_host = vec![1.0f32; total_elements];
        let input_bytes = unsafe {
            std::slice::from_raw_parts(
                input_host.as_ptr() as *const u8,
                input_host.len() * std::mem::size_of::<f32>(),
            )
        };

        let buf_a = client.create_from_slice(input_bytes);
        let buf_b = client.empty(data_bytes);

        // Warmup
        execute_scaling::<WgpuRuntime>(&client, &buf_a, &buf_b, total_elements, 1.0, 0.0, false);
        execute_notch::<WgpuRuntime>(&client, &buf_b, &buf_a, notch_coeffs, channels, samples_per_channel, false);
        execute_teager_kaiser::<WgpuRuntime>(&client, &buf_a, &buf_b, channels, samples_per_channel, false);

        let mut durations = Vec::with_capacity(iterations);
        for _ in 0..iterations {
            let start = Instant::now();
            execute_scaling::<WgpuRuntime>(&client, &buf_a, &buf_b, total_elements, 0.195, 0.0, false);
            execute_notch::<WgpuRuntime>(&client, &buf_b, &buf_a, notch_coeffs, channels, samples_per_channel, false);
            execute_teager_kaiser::<WgpuRuntime>(&client, &buf_a, &buf_b, channels, samples_per_channel, false);
            durations.push(start.elapsed());
        }

        let avg_ms = durations.iter().map(|d| d.as_secs_f64() * 1000.0).sum::<f64>() / iterations as f64;
        let throughput_msamples = (total_elements as f64 / 1_000_000.0) / (avg_ms / 1000.0);
        let bandwidth_gbs = (throughput_msamples * 4.0) / 1024.0;
        let stream_duration_ms = (samples_per_channel as f64 / 30000.0) * 1000.0;
        let realtime_factor = stream_duration_ms / avg_ms;

        println!(
            "{:<10} | {:<12} | {:<11.3} ms | {:<13.1} MS/s | {:<10.2} GB/s | {:<8.1}x",
            channels, total_elements, avg_ms, throughput_msamples, bandwidth_gbs, realtime_factor
        );
    }

    println!("-----------------------------------------------------------------------------------------");
    println!("Dynamic channel scaling benchmark completed successfully.");
    println!("=========================================================================================");
    Ok(())
}
