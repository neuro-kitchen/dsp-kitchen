use std::fs::File;
use std::io::Write;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::time::{Duration, Instant};
use anyhow::{Context, Result};


use dsp_stream::network::{
    make_client_config_with_cert, make_insecure_client_config, QuicStreamClient,
};

/// High-precision statistics accumulator using Welford's algorithm for online variance/jitter calculation.
#[derive(Default)]
struct JitterStats {
    count: u64,
    mean: f64,
    m2: f64,
    min: f64,
    max: f64,
    samples: Vec<f64>,
}

impl JitterStats {
    fn new() -> Self {
        Self {
            count: 0,
            mean: 0.0,
            m2: 0.0,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
            samples: Vec::with_capacity(50_000),
        }
    }

    fn update(&mut self, val_ms: f64) {
        self.count += 1;
        let delta = val_ms - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = val_ms - self.mean;
        self.m2 += delta * delta2;

        if val_ms < self.min {
            self.min = val_ms;
        }
        if val_ms > self.max {
            self.max = val_ms;
        }

        if self.samples.len() < 100_000 {
            self.samples.push(val_ms);
        }
    }

    fn std_dev(&self) -> f64 {
        if self.count < 2 {
            0.0
        } else {
            (self.m2 / (self.count - 1) as f64).sqrt()
        }
    }

    fn percentile(&mut self, pct: f64) -> f64 {
        if self.samples.is_empty() {
            return 0.0;
        }
        self.samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let idx = ((self.samples.len() as f64 * pct / 100.0) as usize).min(self.samples.len() - 1);
        self.samples[idx]
    }
}

/// Connect to a QUIC stream server and benchmark transmission speed, latency, jitter, and packet loss.
pub async fn run_receive(
    server_addr: SocketAddr,
    server_name: String,
    cert_path: Option<PathBuf>,
    insecure: bool,
    duration_secs: f64,
    max_frames: u64,
    save_path: Option<PathBuf>,
) -> Result<()> {
    println!("============================================================");
    println!("        DSP-KITCHEN QUIC STREAMING BENCHMARK CLIENT         ");
    println!("============================================================");
    println!("  [Target]      Server Address:      {}", server_addr);
    println!("  [TLS]         Server Name:         {}", server_name);
    println!(
        "  [Duration]    Limit:               {}",
        if duration_secs > 0.0 {
            format!("{:.1} seconds", duration_secs)
        } else {
            "Unlimited (Ctrl+C to stop)".to_string()
        }
    );
    if max_frames > 0 {
        println!("  [Frames]      Max Frames:          {}", max_frames);
    }
    if let Some(ref path) = save_path {
        println!("  [Output]      Saving Stream To:    {}", path.display());
    }
    println!("------------------------------------------------------------");

    // 1. Configure TLS
    let client_config = match cert_path {
        Some(path) => {
            println!("  [TLS]         Loading custom cert: {}", path.display());
            let cert_der = std::fs::read(&path)
                .with_context(|| format!("Failed to read certificate: {}", path.display()))?;
            make_client_config_with_cert(&cert_der)
                .map_err(|e| anyhow::anyhow!("Failed to build TLS client config from cert: {e}"))?
        }
        None if !insecure => {
            anyhow::bail!(
                "no server certificate given: pass --cert <server.der> (from `serve --cert-out`), \
                 or --insecure to skip verification on a trusted network"
            );
        }
        None => {
            println!("  [TLS]         INSECURE: server certificate is not verified (--insecure)");
            make_insecure_client_config()
                .map_err(|e| anyhow::anyhow!("Failed to build insecure TLS client config: {e}"))?
        }
    };


    // 2. Connect to QUIC server
    println!("  [Connection]  Initiating QUIC handshake...");
    let t_connect_start = Instant::now();
    let client = QuicStreamClient::bind(client_config)
        .context("Failed to bind QUIC client UDP endpoint")?;

    let conn = client
        .connect(server_addr, &server_name)
        .await
        .with_context(|| format!("Failed to connect to QUIC server at {}", server_addr))?;

    let handshake_duration = t_connect_start.elapsed();
    println!(
        "  [Connection]  CONNECTED! Handshake RTT: {:.2} ms",
        handshake_duration.as_secs_f64() * 1000.0
    );

    println!("  [Stream]      Awaiting bidirectional stream from server...");
    let (mut client_send, mut recv_stream) = conn
        .accept_bi()
        .await
        .context("Failed to accept bidirectional QUIC stream from server")?;

    println!("  [Status]      Stream established. Receiving frames...\n");

    // 3. Reception & Benchmarking loop
    let mut frames_received = 0u64;
    let mut frames_dropped = 0u64;
    let mut last_seq: Option<u64> = None;
    let mut total_bytes = 0usize;
    let mut total_samples_per_ch = 0u64;
    let mut channels = 0u32;
    let mut samples_per_frame = 0u32;
    let mut sample_rate_hz = 0.0f64;

    let mut saved_chunks: Vec<Vec<f32>> = Vec::new();
    let mut jitter_stats = JitterStats::new();

    let benchmark_start = Instant::now();
    let mut last_arrival: Option<Instant> = None;
    let mut last_report_time = Instant::now();
    let mut last_report_frames = 0u64;
    let mut last_report_bytes = 0usize;

    let mut _terminated_early = false;

    // Run until duration, max_frames, stream EOF, or Ctrl+C
    let result: Result<()> = tokio::select! {
        _ = tokio::signal::ctrl_c() => {
            println!("\n[Client] Benchmark interrupted by user (Ctrl+C).");
            _terminated_early = true;
            Ok(())
        }

        res = async {
            loop {
                // Check duration limit
                if duration_secs > 0.0 && benchmark_start.elapsed().as_secs_f64() >= duration_secs {
                    break;
                }

                // Check frame limit
                if max_frames > 0 && frames_received >= max_frames {
                    break;
                }

                match QuicStreamClient::recv_frame(&mut recv_stream).await {
                    Ok(Some(frame)) => {
                        let arrival = Instant::now();

                        // Measure inter-frame arrival delta
                        if let Some(prev) = last_arrival {
                            let delta_ms = (arrival - prev).as_secs_f64() * 1000.0;
                            jitter_stats.update(delta_ms);
                        }
                        last_arrival = Some(arrival);

                        // Track sequence and detect packet drops
                        if let Some(prev_seq) = last_seq {
                            if frame.sequence_number > prev_seq + 1 {
                                let lost = frame.sequence_number - (prev_seq + 1);
                                frames_dropped += lost;
                            }
                        }
                        last_seq = Some(frame.sequence_number);

                        // Capture stream header metadata
                        if frames_received == 0 {
                            channels = frame.channels;
                            samples_per_frame = frame.samples;
                            sample_rate_hz = frame.sample_rate_hz;
                        }

                        let payload_bytes = frame.total_wire_bytes();
                        total_bytes += payload_bytes;

                        frames_received += 1;
                        total_samples_per_ch += frame.samples as u64;

                        if save_path.is_some() {
                            saved_chunks.push(frame.values());
                        }

                        // Periodic live report every ~1.0 second
                        if last_report_time.elapsed() >= Duration::from_secs(1) {
                            let dt = last_report_time.elapsed().as_secs_f64();
                            let df = frames_received - last_report_frames;
                            let db = total_bytes - last_report_bytes;

                            let current_fps = df as f64 / dt;
                            let current_mb_s = (db as f64 / (1024.0 * 1024.0)) / dt;
                            let current_msamples_s = (df as f64 * samples_per_frame as f64 * channels as f64 / 1_000_000.0) / dt;

                            println!(
                                "  [{:5.1}s] {:6.1} fps | {:7.2} MB/s | {:6.2} MSamp/s | Inter-frame: {:5.2} ms (std: {:4.2} ms) | Drops: {}",
                                benchmark_start.elapsed().as_secs_f64(),
                                current_fps,
                                current_mb_s,
                                current_msamples_s,
                                jitter_stats.mean,
                                jitter_stats.std_dev(),
                                frames_dropped
                            );

                            last_report_time = Instant::now();
                            last_report_frames = frames_received;
                            last_report_bytes = total_bytes;
                        }
                    }
                    Ok(None) => {
                        // Server closed send stream
                        break;
                    }
                    Err(e) => {
                        eprintln!("[Client] Stream read error: {}", e);
                        break;
                    }
                }
            }
            Ok(())
        } => res,
    };

    result?;

    // Send ACK to server
    let _ = client_send.write_all(b"OK").await;
    let _ = client_send.finish();

    let total_elapsed = benchmark_start.elapsed().as_secs_f64();
    if frames_received == 0 || total_elapsed == 0.0 {
        println!("\n[Client] No frames received.");
        return Ok(());
    }

    // 4. Compute comprehensive benchmark statistics
    let total_mb = total_bytes as f64 / (1024.0 * 1024.0);
    let throughput_mb_s = total_mb / total_elapsed;
    let throughput_gbps = (total_bytes as f64 * 8.0) / (total_elapsed * 1_000_000_000.0);
    let avg_fps = frames_received as f64 / total_elapsed;
    let total_channel_samples = total_samples_per_ch * channels as u64;
    let msamples_s = (total_channel_samples as f64 / 1_000_000.0) / total_elapsed;

    let simulated_duration = total_samples_per_ch as f64 / sample_rate_hz;
    let speedup_factor = if total_elapsed > 0.0 {
        simulated_duration / total_elapsed
    } else {
        1.0
    };

    let total_expected_frames = frames_received + frames_dropped;
    let drop_rate_pct = if total_expected_frames > 0 {
        (frames_dropped as f64 / total_expected_frames as f64) * 100.0
    } else {
        0.0
    };

    let nominal_chunk_ms = if sample_rate_hz > 0.0 {
        (samples_per_frame as f64 / sample_rate_hz) * 1000.0
    } else {
        0.0
    };

    let p95 = jitter_stats.percentile(95.0);
    let p99 = jitter_stats.percentile(99.0);

    // 5. Print benchmark report
    println!("\n============================================================");
    println!("             QUIC TRANSMISSION BENCHMARK REPORT             ");
    println!("============================================================");
    println!("  [Endpoint]");
    println!("    Server Address:          {}", server_addr);
    println!("    Handshake Latency:       {:.2} ms", handshake_duration.as_secs_f64() * 1000.0);
    println!("  [Signal Characteristics]");
    println!("    Active Channels:         {}", channels);
    println!("    Sampling Rate:           {:.1} Hz", sample_rate_hz);
    println!("    Chunk Size:              {} samples/frame ({:.2} ms)", samples_per_frame, nominal_chunk_ms);
    println!("    Simulated Duration:      {:.2} seconds", simulated_duration);
    println!("  [Benchmark Performance]");
    println!("    Wall-Clock Time:         {:.3} seconds", total_elapsed);
    println!("    Frames Received:         {} frames", frames_received);
    println!("    Total Data Transferred:  {:.2} MB ({:.4} GB)", total_mb, total_mb / 1024.0);
    println!("    Average Frame Rate:      {:.2} FPS", avg_fps);
    println!("    Network Throughput:      {:.2} MB/s ({:.3} Gbit/s)", throughput_mb_s, throughput_gbps);
    println!("    Signal Throughput:       {:.2} MSamples/s", msamples_s);
    println!("    Real-Time Factor:        {:.2}x Hardware Real-Time", speedup_factor);
    println!("  [Latency & Jitter (Inter-Frame Arrival)]");
    println!("    Min Interval:            {:.3} ms", if jitter_stats.min.is_finite() { jitter_stats.min } else { 0.0 });
    println!("    Mean Interval:           {:.3} ms", jitter_stats.mean);
    println!("    Max Interval:            {:.3} ms", if jitter_stats.max.is_finite() { jitter_stats.max } else { 0.0 });
    println!("    Jitter (Std Dev):        {:.3} ms", jitter_stats.std_dev());
    println!("    P95 Interval:            {:.3} ms", p95);
    println!("    P99 Interval:            {:.3} ms", p99);
    println!("  [Packet Loss & Integrity]");
    println!("    Expected Frames:         {}", total_expected_frames);
    println!("    Received Frames:         {}", frames_received);
    println!("    Dropped Frames:          {}", frames_dropped);
    println!("    Packet Drop Rate:        {:.4}%", drop_rate_pct);
    println!("============================================================\n");

    // 6. Optionally save received binary data
    if let Some(save_path) = save_path {
        println!("[Saving Recording]");
        println!("  Writing binary data to: {}", save_path.display());

        let total_samples = total_samples_per_ch as usize;
        let mut continuous_data = vec![0.0f32; channels as usize * total_samples];

        // Reconstruct channel-major [channels, total_samples]
        let mut sample_offset = 0usize;
        for chunk in &saved_chunks {
            let chunk_samples = samples_per_frame as usize;
            for ch in 0..channels as usize {
                let src_start = ch * chunk_samples;
                let src_end = src_start + chunk_samples;
                let dst_start = ch * total_samples + sample_offset;
                continuous_data[dst_start..dst_start + chunk_samples]
                    .copy_from_slice(&chunk[src_start..src_end]);
            }
            sample_offset += chunk_samples;
        }

        let mut out_file = File::create(&save_path)?;
        let byte_slice = unsafe {
            std::slice::from_raw_parts(
                continuous_data.as_ptr() as *const u8,
                continuous_data.len() * std::mem::size_of::<f32>(),
            )
        };
        out_file.write_all(byte_slice)?;
        out_file.flush()?;

        let meta_path = save_path.with_extension("meta");
        let meta_json = serde_json::json!({
            "channels": channels,
            "samples": total_samples,
            "sample_rate_hz": sample_rate_hz,
            "duration_seconds": simulated_duration,
            "format": "float32-le",
            "frames_received": frames_received,
            "packet_drop_rate": drop_rate_pct,
        });
        std::fs::write(&meta_path, serde_json::to_string_pretty(&meta_json)?)?;
        println!("  Saved successfully ({} MB, meta: {})", total_mb as usize, meta_path.display());
    }

    Ok(())
}
