use std::fs::File;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{bail, Context, Result};
use memmap2::Mmap;
use serde::Deserialize;
use tokio::time::{interval, MissedTickBehavior};

use dsp_stream::network::{generate_server_config, QuicStreamServer, StreamFrame};
use dsp_stream::purpose::StreamPurpose;

#[derive(Deserialize, Debug)]
struct SidecarMetadata {
    channels: Option<usize>,
    samples: Option<usize>,
    sample_rate_hz: Option<f64>,
}

/// Serve a dataset or continuous synthetic signal over high-throughput QUIC transport.
pub async fn run_serve(
    bind_addr: SocketAddr,
    file_path: Option<PathBuf>,
    cli_channels: Option<usize>,
    cli_sample_rate: Option<f64>,
    chunk_size: usize,
    loop_stream: bool,
    realtime: bool,
    cert_out: Option<PathBuf>,
) -> Result<()> {
    println!("============================================================");
    println!("         DSP-KITCHEN QUIC SIGNAL STREAMING SERVER           ");
    println!("============================================================");

    // 1. Resolve dataset / synthetic stream parameters
    let (data_buffer, channels, total_samples, sample_rate) = match &file_path {
        Some(path) => {
            if !path.exists() {
                bail!("Specified dataset file does not exist: {}", path.display());
            }

            println!("  [Mode]        Serving File: {}", path.display());

            // Check for sidecar metadata (.meta or .json)
            let mut detected_channels = cli_channels;
            let mut detected_samples = None;
            let mut detected_sr = cli_sample_rate;

            let meta_path = path.with_extension("meta");
            if meta_path.exists() {
                if let Ok(content) = std::fs::read_to_string(&meta_path) {
                    if let Ok(meta) = serde_json::from_str::<SidecarMetadata>(&content) {
                        println!("  [Metadata]    Loaded sidecar: {}", meta_path.display());
                        if detected_channels.is_none() {
                            detected_channels = meta.channels;
                        }
                        if detected_samples.is_none() {
                            detected_samples = meta.samples;
                        }
                        if detected_sr.is_none() {
                            detected_sr = meta.sample_rate_hz;
                        }
                    }
                }
            }

            let channels = detected_channels.unwrap_or(384);
            let sample_rate = detected_sr.unwrap_or(30000.0);

            // Memory-map the file
            let file = File::open(path)
                .with_context(|| format!("Failed to open dataset file: {}", path.display()))?;
            let mmap = unsafe { Mmap::map(&file)? };

            let total_floats = mmap.len() / std::mem::size_of::<f32>();
            if total_floats == 0 {
                bail!("Dataset file is empty: {}", path.display());
            }

            let computed_samples = detected_samples.unwrap_or(total_floats / channels);
            if channels * computed_samples > total_floats {
                bail!(
                    "File size ({} floats) is smaller than channels ({}) * samples ({})",
                    total_floats,
                    channels,
                    computed_samples
                );
            }

            // Copy mapped floats into an Arc buffer for zero-overhead multi-client sharing
            let float_slice = unsafe {
                std::slice::from_raw_parts(mmap.as_ptr() as *const f32, channels * computed_samples)
            };
            let buffer = Arc::new(float_slice.to_vec());

            (buffer, channels, computed_samples, sample_rate)
        }
        None => {
            println!("  [Mode]        Serving Live Synthetic Multi-Channel Signal");
            let channels = cli_channels.unwrap_or(384);
            let sample_rate = cli_sample_rate.unwrap_or(30000.0);
            let samples = (sample_rate * 2.0) as usize; // 2 seconds continuous buffer

            println!("  [Synthesizer] Pre-generating 2.0s multi-channel test pattern...");
            let mut buffer = vec![0.0f32; channels * samples];

            // Generate deterministic 60Hz hum + noise + spikes pattern
            let dt = 1.0 / sample_rate;
            let omega_60 = 2.0 * std::f64::consts::PI * 60.0;
            for ch in 0..channels {
                let ch_offset = ch * samples;
                let phase = (ch as f64 * 0.1).fract() * 2.0 * std::f64::consts::PI;
                for s in 0..samples {
                    let t = s as f64 * dt;
                    let hum = (25.0 * (omega_60 * t + phase).sin()) as f32;
                    let pseudo_noise = ((s * 37 + ch * 101) % 1000) as f32 / 1000.0 * 15.0;
                    buffer[ch_offset + s] = hum + pseudo_noise;
                }
            }

            (Arc::new(buffer), channels, samples, sample_rate)
        }
    };

    let duration_sec = total_samples as f64 / sample_rate;
    let chunk_duration_ms = (chunk_size as f64 / sample_rate) * 1000.0;

    println!("  [Config]      Channels:            {}", channels);
    println!("  [Config]      Buffer Length:       {} samples ({:.2} s)", total_samples, duration_sec);
    println!("  [Config]      Sampling Rate:       {:.1} Hz", sample_rate);
    println!("  [Config]      Chunk Size:          {} samples ({:.2} ms / frame)", chunk_size, chunk_duration_ms);
    println!("  [Config]      Loop Stream:         {}", loop_stream);
    println!("  [Config]      Real-time Pacing:    {}", if realtime { "ENABLED (Clock Paced)" } else { "DISABLED (Line Rate Maximum Stress)" });

    // 2. Generate TLS certificates and initialize QUIC server
    let (server_config, cert_der) = generate_server_config(vec![
        "localhost".to_string(),
        bind_addr.ip().to_string(),
        "127.0.0.1".to_string(),
    ])
    .map_err(|e| anyhow::anyhow!("Failed to generate QUIC TLS configuration: {e}"))?;


    if let Some(ref cert_path) = cert_out {
        std::fs::write(cert_path, &cert_der)
            .with_context(|| format!("Failed to write certificate to {}", cert_path.display()))?;
        println!("  [TLS]         Certificate saved to: {}", cert_path.display());
    }

    let server = QuicStreamServer::bind(bind_addr, server_config)
        .with_context(|| format!("Failed to bind QUIC server on {}", bind_addr))?;

    let actual_addr = server.local_addr().unwrap_or(bind_addr);
    println!("  [Network]     QUIC UDP Listening on: {}", actual_addr);
    println!("------------------------------------------------------------");
    println!("Server is ready. In another terminal, connect using:");
    println!("  dsp-cli receive --addr {}", actual_addr);
    println!("Press Ctrl+C to stop.");
    println!("============================================================\n");

    let server = Arc::new(server);

    // Accept loop with graceful shutdown on Ctrl+C
    tokio::select! {
        _ = tokio::signal::ctrl_c() => {
            println!("\n[QUIC Server] Shutting down gracefully...");
        }
        _ = async {
            let mut client_id = 0u64;
            while let Some(conn) = server.accept().await {
                client_id += 1;
                let remote_addr = conn.remote_address();
                println!("[QUIC Server] [Client #{}] Connected from {}", client_id, remote_addr);

                let buffer_clone = Arc::clone(&data_buffer);
                tokio::spawn(async move {
                    if let Err(e) = handle_client_stream(
                        client_id,
                        conn,
                        buffer_clone,
                        channels,
                        total_samples,
                        sample_rate,
                        chunk_size,
                        loop_stream,
                        realtime,
                    ).await {
                        println!("[QUIC Server] [Client #{}] Stream closed: {}", client_id, e);
                    }
                });
            }
        } => {}
    }

    Ok(())
}

async fn handle_client_stream(
    client_id: u64,
    conn: quinn::Connection,
    buffer: Arc<Vec<f32>>,
    channels: usize,
    total_samples: usize,
    sample_rate: f64,
    chunk_size: usize,
    loop_stream: bool,
    realtime: bool,
) -> Result<()> {
    let (mut send_stream, mut recv_stream) = conn
        .open_bi()
        .await
        .context("Failed to open bidirectional QUIC stream")?;

    let chunk_interval = Duration::from_secs_f64(chunk_size as f64 / sample_rate);
    let mut ticker = interval(chunk_interval);
    ticker.set_missed_tick_behavior(MissedTickBehavior::Skip);

    let mut seq = 0u64;
    let mut current_sample = 0usize;
    let mut total_bytes_sent = 0usize;
    let t0 = std::time::Instant::now();

    loop {
        if realtime {
            ticker.tick().await;
        }

        // Check if we need to wrap around or terminate
        if current_sample + chunk_size > total_samples {
            if loop_stream {
                current_sample = 0;
            } else {
                break;
            }
        }

        // Extract chunk in channel-major layout [channels x chunk_size]
        let mut frame_data = Vec::with_capacity(channels * chunk_size);
        for ch in 0..channels {
            let start = ch * total_samples + current_sample;
            let end = start + chunk_size;
            frame_data.extend_from_slice(&buffer[start..end]);
        }

        let frame = StreamFrame::new(
            seq,
            current_sample as u64,
            channels as u32,
            chunk_size as u32,
            sample_rate,
            frame_data,
            StreamPurpose::Processing,
        );

        let encoded_len = frame.total_wire_bytes();
        total_bytes_sent += encoded_len;


        if let Err(_e) = QuicStreamServer::send_frame(&mut send_stream, &frame).await {
            // Client closed or connection lost
            break;
        }


        seq += 1;
        current_sample += chunk_size;
    }

    let _ = send_stream.finish();

    // Await client ACK or stream close
    let mut ack_buf = [0u8; 8];
    let _ = recv_stream.read(&mut ack_buf).await;

    let elapsed = t0.elapsed().as_secs_f64();
    let mb_sent = total_bytes_sent as f64 / (1024.0 * 1024.0);
    let throughput = if elapsed > 0.0 { mb_sent / elapsed } else { 0.0 };

    println!(
        "[QUIC Server] [Client #{}] Finished: Sent {} frames ({:.2} MB in {:.2}s, avg {:.1} MB/s)",
        client_id, seq, mb_sent, elapsed, throughput
    );

    Ok(())
}
