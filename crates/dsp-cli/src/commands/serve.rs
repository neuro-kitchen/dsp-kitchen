use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{bail, Context, Result};
use tokio::time::{interval, MissedTickBehavior};

use dsp_core::{ChunkSchedule, DspError, RecordingSource};
use dsp_io::synthetic::{SyntheticParams, SyntheticRecording};
use dsp_stream::network::{generate_server_config, QuicStreamServer, StreamFrame};
use dsp_stream::purpose::StreamPurpose;
use dsp_stream::PrefetchReader;

/// Chunks read ahead per client (bounds memory to a few frames regardless of recording length).
const CLIENT_QUEUE_CHUNKS: usize = 8;

/// Serve a recording (any format dsp-io opens) or a procedural synthetic signal over QUIC.
/// Data is read chunk by chunk, so recordings larger than memory stream with bounded memory.
#[allow(clippy::too_many_arguments)]
pub async fn run_serve(
    bind_addr: SocketAddr,
    file_path: Option<PathBuf>,
    cli_channels: Option<usize>,
    cli_sample_rate: Option<f64>,
    chunk_size: usize,
    loop_stream: bool,
    realtime: bool,
    cert_out: Option<PathBuf>,
    int16_gain_uv: Option<f32>,
) -> Result<()> {
    println!("============================================================");
    println!("         DSP-KITCHEN QUIC SIGNAL STREAMING SERVER           ");
    println!("============================================================");
    if chunk_size == 0 {
        bail!("chunk size must be at least 1 sample");
    }

    let source: Arc<dyn RecordingSource> = match &file_path {
        Some(path) => {
            println!("  [Mode]        Serving File: {}", path.display());
            if cli_channels.is_some() || cli_sample_rate.is_some() {
                println!("  [Note]        --channels / --sample-rate apply to the synthetic source only; file layout comes from dsp-io");
            }
            Arc::from(dsp_io::open(path).with_context(|| format!("Failed to open {}", path.display()))?)
        }
        None => {
            println!("  [Mode]        Serving Procedural Synthetic Signal (1 h, generated on the fly)");
            let params = SyntheticParams {
                channels: cli_channels.unwrap_or(384),
                sample_rate_hz: cli_sample_rate.unwrap_or(30_000.0),
                duration_sec: 3600.0,
                ..Default::default()
            };
            Arc::new(SyntheticRecording::new(params)?)
        }
    };
    let info = source.info();
    let (channels, total_samples, sample_rate) = (info.channel_count(), info.samples as usize, info.sample_rate_hz());
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

                let source = Arc::clone(&source);
                tokio::spawn(async move {
                    if let Err(e) = handle_client_stream(
                        client_id,
                        conn,
                        source,
                        chunk_size,
                        loop_stream,
                        realtime,
                        int16_gain_uv,
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
    source: Arc<dyn RecordingSource>,
    chunk_size: usize,
    loop_stream: bool,
    realtime: bool,
    int16_gain_uv: Option<f32>,
) -> Result<()> {
    let (mut send_stream, mut recv_stream) = conn
        .open_bi()
        .await
        .context("Failed to open bidirectional QUIC stream")?;

    let info = source.info().clone();
    let (channels, total, sample_rate) = (info.channel_count(), info.samples, info.sample_rate_hz());

    // Producer thread: sequential prefetching reads into a bounded queue.
    let (tx, mut rx) = tokio::sync::mpsc::channel::<(u64, Vec<f32>)>(CLIENT_QUEUE_CHUNKS);
    let producer = std::thread::spawn(move || {
        loop {
            let schedule = ChunkSchedule::full_recording(total, chunk_size as u64, 0, 0);
            let reader = PrefetchReader::new(source.as_ref(), schedule);
            let result = reader.for_each_window(|win, data| {
                tx.blocking_send((win.valid_global.start, data.to_vec()))
                    .map_err(|_| DspError::Io("client disconnected".into()))
            });
            if result.is_err() || !loop_stream {
                break;
            }
        }
    });

    let chunk_interval = Duration::from_secs_f64(chunk_size as f64 / sample_rate);
    let mut ticker = interval(chunk_interval);
    ticker.set_missed_tick_behavior(MissedTickBehavior::Skip);

    let mut seq = 0u64;
    let mut total_bytes_sent = 0usize;
    let t0 = std::time::Instant::now();

    while let Some((start, data)) = rx.recv().await {
        if realtime {
            ticker.tick().await;
        }
        let samples = (data.len() / channels) as u32;
        let frame = match int16_gain_uv {
            Some(gain) => {
                let raw: Vec<i16> = data
                    .iter()
                    .map(|v| (v / gain).round().clamp(i16::MIN as f32, i16::MAX as f32) as i16)
                    .collect();
                StreamFrame::new_i16(seq, start, channels as u32, samples, sample_rate, &raw, gain, StreamPurpose::Processing)
            }
            None => StreamFrame::new(seq, start, channels as u32, samples, sample_rate, data, StreamPurpose::Processing),
        };
        total_bytes_sent += frame.total_wire_bytes();
        if QuicStreamServer::send_frame(&mut send_stream, &frame).await.is_err() {
            break; // client closed or connection lost
        }
        seq += 1;
    }
    drop(rx);
    let _ = producer.join();

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
