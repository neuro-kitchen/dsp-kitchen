//! QUIC transport protocol for high-throughput multi-channel signal streaming.
//!
//! Provides zero head-of-line blocking, connection migration, and TLS 1.3 encryption
//! for streaming raw continuous waveforms and decimated visualization packets over UDP.

use std::net::SocketAddr;
use tokio::io::AsyncWriteExt;

use super::frame::StreamFrame;
use super::tls::{generate_self_signed_tls, QuicTlsBundle};

/// High-throughput QUIC Streaming Server (Data Producer).
pub struct QuicStreamServer {
    endpoint: quinn::Endpoint,
}

impl QuicStreamServer {
    /// Binds a QUIC server on the specified UDP socket address using the provided server configuration.
    pub fn bind(
        addr: SocketAddr,
        server_config: quinn::ServerConfig,
    ) -> Result<Self, quinn::ConnectionError> {
        let endpoint = quinn::Endpoint::server(server_config, addr)
            .map_err(|_e| quinn::ConnectionError::LocallyClosed)?;
        Ok(Self { endpoint })
    }

    /// Convenience constructor binding on `addr` with an automatically generated in-memory self-signed certificate.
    pub fn bind_self_signed(
        addr: SocketAddr,
        server_names: Vec<String>,
    ) -> Result<(Self, quinn::ClientConfig), Box<dyn std::error::Error + Send + Sync>> {
        let QuicTlsBundle {
            server_config,
            client_config,
        } = generate_self_signed_tls(server_names)?;
        let server = Self::bind(addr, server_config)?;
        Ok((server, client_config))
    }

    /// Returns the local UDP socket address this server is bound to.
    pub fn local_addr(&self) -> Result<SocketAddr, std::io::Error> {
        self.endpoint.local_addr()
    }

    /// Asynchronously accepts the next incoming QUIC client connection.
    pub async fn accept(&self) -> Option<quinn::Connection> {
        let incoming = self.endpoint.accept().await?;
        incoming.await.ok()
    }

    /// Sends a length-delimited `StreamFrame` over a QUIC send stream.
    pub async fn send_frame(
        stream: &mut quinn::SendStream,
        frame: &StreamFrame,
    ) -> Result<(), std::io::Error> {
        let bytes = frame.encode_length_delimited();
        stream.write_all(&bytes).await?;
        stream.flush().await?;
        Ok(())
    }
}

/// High-throughput QUIC Streaming Client (Data Consumer / Visualizer).
pub struct QuicStreamClient {
    endpoint: quinn::Endpoint,
}

impl QuicStreamClient {
    /// Binds a QUIC client endpoint on an ephemeral local UDP port with the given client configuration.
    pub fn bind(client_config: quinn::ClientConfig) -> Result<Self, std::io::Error> {
        let mut endpoint = quinn::Endpoint::client("0.0.0.0:0".parse().unwrap())?;
        endpoint.set_default_client_config(client_config);
        Ok(Self { endpoint })
    }

    /// Connects to a remote QUIC server at `server_addr` with the expected TLS `server_name`.
    pub async fn connect(
        &self,
        server_addr: SocketAddr,
        server_name: &str,
    ) -> Result<quinn::Connection, quinn::ConnectionError> {
        let connecting = self
            .endpoint
            .connect(server_addr, server_name)
            .map_err(|_| quinn::ConnectionError::LocallyClosed)?;
        connecting.await
    }

    /// Receives a length-delimited `StreamFrame` from a QUIC receive stream.
    pub async fn recv_frame(
        stream: &mut quinn::RecvStream,
    ) -> Result<Option<StreamFrame>, Box<dyn std::error::Error + Send + Sync>> {
        let mut len_buf = [0u8; 4];
        match stream.read_exact(&mut len_buf).await {
            Ok(_) => {}
            Err(quinn::ReadExactError::FinishedEarly(_)) => return Ok(None),
            Err(e) => return Err(Box::new(e)),
        }

        let msg_len = u32::from_be_bytes(len_buf) as usize;
        let mut msg_buf = vec![0u8; msg_len];
        match stream.read_exact(&mut msg_buf).await {
            Ok(_) => {}
            Err(quinn::ReadExactError::FinishedEarly(_)) => return Ok(None),
            Err(e) => return Err(Box::new(e)),
        }

        let frame = StreamFrame::decode_from_slice(&msg_buf)?;
        Ok(Some(frame))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::purpose::StreamPurpose;

    #[tokio::test]
    async fn test_quic_streaming_end_to_end() {
        let server_addr: SocketAddr = "127.0.0.1:0".parse().unwrap();
        let (server, client_config) =
            QuicStreamServer::bind_self_signed(server_addr, vec!["localhost".to_string()])
                .expect("Failed to bind server");

        let bound_addr = server.local_addr().expect("Failed to get local addr");

        // Spawn server task
        let num_frames = 20u64;
        let channels = 32u32;
        let samples = 250u32;

        let server_task = tokio::spawn(async move {
            let connection = server.accept().await.expect("Server failed to accept connection");
            let (mut send_stream, mut server_recv) = connection
                .open_bi()
                .await
                .expect("Server failed to open bidirectional stream");

            for seq in 0..num_frames {
                let data = vec![seq as f32; (channels * samples) as usize];
                let frame = StreamFrame::new(
                    seq,
                    seq * (samples as u64),
                    channels,
                    samples,
                    30000.0,
                    data,
                    StreamPurpose::Processing,
                );

                QuicStreamServer::send_frame(&mut send_stream, &frame)
                    .await
                    .expect("Failed to send frame");
            }

            let _ = send_stream.finish();

            // Wait for client acknowledgement before closing connection
            let mut ack = [0u8; 2];
            let _ = server_recv.read_exact(&mut ack).await;
        });

        // Client task
        let client = QuicStreamClient::bind(client_config).expect("Failed to bind client");
        let conn = client
            .connect(bound_addr, "localhost")
            .await
            .expect("Client failed to connect");

        let (mut client_send, mut recv_stream) = conn
            .accept_bi()
            .await
            .expect("Client failed to accept bi stream");

        let mut received_count = 0u64;
        while let Some(frame) = QuicStreamClient::recv_frame(&mut recv_stream)
            .await
            .expect("Failed to recv frame")
        {
            assert_eq!(frame.sequence_number, received_count);
            assert_eq!(frame.channels, channels);
            assert_eq!(frame.samples, samples);
            assert_eq!(frame.data.len(), (channels * samples) as usize);
            assert_eq!(frame.data[0], received_count as f32);
            received_count += 1;
        }

        // Send ACK back to server
        client_send.write_all(b"OK").await.expect("Failed to send ACK");
        let _ = client_send.finish();

        assert_eq!(received_count, num_frames);
        server_task.await.expect("Server task panicked");
    }

    #[tokio::test]
    async fn test_quic_neuropixels_high_throughput_streaming() {
        let server_addr: SocketAddr = "127.0.0.1:0".parse().unwrap();
        let (server, client_config) =
            QuicStreamServer::bind_self_signed(server_addr, vec!["localhost".to_string()])
                .expect("Failed to bind server");

        let bound_addr = server.local_addr().expect("Failed to get local addr");

        // 384 channels x 500 samples (Neuropixels density)
        let num_frames = 50u64;
        let channels = 384u32;
        let samples = 500u32;
        let total_floats = (channels * samples) as usize;
        let total_bytes = num_frames * (total_floats as u64) * 4;

        let server_task = tokio::spawn(async move {
            let connection = server.accept().await.expect("Server failed to accept connection");
            let (mut send_stream, mut server_recv) = connection
                .open_bi()
                .await
                .expect("Server failed to open bidirectional stream");

            let payload = vec![42.0f32; total_floats];

            for seq in 0..num_frames {
                let frame = StreamFrame::new(
                    seq,
                    seq * (samples as u64),
                    channels,
                    samples,
                    30000.0,
                    payload.clone(),
                    StreamPurpose::Processing,
                );

                QuicStreamServer::send_frame(&mut send_stream, &frame)
                    .await
                    .expect("Failed to send frame");
            }

            let _ = send_stream.finish();

            let mut ack = [0u8; 2];
            let _ = server_recv.read_exact(&mut ack).await;
        });

        let client = QuicStreamClient::bind(client_config).expect("Failed to bind client");
        let conn = client
            .connect(bound_addr, "localhost")
            .await
            .expect("Client failed to connect");

        let (mut client_send, mut recv_stream) = conn
            .accept_bi()
            .await
            .expect("Client failed to accept bi stream");

        let t0 = std::time::Instant::now();
        let mut received = 0u64;
        while let Some(frame) = QuicStreamClient::recv_frame(&mut recv_stream)
            .await
            .expect("Failed to recv frame")
        {
            assert_eq!(frame.channels, 384);
            assert_eq!(frame.samples, 500);
            assert_eq!(frame.data.len(), total_floats);
            received += 1;
        }

        let elapsed = t0.elapsed().as_secs_f64();
        let mb_transferred = (total_bytes as f64) / (1024.0 * 1024.0);
        let throughput_mb_s = mb_transferred / elapsed;

        client_send.write_all(b"OK").await.expect("Failed to send ACK");
        let _ = client_send.finish();

        assert_eq!(received, num_frames);
        server_task.await.expect("Server task panicked");

        println!(
            "QUIC Throughput: Transferred {:.2} MB in {:.3} s ({:.1} MB/s)",
            mb_transferred, elapsed, throughput_mb_s
        );
    }
}
