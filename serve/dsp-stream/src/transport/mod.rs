//! QUIC endpoints with TLS 1.3. Each client gets one connection holding a control stream, a
//! signal stream and one stream per view request (see `proto/dsp_stream.proto`), so a view
//! never waits behind signal data.

mod client;
mod server;
pub mod tls;

pub use client::{Session, SignalReceiver};
pub use server::{serve_recording, Pacing, ServeOptions, Server, DEFAULT_FRAME_SEC, DEFAULT_QUEUE_FRAMES};
pub use tls::{client_config, server_config, ServerIdentity, ServerTls, ServerTrust};

#[cfg(test)]
mod tests {
    use std::net::SocketAddr;
    use std::sync::Arc;

    use dsp_core::{MemoryRecording, RecordingSource};
    use dsp_view::{Envelope, Pyramid, View};

    use super::*;
    use crate::error::StreamError;

    const LOCALHOST: &str = "localhost";

    fn recording() -> Arc<MemoryRecording> {
        let (channels, samples) = (4usize, 5_000usize);
        let data: Vec<f32> = (0..channels * samples).map(|i| ((i * 7919) % 1013) as f32 - 500.0).collect();
        Arc::new(MemoryRecording::new("rec", data, channels, 1000.0).unwrap())
    }

    /// A server on an ephemeral port serving `source` to every client, and a trust for it.
    fn start(source: Arc<dyn RecordingSource>, pyramid: Option<Arc<Pyramid>>, options: ServeOptions) -> (SocketAddr, quinn::ClientConfig) {
        let tls = server_config(&ServerIdentity::SelfSigned { names: vec![LOCALHOST.into()] }).unwrap();
        let server = Server::bind("127.0.0.1:0".parse().unwrap(), tls.config).unwrap();
        let addr = server.local_addr().unwrap();
        tokio::spawn(async move {
            while let Some(connection) = server.accept().await {
                tokio::spawn(serve_recording(connection, source.clone(), pyramid.clone(), options.clone()));
            }
        });
        (addr, client_config(&ServerTrust::Certificates(vec![tls.certificate])).unwrap())
    }

    #[tokio::test]
    async fn the_signal_arrives_whole_and_exact() {
        let rec = recording();
        let options = ServeOptions { pacing: Pacing::Unpaced, frame_sec: 0.3, ..Default::default() };
        let (addr, config) = start(rec.clone(), None, options);
        let mut session = Session::connect(addr, LOCALHOST, config).await.unwrap();
        assert_eq!(session.info().channel_count(), 4);
        assert_eq!(session.info().sample_rate, rec.info().sample_rate);

        let first = 1_234u64;
        let mut signal = session.subscribe(first).await.unwrap();
        let mut next = first;
        let mut received = vec![Vec::new(); 4];
        while let Some((start, values)) = signal.next_values().await.unwrap() {
            assert_eq!(start, next, "frames in order, none missing");
            let n = values.len() / 4;
            for (c, row) in values.chunks_exact(n).enumerate() {
                received[c].extend_from_slice(row);
            }
            next += n as u64;
        }
        assert_eq!(next, 5_000);
        let mut expected = vec![0.0f32; 4 * (5_000 - first as usize)];
        rec.read(&[0, 1, 2, 3], first..5_000, &mut expected).unwrap();
        assert_eq!(received.concat(), expected);
        session.close().await;
    }

    #[tokio::test]
    async fn views_are_answered_from_the_pyramid_or_raw() {
        let rec = recording();
        let pyramid = Arc::new(Pyramid::in_memory(rec.as_ref(), 16).unwrap());
        assert!(pyramid.fill(rec.as_ref(), 0, 5_000, 1 << 20, || false).unwrap());
        let (addr, config) = start(rec.clone(), Some(pyramid.clone()), ServeOptions::default());
        let session = Session::connect(addr, LOCALHOST, config).await.unwrap();

        let wide = View { channels: vec![3, 0], start: 0, end: 5_000, width: 40 };
        assert_eq!(session.view(&wide).await.unwrap(), wide.read(rec.as_ref(), Some(&pyramid)).unwrap());
        let short = View { channels: vec![1], start: 10, end: 30, width: 40 };
        assert!(matches!(session.view(&short).await.unwrap(), Envelope::Samples(s) if s.len() == 20));
        let bad = View { channels: vec![9], start: 0, end: 10, width: 4 };
        assert!(matches!(session.view(&bad).await, Err(StreamError::Protocol(_))));
        session.close().await;
    }

    #[tokio::test]
    async fn untrusted_servers_are_refused_unless_told_otherwise() {
        let (addr, _) = start(recording(), None, ServeOptions::default());
        let other = server_config(&ServerIdentity::SelfSigned { names: vec![LOCALHOST.into()] }).unwrap();
        let wrong = client_config(&ServerTrust::Certificates(vec![other.certificate])).unwrap();
        assert!(Session::connect(addr, LOCALHOST, wrong).await.is_err());
        let any = client_config(&ServerTrust::DangerAcceptAnyCertificate).unwrap();
        Session::connect(addr, LOCALHOST, any).await.unwrap().close().await;
    }
}
