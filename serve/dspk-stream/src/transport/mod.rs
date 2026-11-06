//! QUIC endpoints with TLS 1.3. Each client gets one connection holding a control stream, a
//! signal stream and one stream per view request (see `proto/dspk_stream.proto`), so a view
//! never waits behind signal data.

pub mod auth;
mod client;
pub mod files;
pub mod jobs;
mod server;
pub mod tls;

pub use auth::{Authority, ClientIdentity, ClientRecord, Enrollment, Peer, Permission};

pub use client::{Session, SignalReceiver};
pub use server::{
    checked_view, serve_recording, serve_session, Device, OpenRecording, Pacing, RecordingOpener, ServeOptions, Server, REFUSED_CLIENT, DEFAULT_FRAME_SEC, DEFAULT_MAX_CONNECTIONS, DEFAULT_MAX_REQUEST_BYTES, DEFAULT_QUEUE_FRAMES,
    MAX_VIEW_COLUMNS, MAX_VIEW_VALUES,
};
pub use tls::{client_config, client_config_with, server_config, server_config_with_clients, ServerIdentity, ServerTls, ServerTrust};

#[cfg(test)]
mod tests {
    use std::net::SocketAddr;
    use std::sync::Arc;

    use dspk_core::{MemoryRecording, RecordingSource};
    use dspk_view::{Envelope, Pyramid, View};

    use super::*;
    use crate::error::StreamError;
    use crate::protocol::wire;

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
            while let Some((connection, _peer)) = server.accept().await {
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
        assert_eq!(session.recording().unwrap().channel_count(), 4);
        assert_eq!(session.recording().unwrap().sample_rate, rec.info().sample_rate);

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
        assert!(matches!(session.view(&bad).await, Err(StreamError::Server(_))));
        session.close().await;
    }

    /// Requests that would make the server allocate without bound are refused, over the wire
    /// and before any read: repeated channels, too many columns, too many values.
    #[tokio::test]
    async fn oversized_views_are_refused() {
        let (addr, config) = start(recording(), None, ServeOptions::default());
        let session = Session::connect(addr, LOCALHOST, config).await.unwrap();
        let twice = View { channels: vec![1, 1], start: 0, end: 100, width: 10 };
        assert!(matches!(session.view(&twice).await, Err(StreamError::Server(m)) if m.contains("twice")));
        let wide = View { channels: vec![0], start: 0, end: 5_000, width: MAX_VIEW_COLUMNS as usize + 1 };
        assert!(matches!(session.view(&wide).await, Err(StreamError::Server(m)) if m.contains("columns")));
        // The connection still answers after refusals
        let ok = View { channels: vec![0, 1], start: 0, end: 100, width: 10 };
        assert!(session.view(&ok).await.is_ok());
        session.close().await;

        let info = recording().info().clone();
        let mut big = info.clone();
        big.samples = u64::MAX / 2;
        let request = wire::ViewRequest { channels: vec![0, 1, 2, 3], start: 0, end: big.samples, width: MAX_VIEW_COLUMNS };
        assert!(checked_view(&request, &big).is_ok(), "4 channels × 65536 columns is within the limit");
        let mut many = info;
        many.channels = (0..1024).map(|_| many.channels[0].clone()).collect();
        many.samples = u64::MAX / 2;
        let request = wire::ViewRequest { channels: (0..1024).collect(), start: 0, end: many.samples, width: MAX_VIEW_COLUMNS };
        assert!(checked_view(&request, &many).is_err(), "1024 × 65536 values is over the limit");
    }

    /// A server requiring client certificates from its authority, on an ephemeral port.
    fn start_with_clients(authority: Arc<Authority>, anonymous: bool) -> SocketAddr {
        let tls = server_config_with_clients(&authority, &ServerIdentity::SelfSigned { names: vec![LOCALHOST.into()] }, anonymous).unwrap();
        let server = Server::bind("127.0.0.1:0".parse().unwrap(), tls.config).unwrap().with_authority(authority);
        let addr = server.local_addr().unwrap();
        let source: Arc<dyn RecordingSource> = recording();
        tokio::spawn(async move {
            while let Some((connection, _peer)) = server.accept().await {
                tokio::spawn(serve_recording(connection, source.clone(), None, ServeOptions::default()));
            }
        });
        addr
    }

    /// Mutual TLS: an enrolled client connects; one without a certificate, one enrolled by
    /// another authority and a revoked one are refused; anonymous only when allowed.
    #[tokio::test]
    async fn only_enrolled_clients_connect() {
        let dir = std::env::temp_dir().join(format!("dspk-stream-mtls-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let authority = Arc::new(Authority::open_or_create(&dir.join("server")).unwrap());
        let addr = start_with_clients(authority.clone(), false);
        let alice = ClientIdentity::from_pem(authority.enroll("alice", Permission::Read).unwrap().pem.as_bytes()).unwrap();
        let trust = ServerTrust::from_identity(&alice);

        let session = Session::connect(addr, LOCALHOST, client_config_with(&trust, Some(&alice)).unwrap()).await.unwrap();
        assert_eq!(session.recording().unwrap().channel_count(), 4);
        session.close().await;

        assert!(Session::connect(addr, LOCALHOST, client_config(&trust).unwrap()).await.is_err(), "no certificate");
        let other = Authority::open_or_create(&dir.join("other")).unwrap();
        let mallory = ClientIdentity::from_pem(other.enroll("mallory", Permission::Control).unwrap().pem.as_bytes()).unwrap();
        assert!(Session::connect(addr, LOCALHOST, client_config_with(&trust, Some(&mallory)).unwrap()).await.is_err(), "another authority");
        authority.revoke("alice").unwrap();
        assert!(Session::connect(addr, LOCALHOST, client_config_with(&trust, Some(&alice)).unwrap()).await.is_err(), "revoked");

        // Anonymous access, when the server allows it: read only, no certificate needed
        let open = start_with_clients(authority.clone(), true);
        let session = Session::connect(open, LOCALHOST, client_config(&trust).unwrap()).await.unwrap();
        session.close().await;
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Opens every recording as the test recording (the formats are the caller's).
    struct TestOpener;
    impl RecordingOpener for TestOpener {
        fn open(&self, _path: &std::path::Path, _source: Option<&str>) -> crate::error::StreamResult<OpenRecording> {
            Ok(OpenRecording::new(recording(), None))
        }
    }

    /// A server with a data root and no initial recording: a client lists, opens, views; paths
    /// outside, hidden or not recordings are refused; nothing streams before a recording is open.
    #[tokio::test]
    async fn files_are_listed_and_opened_under_the_data_root() {
        let base = std::env::temp_dir().join(format!("dspk-stream-device-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        let root = base.join("data");
        std::fs::create_dir_all(root.join("mouse1")).unwrap();
        std::fs::create_dir_all(root.join(".dspk-auth")).unwrap();
        std::fs::write(root.join("mouse1/rec.ap.bin"), [0u8; 8]).unwrap();
        std::fs::write(root.join("notes.txt"), "x").unwrap();
        let device = Arc::new(Device {
            name: "test".into(),
            version: "0".into(),
            files: Some((files::DataRoot::new(&root).unwrap(), Arc::new(TestOpener))),
            recording: None,
            jobs: None,
            options: ServeOptions::default(),
        });
        let tls = server_config(&ServerIdentity::SelfSigned { names: vec![LOCALHOST.into()] }).unwrap();
        let server = Server::bind("127.0.0.1:0".parse().unwrap(), tls.config).unwrap();
        let addr = server.local_addr().unwrap();
        tokio::spawn(async move {
            while let Some((connection, peer)) = server.accept().await {
                tokio::spawn(serve_session(connection, peer, device.clone()));
            }
        });
        let config = client_config(&ServerTrust::Certificates(vec![tls.certificate])).unwrap();
        let mut session = Session::connect(addr, LOCALHOST, config).await.unwrap();
        assert!(session.server().has_files && session.recording().is_none());
        assert!(session.view(&View { channels: vec![0], start: 0, end: 10, width: 4 }).await.is_err(), "nothing open yet");

        let top = session.list_files("").await.unwrap();
        let names: Vec<&str> = top.entries.iter().map(|e| e.name.as_str()).collect();
        assert_eq!(names, ["mouse1", "notes.txt"], "hidden entries left out, directories first");
        for bad in ["../data/notes.txt", ".dspk-auth", "/etc/passwd"] {
            assert!(session.list_files(bad).await.is_err(), "{bad}");
        }
        assert!(session.open("notes.txt", None).await.is_err(), "not a recording");
        assert!(session.open(".dspk-auth/ca.key.pem", None).await.is_err(), "hidden");

        assert_eq!(session.open("mouse1/rec.ap.bin", None).await.unwrap().channel_count(), 4);
        let view = View { channels: vec![0, 1], start: 0, end: 100, width: 10 };
        assert!(session.view(&view).await.is_ok());
        let mut signal = session.subscribe(4_990).await.unwrap();
        assert!(signal.next_values().await.unwrap().is_some());
        session.close().await;
        let _ = std::fs::remove_dir_all(&base);
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
