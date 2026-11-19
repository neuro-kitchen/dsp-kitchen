//! A device with a generator and its clients, in one process, over QUIC + TLS as on a network:
//! a refused chain, a configured one, control, a recording, taps, a range read that matches the
//! generator, the stop, what is left on disk; a computer without a certificate gets nothing.

use std::time::Duration;

use dspk_device::{serve, DeviceConfig, LiveDevice};
use dspk_io::live::{Generator, GeneratorParams, SessionLog, RAW_ZARR};
use dspk_stream::live::quic::DeviceClient;
use dspk_stream::live::wire;
use dspk_stream::transport::{live_client_config, Authority, ClientIdentity, Permission, ServerTrust};

const NAME: &str = "localhost";
const WAIT: Duration = Duration::from_secs(20);

fn params() -> GeneratorParams {
    GeneratorParams { channels: 4, sample_rate_hz: 1000.0, block_ms: 5.0, seed: 7, ..Default::default() }
}

async fn client(addr: std::net::SocketAddr, identity: &ClientIdentity) -> DeviceClient {
    let config = live_client_config(&ServerTrust::from_identity(identity), Some(identity)).unwrap();
    DeviceClient::connect(addr, NAME, config).await.unwrap()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_recording_from_configure_to_disk() {
    let data = std::env::temp_dir().join(format!("dspk-device-e2e-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&data);
    let config = DeviceConfig::generator(data.clone(), "127.0.0.1:0".parse().unwrap(), params());
    let device = LiveDevice::open(config).unwrap();
    let (tx, rx) = tokio::sync::oneshot::channel();
    tokio::spawn(serve(device, false, move |l| {
        let _ = tx.send(l.addr);
    }));
    let addr = rx.await.unwrap();

    // Enrolled as pairing would: a director and a viewer
    let authority = Authority::open_or_create(&data.join("authority")).unwrap();
    let director_id = ClientIdentity::from_pem(authority.enroll("director", Permission::Control).unwrap().pem.as_bytes()).unwrap();
    let viewer_id = ClientIdentity::from_pem(authority.enroll("viewer", Permission::Read).unwrap().pem.as_bytes()).unwrap();

    // No certificate: refused
    let anyone = live_client_config(&ServerTrust::DangerAcceptAnyCertificate, None).unwrap();
    assert!(tokio::time::timeout(WAIT, DeviceClient::connect(addr, NAME, anyone)).await.map(|r| r.is_err()).unwrap_or(true), "no certificate, no connection");

    let director = client(addr, &director_id).await;
    let viewer = client(addr, &viewer_id).await;
    assert_eq!(director.welcome().device.as_ref().unwrap().sources[0].name, "gen");

    // A chain that needs the future: every such node refused, with its reason
    let refused = r#"{"dspk_pipeline": 1, "nodes": {"in": {"op": "read"}, "bp": {"op": "iir", "band": "bandpass", "cutoff": [5, 100]}}, "edges": [["in", "bp"]]}"#;
    let r = director.configure(refused, true).await.unwrap();
    assert_eq!(r.refusals.len(), 1);
    assert!(r.refusals[0].reason.contains("forward"));

    // Not the director yet
    let chain = r#"{"dspk_pipeline": 1, "live": {"block_ms": 10},
        "nodes": {"in": {"op": "read"}, "hp": {"op": "iir", "band": "highpass", "cutoff": [5], "mode": "forward", "tap": "hp"}},
        "edges": [["in", "hp"]]}"#;
    let code = director.request_coded(wire::live_request::Kind::Configure(wire::ConfigureChain { spec_json: chain.into(), dry_run: false })).await.unwrap().unwrap_err().code;
    assert_eq!(code, wire::ErrorCode::NotDirector);
    let status = director.request_control().await.unwrap();
    assert_eq!(status.director, "director");
    assert!(viewer.request_control().await.is_err(), "a viewer may not direct");
    let configured = director.configure(chain, false).await.unwrap();
    assert!(configured.refusals.is_empty(), "{:?}", configured.refusals);
    let taps: Vec<String> = director.list_taps().await.unwrap().into_iter().map(|t| t.name).collect();
    assert_eq!(taps, vec!["gen", "hp"]);

    // Record, watch from the viewer
    let mut raw = viewer.subscribe("gen", &[]).await.unwrap();
    let mut hp = viewer.subscribe("hp", &[0]).await.unwrap();
    let mut view = viewer
        .subscribe_view(wire::SubscribeView { tap: "gen".into(), plot_key: 1, channels: vec![], columns_per_second: 100.0, processing: None })
        .await
        .unwrap();
    let status = director.start(true, "e2e", "test").await.unwrap();
    assert_eq!(status.state, wire::DeviceState::Running as i32);

    let first = tokio::time::timeout(WAIT, raw.next()).await.unwrap().unwrap().unwrap();
    assert_eq!(first.first_sample, 0);
    assert_eq!(first.data.len(), 4 * first.count as usize * 2, "int16, every channel");
    let filtered = tokio::time::timeout(WAIT, hp.next()).await.unwrap().unwrap().unwrap();
    assert_eq!(filtered.data.len(), filtered.count as usize * 4, "float32, one channel");
    let columns = tokio::time::timeout(WAIT, view.next()).await.unwrap().unwrap().unwrap();
    assert_eq!(columns.columns.unwrap().min_max.len() % (2 * 4), 0);

    // Let it run, mark it, read back the first 200 ms: exactly the generator's
    tokio::time::sleep(Duration::from_millis(400)).await;
    viewer.mark(0, "halfway").await.unwrap();
    let back = director.read_range("gen", 0, 200, &[]).await.unwrap();
    let expected = Generator::new("gen", &params()).unwrap().block_at(0, 200).unwrap();
    assert_eq!(back.data, expected.data);

    let status = director.stop().await.unwrap();
    assert_eq!(status.state, wire::DeviceState::Stopped as i32);
    let run = status.run.unwrap();
    assert!(run.head_sample >= 300, "{run:?}");

    // On disk: the raw data, the session log, cleanly ended
    let folder = data.join(&run.path);
    assert!(folder.file_name().unwrap().to_string_lossy().ends_with("_e2e"));
    let rec = dspk_io::ZarrRecording::open(&folder.join(RAW_ZARR)).unwrap();
    use dspk_core::RecordingSource;
    assert_eq!(rec.info().samples, run.head_sample);
    let (entries, cut) = SessionLog::read(&folder).unwrap();
    assert!(!cut);
    let kinds: Vec<&str> = entries.iter().map(|e| e.kind.as_str()).collect();
    assert_eq!(kinds.first(), Some(&"start"));
    assert_eq!(kinds.last(), Some(&"stop"));
    assert!(kinds.contains(&"mark"));

    // The director leaves and comes back: still the director
    director.close();
    drop(director);
    tokio::time::sleep(Duration::from_millis(300)).await;
    assert!(viewer.status().await.unwrap().director_kept_until_unix_ns > 0);
    let again = client(addr, &director_id).await;
    tokio::time::sleep(Duration::from_millis(200)).await;
    let s = again.status().await.unwrap();
    assert_eq!((s.director.as_str(), s.director_kept_until_unix_ns), ("director", 0));

    let _ = std::fs::remove_dir_all(&data);
}
