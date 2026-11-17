//! A client that loses its connection mid-run: the device runs on and records; the client
//! connects again and reads the whole run back, with no hole where it was away; a view zoomed out
//! over the run is answered from the device's overview (built as blocks arrived).

use std::time::Duration;

use dspk_device::{serve, DeviceConfig, LiveDevice};
use dspk_io::live::{Generator, GeneratorParams};
use dspk_stream::live::quic::DeviceClient;
use dspk_stream::live::wire;
use dspk_stream::protocol::wire as session;
use dspk_stream::transport::{live_client_config, Authority, ClientIdentity, Permission, ServerTrust};

const NAME: &str = "localhost";

fn params() -> GeneratorParams {
    GeneratorParams { channels: 2, sample_rate_hz: 8000.0, block_ms: 5.0, seed: 11, ..Default::default() }
}

async fn client(addr: std::net::SocketAddr, identity: &ClientIdentity) -> DeviceClient {
    let config = live_client_config(&ServerTrust::from_identity(identity), Some(identity)).unwrap();
    DeviceClient::connect(addr, NAME, config).await.unwrap()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_client_that_reconnects_sees_no_hole() {
    let data = std::env::temp_dir().join(format!("dspk-device-reconnect-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&data);
    let config = DeviceConfig::generator(data.clone(), "127.0.0.1:0".parse().unwrap(), params());
    let device = LiveDevice::open(config).unwrap();
    let (tx, rx) = tokio::sync::oneshot::channel();
    tokio::spawn(serve(device, false, move |l| {
        let _ = tx.send(l.addr);
    }));
    let addr = rx.await.unwrap();
    let authority = Authority::open_or_create(&data.join("authority")).unwrap();
    let id = ClientIdentity::from_pem(authority.enroll("director", Permission::Control).unwrap().pem.as_bytes()).unwrap();

    let first = client(addr, &id).await;
    first.request_control().await.unwrap();
    let chain = r#"{"dspk_pipeline": 1, "live": {"block_ms": 5},
        "nodes": {"in": {"op": "read"}, "hp": {"op": "iir", "band": "highpass", "cutoff": [5], "mode": "forward", "tap": "hp"}},
        "edges": [["in", "hp"]]}"#;
    assert!(first.configure(chain, false).await.unwrap().refusals.is_empty());
    first.start(true, "reconnect", "").await.unwrap();
    tokio::time::sleep(Duration::from_millis(300)).await;
    // The connection goes; the device runs on
    first.finish().await;
    tokio::time::sleep(Duration::from_millis(500)).await;

    let again = client(addr, &id).await;
    let status = again.status().await.unwrap();
    assert_eq!(status.state, wire::DeviceState::Running as i32, "a lost connection never stops the device");
    let head = status.run.as_ref().unwrap().head_sample;
    assert!(head >= 4096, "the run went on while the client was away: {head}");
    // Everything from the start, through the time away: what the generator made, sample for sample
    let n = (head as u32).min(1500);
    let back = again.read_range("gen", 0, n, &[]).await.unwrap();
    let expected = Generator::new("gen", &params()).unwrap().block_at(0, n as usize).unwrap();
    assert_eq!(back.data, expected.data, "no hole where the client was away");

    // Zoomed out over the run: columns from the overview, whatever the run's length
    let view = session::ViewRequest { start: 0, end: head / 4096 * 4096, width: 1, channels: vec![0], ..Default::default() };
    let frame = again.view("gen", 1, view).await.unwrap();
    assert!(matches!(frame.content, Some(session::envelope_frame::Content::Columns(_))));
    again.request_control().await.unwrap();
    again.stop().await.unwrap();
    again.finish().await;
    let _ = std::fs::remove_dir_all(&data);
}
