//! The device's second round over QUIC + TLS, in one process: two sources, viewer ops on a live
//! view, a parameter changed while running, a derived tap read back, events in their own file, a
//! pairing code for a viewer, a superseded view dropped, the finished recording served by the
//! session protocol, and a clean stop when the device shuts down.

use std::time::Duration;

use dspk_core::SampleFormat;
use dspk_device::config::{RecordingConfig, RingConfig, SourceConfig};
use dspk_device::{serve, DeviceConfig, LiveDevice};
use dspk_io::live::{GeneratorParams, SessionLog};
use dspk_stream::live::quic::DeviceClient;
use dspk_stream::live::wire;
use dspk_stream::protocol::wire as session;
use dspk_stream::transport::{live_client_config, Authority, ClientIdentity, Permission, ServerTrust};

const NAME: &str = "localhost";
const WAIT: Duration = Duration::from_secs(20);

fn config(data: std::path::PathBuf) -> DeviceConfig {
    let source = |channels, rate: f64, format| GeneratorParams { channels, sample_rate_hz: rate, block_ms: 5.0, format, seed: 3, ..Default::default() };
    DeviceConfig {
        name: "two-sources".into(),
        data,
        listen: "127.0.0.1:0".parse().unwrap(),
        names: vec![NAME.into()],
        runtime: None,
        ring: RingConfig::default(),
        recording: RecordingConfig { chunk_seconds: 0.2, ..Default::default() },
        sources: vec![
            SourceConfig::Generator { name: "ephys".into(), params: source(4, 2000.0, SampleFormat::I16) },
            SourceConfig::Generator { name: "ppg".into(), params: source(2, 400.0, SampleFormat::F16) },
        ],
        web: Some(dspk_device::config::WebConfig { listen: "127.0.0.1:0".parse().unwrap(), root: None }),
    }
}

async fn client(addr: std::net::SocketAddr, identity: &ClientIdentity) -> DeviceClient {
    let config = live_client_config(&ServerTrust::from_identity(identity), Some(identity)).unwrap();
    DeviceClient::connect(addr, NAME, config).await.unwrap()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_second_round() {
    let data = std::env::temp_dir().join(format!("dspk-device-features-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&data);
    let device = LiveDevice::open(config(data.clone())).unwrap();
    let (tx, rx) = tokio::sync::oneshot::channel();
    let served = device.clone();
    tokio::spawn(serve(served, false, move |l| {
        let _ = tx.send((l.addr, l.web_addr.unwrap()));
    }));
    let (addr, web_addr) = rx.await.unwrap();
    let authority = Authority::open_or_create(&data.join("authority")).unwrap();
    let id = ClientIdentity::from_pem(authority.enroll("director", Permission::Control).unwrap().pem.as_bytes()).unwrap();
    let d = client(addr, &id).await;
    assert_eq!(d.welcome().device.as_ref().unwrap().sources.len(), 2);
    d.request_control().await.unwrap();

    // A chain on the second source: a stencil (median), a tap recorded, events recorded
    let chain = r#"{"dspk_pipeline": 1, "live": {"source": "ephys", "block_ms": 5},
        "nodes": {"in": {"op": "read"},
                  "hp": {"op": "iir", "band": "highpass", "cutoff": [20], "mode": "forward"},
                  "med": {"op": "median", "width": 3, "tap": {"name": "smooth", "record": true}},
                  "det": {"op": "crossings", "level": -30, "direction": "falling", "tap": {"name": "spikes", "record": true}}},
        "edges": [["in", "hp"], ["hp", "med"], ["med", "det"]]}"#;
    let r = d.configure(chain, false).await.unwrap();
    assert!(r.refusals.is_empty(), "{:?}", r.refusals);
    assert_eq!(r.chain.as_ref().unwrap().delay_samples, 1);

    // A live view through the viewer's ops (rectify the filtered signal)
    let ops = vec![r#"{"op": "rectify"}"#.to_string()];
    let mut view = d
        .subscribe_view(wire::SubscribeView { tap: "smooth".into(), plot_key: 9, channels: vec![], columns_per_second: 50.0, processing: Some(session::Processing { channels: vec![0, 1], ops }) })
        .await
        .unwrap();
    let mut ppg = d.subscribe("ppg", &[]).await.unwrap();
    d.start(true, "round2", "").await.unwrap();

    let frame = tokio::time::timeout(WAIT, view.next()).await.unwrap().unwrap().unwrap();
    let cols = frame.columns.unwrap().min_max;
    assert_eq!(cols.len() % 4, 0, "two rows of [min, max]");
    assert!(cols.iter().all(|&v| v >= 0.0), "rectified");
    let p = tokio::time::timeout(WAIT, ppg.next()).await.unwrap().unwrap().unwrap();
    assert_eq!(p.data.len(), 2 * p.count as usize * 2, "float16, two channels");

    // A parameter while running: the taps stay, the subscription goes on
    tokio::time::sleep(Duration::from_millis(300)).await;
    d.set_parameter("det", "level", "-40").await.unwrap();
    assert!(d.set_parameter("det", "op", "\"car\"").await.is_err(), "not the op itself");
    tokio::time::timeout(WAIT, view.next()).await.unwrap().unwrap().unwrap();

    // The derived tap read back (from its ring), float32
    tokio::time::sleep(Duration::from_millis(300)).await;
    let back = d.read_range("smooth", 10, 100, &[1]).await.unwrap();
    assert_eq!(back.data.len(), 100 * 4);

    // Views of the past: the newest per plot wins
    let past = session::ViewRequest { channels: vec![0], start: 0, end: 400, width: 40, processing: None };
    let (a, b) = tokio::join!(d.view("ephys", 5, past.clone()), d.view("ephys", 5, past.clone()));
    assert!(a.is_ok() || b.is_ok(), "one of them is answered");

    // A pairing code for a viewer
    let code = d.pairing_code(false).await.unwrap();
    assert!(code.code.contains('-'));

    // A page: the live protocol over a WebSocket, with a token (as `dspk serve --web` pages)
    {
        use futures_util::{SinkExt, StreamExt};
        use prost::Message as _;
        use tokio_tungstenite::tungstenite::Message;
        let (_, token) = authority.enroll_token("page", Permission::Read).unwrap();
        let tcp = tokio::net::TcpStream::connect(web_addr).await.unwrap();
        let (mut ws, _) = tokio_tungstenite::client_async(format!("ws://{web_addr}/live"), tcp).await.unwrap();
        ws.send(Message::text(serde_json::json!({"hello": "token", "token": token}).to_string())).await.unwrap();
        let first = tokio::time::timeout(WAIT, ws.next()).await.unwrap().unwrap().unwrap().into_data();
        assert_eq!(first[0], 1, "the welcome is an event frame");
        let welcome = wire::DeviceWelcome::decode(&first[5..]).unwrap();
        assert_eq!(welcome.client_name, "page");
        let mut request = vec![0u8];
        request.extend_from_slice(&7u32.to_be_bytes());
        wire::LiveRequest { kind: Some(wire::live_request::Kind::Status(wire::GetStatus {})) }.encode(&mut request).unwrap();
        ws.send(Message::binary(request)).await.unwrap();
        loop {
            let bytes = tokio::time::timeout(WAIT, ws.next()).await.unwrap().unwrap().unwrap().into_data();
            if bytes[0] == 0 {
                assert_eq!(u32::from_be_bytes(bytes[1..5].try_into().unwrap()), 7);
                let answer = wire::LiveResponse::decode(&bytes[5..]).unwrap();
                assert!(matches!(answer.kind, Some(wire::live_response::Kind::Status(_))));
                break;
            }
        }
    }

    // The device shuts down (a signal): the recording ends cleanly
    let status = d.status().await.unwrap();
    let path = data.join(status.run.unwrap().path);
    d.finish().await;
    tokio::task::spawn_blocking(move || device.shutdown("test")).await.unwrap();
    assert!(SessionLog::ended_cleanly(&path));
    for f in ["raw-ephys.zarr", "raw-ppg.zarr", "smooth.zarr", "spikes.events.jsonl"] {
        assert!(path.join(f).exists(), "{f}");
    }
    let (entries, _) = SessionLog::read(&path).unwrap();
    assert!(entries.iter().any(|e| e.kind == "parameter"));
    assert!(!entries.iter().any(|e| e.kind == "event"), "events are in their own file");

    // The finished recording, through the session protocol on the same port
    let config = dspk_stream::client_config_with(&ServerTrust::from_identity(&id), Some(&id)).unwrap();
    let mut s = dspk_stream::Session::connect(addr, NAME, config).await.unwrap();
    let folder = path.file_name().unwrap().to_string_lossy().to_string();
    let list = s.list_files(&folder).await.unwrap();
    assert!(list.entries.iter().any(|e| e.name == "raw-ephys.zarr"), "{:?}", list.entries.iter().map(|e| &e.name).collect::<Vec<_>>());
    s.open(&format!("{folder}/raw-ephys.zarr"), None).await.unwrap();
    assert!(s.recording().unwrap().samples > 0);

    let _ = std::fs::remove_dir_all(&data);
}
