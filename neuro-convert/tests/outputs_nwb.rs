//! NWB output on a small synthetic session. Writes `target/nwb-test/small.nwb.zarr`, which
//! `tests/validate_nwb.py` reads back with pynwb + hdmf-zarr:
//! `uv run --with pynwb --with hdmf-zarr --with nwbinspector tests/validate_nwb.py target/nwb-test/small.nwb.zarr`

use std::path::Path;
use std::sync::Arc;

use neuro_convert::metadata::MetadataFile;
use neuro_convert::model::{Device, EventSeries, MemoryRecording, Session, Table};
use neuro_convert::outputs::nwb::{self, NwbOptions};

fn session() -> Session {
    let mut s = Session::default();
    s.metadata.start_time = Some("2025-02-26T15:25:56".into());
    s.metadata.experiment = Some("synthetic".into());
    s.metadata.subject.id = Some("rat1".into());
    s.metadata.devices = vec![Device { name: "RZ2(1)".into(), description: "RZn Processor".into(), manufacturer: Some("TDT".into()) }];

    // 3-channel "EMG" (value = ch * 1000 + t) and a 1-channel "Temp", 1000 samples each
    let emg: Vec<f32> = (0..3).flat_map(|c| (0..1000).map(move |t| (c * 1000 + t) as f32)).collect();
    s.recordings.push(Arc::new(MemoryRecording::new("EMG1", emg, 3, 1000.0, "V").unwrap()));
    s.recordings.push(Arc::new(MemoryRecording::new("Temp", (0..1000).map(|t| t as f32 * 0.5).collect(), 1, 100.0, "a.u.").unwrap()));
    s.recordings.push(Arc::new(MemoryRecording::new("Skip", vec![0.0; 10], 1, 10.0, "a.u.").unwrap()));

    s.events.push(EventSeries { name: "MET/".into(), onsets: vec![0.5, 1.5], offsets: Some(vec![0.6, 1.6]), values: vec![1.0, 2.0], channels: 1, ..Default::default() });
    s.events.push(EventSeries { name: "Tick".into(), onsets: vec![0.0, 1.0, 2.0], values: vec![0.0, 1.0, 2.0], channels: 1, ..Default::default() });
    s.events.push(EventSeries {
        name: "Note".into(),
        onsets: vec![0.2, 0.9],
        values: vec![1.0, 2.0],
        channels: 1,
        labels: vec!["sleep".into(), "Bottle In".into()],
        ..Default::default()
    });
    s.events.push(EventSeries { name: "eS1p".into(), onsets: vec![0.25], values: vec![0.5, -750.0], channels: 2, ..Default::default() });
    s.tables.push(Table {
        name: "Z_EMG".into(),
        description: "impedance".into(),
        columns: vec!["TIME (S)".into(), "R1 (kOhm)".into(), "note".into()],
        rows: vec![vec!["59".into(), "0.96".into(), "ok".into()], vec!["63".into(), "-1.00".into(), "n/a".into()]],
    });
    s
}

const META: &str = r#"
session:
  description: "Synthetic session for the NWB writer test."
  timezone: "-05:00"
  lab: "Example Lab"
  institution: "Example University"
  experimenters: ["Tester"]
subject: { species: "Rattus norvegicus", sex: U, age: P90D }
electrode_groups:
  - { name: EMG, description: "3-channel EMG", location: diaphragm, impedance: { table: Z_EMG } }
streams:
  "*": { type: timeseries, unit: a.u. }
  EMG1: { type: electrical, electrode_group: EMG, name: EMG, conversion: 1.0e-6 }
  Skip: { include: false }
"#;

#[test]
fn writes_small_nwb_zarr() {
    let s = session();
    let meta = MetadataFile::parse(META).unwrap();
    let plan = nwb::resolve(&s, &meta, || "test-id".into());
    assert!(!plan.has_errors(), "{:?}", plan.issues);
    assert_eq!(plan.file.start_time, "2025-02-26T15:25:56-05:00");
    assert_eq!(plan.series.len(), 2);
    assert_eq!(plan.skipped, vec!["stream Skip".to_string()]);

    let dest = Path::new(env!("CARGO_MANIFEST_DIR")).join("target/nwb-test/small.nwb.zarr");
    let options = NwbOptions { overwrite: true, threads: 2, chunk_seconds: 0.3, gzip: Some(1) };
    let summary = nwb::write(&s, &plan, &dest, &options, &|_| {}).unwrap();
    assert_eq!(summary.samples, 3000 + 1000);

    // Spot checks of the layout hdmf-zarr expects
    let root: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dest.join("zarr.json")).unwrap()).unwrap();
    assert_eq!(root["attributes"]["neurodata_type"], "NWBFile");
    assert_eq!(root["attributes"][".specloc"], "specifications");
    let data: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dest.join("acquisition/EMG/data/zarr.json")).unwrap()).unwrap();
    assert_eq!(data["shape"], serde_json::json!([1000, 3]));
    assert_eq!(data["attributes"]["unit"], "volts");
    assert!(dest.join("specifications/core/2.11.0/nwb.ecephys/zarr.json").exists());
    assert!(dest.join("events/MET/timestamp").exists());
    assert!(dest.join("events/MET/duration").exists());
    assert!(dest.join("general/extracellular_ephys/electrodes/imp").exists());
    // Events without offsets: no duration column
    assert!(dest.join("events/Tick/timestamp").exists());
    assert!(!dest.join("events/Tick/duration").exists());

    let issues = nwb::validate::validate(&dest).unwrap();
    assert!(issues.is_empty(), "{issues:?}");

    // Damage a copy (the intact store stays for tests/validate_nwb.py): point the series at a
    // wrong table and drop an electrodes column
    let damaged = dest.with_file_name("damaged.nwb.zarr");
    let _ = std::fs::remove_dir_all(&damaged);
    copy_dir(&dest, &damaged);
    let dest = damaged;
    let region_meta = dest.join("acquisition/EMG/electrodes/zarr.json");
    let text = std::fs::read_to_string(&region_meta).unwrap().replace("/general/extracellular_ephys/electrodes", "/general/nowhere");
    std::fs::write(&region_meta, text).unwrap();
    std::fs::remove_dir_all(dest.join("general/extracellular_ephys/electrodes/channel_name")).unwrap();
    let issues = nwb::validate::validate(&dest).unwrap();
    let text: Vec<&str> = issues.iter().map(|i| i.message.as_str()).collect();
    assert!(text.iter().any(|m| m.contains("does not reference")), "{text:?}");
    assert!(text.iter().any(|m| m.contains("column channel_name listed but missing")), "{text:?}");
}

fn copy_dir(from: &Path, to: &Path) {
    std::fs::create_dir_all(to).unwrap();
    for e in std::fs::read_dir(from).unwrap().flatten() {
        let target = to.join(e.file_name());
        if e.file_type().unwrap().is_dir() {
            copy_dir(&e.path(), &target);
        } else {
            std::fs::copy(e.path(), target).unwrap();
        }
    }
}
