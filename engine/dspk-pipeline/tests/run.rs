//! Spec parsing and validation; the executor against the same stages run on the whole recording
//! at once, on every runtime of `ComputeTarget::for_tests()` (DSPK_TEST_RUNTIMES).

use std::path::{Path, PathBuf};

use dspk_core::{ComputeTarget, MemoryOrder, MemoryRecording, SampleFormat};
use dspk_ops::filter::design::DEFAULT_SETTLING_TOLERANCE;
use dspk_ops::pipeline::PipelineWorkspace;
use dspk_pipeline::{run, ExecutedMode, PipelineError, PipelineSpec, PROVENANCE_SUFFIX};

const FS: f64 = 30_000.0;
const CHANNELS: usize = 6;
const SAMPLES: usize = 50_000;
/// Not a divisor of SAMPLES, so the last window is short
const CHUNK: usize = 7_001;

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("dspk_pipeline_{name}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Tones at 50 Hz (shared by all channels), 400·(c+1) Hz and 9 kHz, plus a DC offset.
fn recording() -> MemoryRecording {
    let data = (0..CHANNELS)
        .flat_map(|c| {
            (0..SAMPLES).map(move |i| {
                let t = i as f64 / FS;
                let x = (2.0 * std::f64::consts::PI * 50.0 * t).sin()
                    + (2.0 * std::f64::consts::PI * 400.0 * (c + 1) as f64 * t).sin()
                    + 0.3 * (2.0 * std::f64::consts::PI * 9_000.0 * t).sin()
                    + 2.0;
                (100.0 * x) as f32
            })
        })
        .collect();
    MemoryRecording::new("synthetic", data, CHANNELS, FS).unwrap()
}

fn write_input(dir: &Path) -> MemoryRecording {
    let rec = recording();
    dspk_io::write_raw(&rec, &dir.join("in.bin"), SampleFormat::F32, MemoryOrder::ChannelMajor, 1.0, 10_000, |_, _| {}).unwrap();
    rec
}

fn read_all(path: &Path) -> Vec<f32> {
    let rec = dspk_io::open(path).unwrap();
    let all: Vec<usize> = (0..rec.info().channel_count()).collect();
    let mut out = vec![0.0f32; all.len() * rec.info().samples as usize];
    rec.read(&all, 0..rec.info().samples, &mut out).unwrap();
    out
}

fn spec(stages: &str, edges: &str, extra: &str) -> PipelineSpec {
    PipelineSpec::from_json(&format!(
        r#"{{"dspk_pipeline": 1,
            "nodes": {{"in": {{"op": "read", "path": "in.bin"}}, {stages}
                       "out": {{"op": "write", "path": "out.bin"}}}},
            "edges": {edges} {extra}}}"#
    ))
    .unwrap()
}

/// The chain's stages over the whole recording as one chunk, on `target`.
fn whole_recording(spec: &PipelineSpec, target: ComputeTarget, stateful: bool) -> Vec<f32> {
    let chain = spec.chain().unwrap();
    let stages = chain.stages.iter().map(|(_, n)| n.stage().unwrap().unwrap()).collect();
    let pipeline = dspk_ops::Pipeline::with_stages(stages);
    let client = target.client().unwrap();
    let mut ws = if stateful {
        PipelineWorkspace::<f32>::new_stateful(client, pipeline, CHANNELS, SAMPLES, FS).unwrap()
    } else {
        PipelineWorkspace::<f32>::new(client, pipeline, CHANNELS, SAMPLES, FS).unwrap()
    };
    let rec = recording();
    let mut out = vec![0.0f32; CHANNELS * SAMPLES];
    ws.process_chunk(rec.data(), SAMPLES, &mut out);
    out
}

fn max_rel_err(got: &[f32], expected: &[f32]) -> f32 {
    assert_eq!(got.len(), expected.len());
    let peak = expected.iter().fold(0.0f32, |m, x| m.max(x.abs()));
    got.iter().zip(expected).fold(0.0f32, |m, (a, b)| m.max((a - b).abs())) / peak
}

fn targets() -> Vec<ComputeTarget> {
    let targets = ComputeTarget::for_tests();
    assert!(!targets.is_empty(), "no CubeCL runtime to test");
    targets
}

#[test]
fn toml_and_json_give_the_same_spec_and_json_round_trips() {
    let json = PipelineSpec::from_json(
        r#"{"dspk_pipeline": 1,
            "nodes": {"raw": {"op": "read", "path": "recording.bin"},
                      "hp": {"op": "iir", "band": "highpass", "cutoff": [300.0], "mode": "forward"},
                      "out": {"op": "write", "path": "filtered.zarr", "format": "zarr"}},
            "edges": [["raw", "hp"], ["hp", "out"]]}"#,
    )
    .unwrap();
    let toml = PipelineSpec::from_toml(
        r#"
        dspk_pipeline = 1
        edges = [["raw", "hp"], ["hp", "out"]]

        [nodes.raw]
        op = "read"
        path = "recording.bin"

        [nodes.hp]
        op = "iir"            # Butterworth high-pass, causal
        band = "highpass"
        cutoff = [300.0]
        mode = "forward"

        [nodes.out]
        op = "write"
        path = "filtered.zarr"
        format = "zarr"
        "#,
    )
    .unwrap();
    assert_eq!(json, toml);
    assert_eq!(PipelineSpec::from_json(&json.to_json()).unwrap(), json);
    // Defaults are written out in the canonical form
    let canonical = json.to_json();
    assert!(canonical.contains("\"order\": 5") && canonical.contains("\"chunk_samples\": 65536"), "{canonical}");
    let schema = PipelineSpec::json_schema();
    assert!(schema.contains("dspk_pipeline") && schema.contains("teager_kaiser"));
}

#[test]
fn invalid_specs_are_refused_with_a_reason() {
    let base = r#""in": {"op": "read", "path": "a.bin"}, "out": {"op": "write", "path": "b.bin"}"#;
    let check = |nodes: &str, edges: &str, want: &str| {
        let text = format!(r#"{{"dspk_pipeline": 1, "nodes": {{{base}{nodes}}}, "edges": {edges}}}"#);
        let err = match PipelineSpec::from_json(&text) {
            Ok(spec) => spec.chain().err().map(|e| e.to_string()).unwrap_or_else(|| "accepted".into()),
            Err(e) => e.to_string(),
        };
        assert!(err.contains(want), "expected {want:?}, got {err:?}");
    };
    check("", r#"[["in", "out"], ["in", "out"]]"#, "more than one output");
    check(r#", "a": {"op": "car"}, "b": {"op": "cmr"}"#, r#"[["in", "a"], ["in", "b"], ["a", "out"]]"#, "more than one output");
    check(r#", "a": {"op": "car"}"#, r#"[["in", "out"]]"#, "not on the chain");
    check(r#", "a": {"op": "car"}"#, r#"[["in", "out"], ["a", "out"]]"#, "more than one input");
    check(r#", "a": {"op": "car"}, "b": {"op": "cmr"}"#, r#"[["in", "out"], ["a", "b"], ["b", "a"]]"#, "not on the chain");
    check("", r#"[["in", "nowhere"]]"#, "no node");
    check(r#", "a": {"op": "car", "strength": 2}"#, r#"[["in", "a"], ["a", "out"]]"#, "unknown field");
    check(r#", "a": {"op": "iir", "band": "lowpass", "cutof": [10.0]}"#, r#"[["in", "a"], ["a", "out"]]"#, "unknown field");
    let bad_version = r#"{"dspk_pipeline": 2, "nodes": {}, "edges": []}"#;
    assert!(PipelineSpec::from_json(bad_version).unwrap().chain().unwrap_err().to_string().contains("version 2"));
    let cheby = spec(r#""f": {"op": "iir", "band": "lowpass", "cutoff": [1000.0], "design": "chebyshev1"},"#, r#"[["in", "f"], ["f", "out"]]"#, "");
    let err = cheby.chain().unwrap().stages[0].1.stage().unwrap_err();
    assert!(err.to_string().contains("ripple_db"), "{err}");
}

#[test]
fn stream_mode_in_uneven_windows_equals_the_whole_recording() {
    for target in targets() {
        let dir = scratch(&format!("stream_{}", target.name()));
        write_input(&dir);
        let spec = spec(
            r#""s": {"op": "scale", "alpha": 0.5}, "c": {"op": "car"},
               "hp": {"op": "iir", "band": "highpass", "cutoff": [300.0], "mode": "forward"},"#,
            r#"[["in", "s"], ["s", "c"], ["c", "hp"], ["hp", "out"]]"#,
            &format!(r#", "execution": {{"chunk_samples": {CHUNK}, "runtime": "{}"}}"#, target.name()),
        );
        let report = run(&spec, &dir).unwrap();
        assert_eq!(report.mode, ExecutedMode::Stream);
        assert_eq!((report.channels, report.samples, report.windows), (CHANNELS, SAMPLES as u64, SAMPLES.div_ceil(CHUNK)));
        let got = read_all(&dir.join("out.bin"));
        let expected = whole_recording(&spec, target, true);
        let err = max_rel_err(&got, &expected);
        assert!(err < 1e-6, "{}: stream vs whole recording, max error {err:.2e} of the peak", target.name());
        std::fs::remove_dir_all(&dir).unwrap();
    }
}

#[test]
fn window_mode_stays_within_the_settling_tolerance() {
    for target in targets() {
        let dir = scratch(&format!("windows_{}", target.name()));
        write_input(&dir);
        let spec = spec(
            r#""bp": {"op": "iir", "band": "bandpass", "cutoff": [300.0, 6000.0]}, "m": {"op": "median", "width": 5},"#,
            r#"[["in", "bp"], ["bp", "m"], ["m", "out"]]"#,
            &format!(r#", "execution": {{"chunk_samples": {CHUNK}, "runtime": "{}"}}"#, target.name()),
        );
        let report = run(&spec, &dir).unwrap();
        let ExecutedMode::Windows { left, right } = report.mode else { panic!("expected windows, got {:?}", report.mode) };
        assert!(left > 2 && right > 2, "zero-phase filter and median need context on both sides");
        let got = read_all(&dir.join("out.bin"));
        let expected = whole_recording(&spec, target, false);
        let err = max_rel_err(&got, &expected);
        assert!(err <= DEFAULT_SETTLING_TOLERANCE as f32, "{}: windows vs whole recording, max error {err:.2e} of the peak", target.name());
        std::fs::remove_dir_all(&dir).unwrap();
    }
}

#[test]
fn stream_mode_is_refused_for_zero_phase_filters() {
    let dir = scratch("refuse");
    write_input(&dir);
    let spec = spec(
        r#""bp": {"op": "iir", "band": "bandpass", "cutoff": [300.0, 6000.0]},"#,
        r#"[["in", "bp"], ["bp", "out"]]"#,
        r#", "execution": {"mode": "stream"}"#,
    );
    let err = run(&spec, &dir).unwrap_err();
    assert!(matches!(err, PipelineError::Invalid(_)) && err.to_string().contains("zero-phase"), "{err}");
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn zarr_output_matches_raw_and_provenance_is_written() {
    let target = targets()[0];
    let dir = scratch("zarr");
    write_input(&dir);
    let stages = r#""hp": {"op": "iir", "band": "highpass", "cutoff": [300.0], "mode": "forward"},"#;
    let edges = r#"[["in", "hp"], ["hp", "out"]]"#;
    let exec = format!(r#", "execution": {{"chunk_samples": {CHUNK}, "runtime": "{}"}}"#, target.name());
    let raw = spec(stages, edges, &exec);
    run(&raw, &dir).unwrap();
    let mut zarr = raw.clone();
    zarr.nodes.insert("out".into(), PipelineSpec::from_json(r#"{"dspk_pipeline": 1, "nodes": {"o": {"op": "write", "path": "out.zarr", "format": "zarr"}}, "edges": []}"#).unwrap().nodes["o"].clone());
    let report = run(&zarr, &dir).unwrap();
    assert_eq!(read_all(&dir.join("out.zarr")), read_all(&dir.join("out.bin")));
    assert!(!dir.join("out.dspk-tmp.bin").exists(), "temporary raw file removed");
    assert_eq!(report.provenance, dir.join(format!("out.zarr{PROVENANCE_SUFFIX}")));
    assert_eq!(PipelineSpec::load(&report.provenance).unwrap(), zarr);
    std::fs::remove_dir_all(&dir).unwrap();
}
