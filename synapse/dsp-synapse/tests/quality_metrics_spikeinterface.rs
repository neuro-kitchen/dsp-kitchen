//! Quality metrics equal SpikeInterface 0.105 (fixtures from `spikeinterface_reference.py`).

use dsp_synapse::metrics::{
    compute_amplitude_cutoff, compute_isi_violations, compute_llobet_contamination, compute_presence_ratio,
    count_refractory_violations,
};
use serde_json::Value;

fn fixtures() -> Value {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/quality_metrics.json");
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn same(got: f64, expected: &Value, what: &str) {
    match expected.as_f64() {
        None => assert!(got.is_nan(), "{what}: expected NaN, got {got}"),
        Some(e) => assert!((got - e).abs() <= 1e-9 * e.abs().max(1.0), "{what}: got {got}, SpikeInterface {e}"),
    }
}

#[test]
fn metrics_match_spikeinterface() {
    let fx = fixtures();
    let fs = fx["fs"].as_f64().unwrap();
    let total = fx["total_samples"].as_u64().unwrap();
    let duration = total as f64 / fs;
    for unit in fx["units"].as_array().unwrap() {
        let name = unit["name"].as_str().unwrap();
        let spikes: Vec<u64> = unit["spike_samples"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap()).collect();
        let amps: Vec<f64> = unit["amplitudes"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();

        let isi = compute_isi_violations(&spikes, fs, duration, fx["isi_threshold_ms"].as_f64().unwrap(), 0.0);
        assert_eq!(isi.violation_count as u64, unit["isi_violations_count"].as_u64().unwrap(), "{name}: ISI count");
        same(isi.isi_violations_ratio, &unit["isi_violations_ratio"], &format!("{name}: ISI ratio"));
        same(isi.violations_per_sec, &unit["isi_violations_rate"], &format!("{name}: ISI rate"));

        let t_r = (fx["refractory_ms"].as_f64().unwrap() * fs * 1e-3).round() as u64;
        assert_eq!(count_refractory_violations(&spikes, t_r), unit["rp_violations"].as_u64().unwrap(), "{name}: rp violations");
        let rp = compute_llobet_contamination(&spikes, total, fs, fx["refractory_ms"].as_f64().unwrap(), fx["censored_ms"].as_f64().unwrap());
        same(rp, &unit["rp_contamination"], &format!("{name}: rp contamination"));

        same(compute_amplitude_cutoff(&amps), &unit["amplitude_cutoff"], &format!("{name}: amplitude cutoff"));
        same(
            compute_presence_ratio(&spikes, total, fs, fx["presence_bin_s"].as_f64().unwrap(), 0.0),
            &unit["presence_ratio"],
            &format!("{name}: presence ratio"),
        );
    }
}
