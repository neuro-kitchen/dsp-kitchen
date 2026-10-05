#![allow(dead_code)]

use dsp_base::filter::{FilterBand, FilterSpec, Section, Sos};
use serde_json::Value;

/// Same formula as `test_signal` in `scipy_reference.py`.
pub fn test_signal(n: usize, fs: f64) -> Vec<f64> {
    use std::f64::consts::PI;
    let mut x: Vec<f64> = (0..n)
        .map(|i| {
            let t = i as f64 / fs;
            -500.0
                + 40.0 * (2.0 * PI * 60.0 * t).sin()
                + 25.0 * (2.0 * PI * 1_000.0 * t + 0.3).sin()
                + 10.0 * (2.0 * PI * 4_500.0 * t + 1.1).sin()
                + 5.0 * (2.0 * PI * 9_000.0 * t).sin()
        })
        .collect();
    for v in &mut x[n / 2..] {
        *v += 80.0;
    }
    // numpy.hanning(30)
    let m = 30;
    for k in 0..m {
        let w = 0.5 - 0.5 * (2.0 * PI * k as f64 / (m - 1) as f64).cos();
        x[n / 3 + k] -= 150.0 * w;
    }
    x
}

pub fn fixtures() -> Value {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/filters.json");
    serde_json::from_str(&std::fs::read_to_string(path).expect("run tests/scipy_reference.py"))
        .expect("valid fixture json")
}

pub fn floats(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}

pub fn sos_from(v: &Value) -> Sos {
    Sos::new(
        v.as_array()
            .unwrap()
            .iter()
            .map(|row| {
                let r = floats(row);
                Section { b: [r[0], r[1], r[2]], a: [r[3], r[4], r[5]] }
            })
            .collect(),
    )
}

/// The `FilterSpec` a fixture case was generated from.
pub fn spec_from(v: &Value) -> FilterSpec {
    match v["kind"].as_str().unwrap() {
        "butter" => {
            let order = v["order"].as_u64().unwrap() as usize;
            let wn = &v["wn"];
            let band = match v["btype"].as_str().unwrap() {
                "lowpass" => FilterBand::Lowpass(wn.as_f64().unwrap()),
                "highpass" => FilterBand::Highpass(wn.as_f64().unwrap()),
                "bandpass" => {
                    let w = floats(wn);
                    FilterBand::Bandpass(w[0], w[1])
                }
                "bandstop" => {
                    let w = floats(wn);
                    FilterBand::Bandstop(w[0], w[1])
                }
                other => panic!("unknown btype {other}"),
            };
            FilterSpec::butterworth(order, band)
        }
        "notch" => FilterSpec::notch(v["f0"].as_f64().unwrap(), v["q"].as_f64().unwrap()),
        other => panic!("unknown kind {other}"),
    }
}

pub fn max_abs(v: &[f64]) -> f64 {
    v.iter().fold(0.0, |m, x| m.max(x.abs()))
}

pub fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).fold(0.0, |m, (x, y)| m.max((x - y).abs()))
}
