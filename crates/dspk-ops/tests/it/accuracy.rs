//! Accuracy sweep: Butterworth low- and high-passes over a grid of orders (1–16) and cutoff ratios
//! (1e-4 to 0.45 of the sample rate), filtered on the device in float32 and compared with the same
//! design filtered on the host in float64. The error table is printed (`--nocapture`); the limits
//! hold where float32 is expected to hold (the state-variable kernel at the float32 rounding
//! floor), and the rest is reported, not hidden.
//!
//! Also the precision rule for recursive filters: designed in float64, computed in float32 or wider,
//! never float16 or bfloat16 (rounded coefficients move poles next to z = 1 out of the unit circle:
//! braindecode #1657). Checked here: every design's sections, rounded to float32, stay stable.

#![cfg(feature = "wgpu")]

use crate::common::*;
use dspk_ops::filter::{FilterBand, FilterMode, FilterSpec, FilterStart, Sos};

/// The recording rate the sweep runs at (the Author's HD-EMG).
const FS: f64 = 24_414.0625;

/// Cutoffs as fractions of the sample rate (the Author's 20 Hz low-pass is 8.2e-4).
const RATIOS: [f64; 9] = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 0.45];
const ORDERS: [usize; 6] = [1, 2, 4, 8, 12, 16];

/// Largest error (fraction of the signal's peak) zero-phase and steady-state passes may have up to
/// order 8: the float32 floor the low-cutoff test measured (~2e-5), with room for higher orders.
const LIMIT_TO_ORDER_8: f64 = 2e-4;

/// Whether every section of `sos`, rounded to float32, keeps its poles inside the unit circle
/// (a biquad 1 + a1 z⁻¹ + a2 z⁻² is stable when |a2| < 1 and |a1| < 1 + a2).
fn stable_in_f32(sos: &Sos) -> bool {
    sos.sections.iter().all(|s| {
        let (a1, a2) = (s.a[1] as f32 as f64, s.a[2] as f32 as f64);
        a2.abs() < 1.0 && a1.abs() < 1.0 + a2
    })
}

#[test]
fn iir_designs_hold_float32_accuracy_across_orders_and_cutoffs() {
    let clients = test_clients();
    let x = test_signal(80_000, FS);
    let scale = max_abs(&x);
    let mut table = Vec::new();
    let mut over = Vec::new();
    for &order in &ORDERS {
        for &ratio in &RATIOS {
            for (band_name, band) in [("lowpass", FilterBand::Lowpass(ratio * FS)), ("highpass", FilterBand::Highpass(ratio * FS))] {
                let Ok(sos) = FilterSpec::butterworth(order, band).design(FS) else { continue };
                if !stable_in_f32(&sos) {
                    // Reported, and an error up to order 8 (beyond, the sweep shows where it stops)
                    table.push(format!("{band_name:>8} order {order:>2} at {ratio:>7.1e}: a section rounded to float32 is unstable"));
                    if order <= 8 {
                        over.push(format!("{band_name} order {order} at {ratio}: unstable in float32"));
                    }
                    continue;
                }
                let pad = sos.settling_samples(1e-3).min(x.len() - 1);
                let expected_fb = sos.filtfilt(&x, pad);
                let expected_steady = sos.filter(&x, true);
                for client in &clients {
                    let fb = &run_device(client, &sos, FilterMode::ForwardBackward, FilterStart::Rest, &x, 1)[0];
                    let steady = &run_device(client, &sos, FilterMode::Forward, FilterStart::SteadyState, &x, 1)[0];
                    let (e_fb, e_steady) = (max_abs_diff(fb, &expected_fb) / scale, max_abs_diff(steady, &expected_steady) / scale);
                    let db = |e: f64| 20.0 * e.max(1e-12).log10();
                    table.push(format!("{:>8} order {order:>2} at {ratio:>7.1e}: zero-phase {e_fb:.2e} ({:>6.1} dB), steady {e_steady:.2e} ({:>6.1} dB)", band_name, db(e_fb), db(e_steady)));
                    if order <= 8 && (e_fb > LIMIT_TO_ORDER_8 || e_steady > LIMIT_TO_ORDER_8) {
                        over.push(format!("{} {band_name} order {order} at {ratio}: {e_fb:.2e} / {e_steady:.2e}", client.name()));
                    }
                }
            }
        }
    }
    println!("float32 error of the peak, device against float64:\n{}", table.join("\n"));
    assert!(over.is_empty(), "over {LIMIT_TO_ORDER_8:.0e} of the peak:\n{}", over.join("\n"));
}
