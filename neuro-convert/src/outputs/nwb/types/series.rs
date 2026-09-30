//! `ElectricalSeries` / `TimeSeries` in `/acquisition`.
//!
//! Continuous data is streamed: worker threads each take a chunk of rows, read it from the
//! source recording (channel-major), transpose it to NWB's `[time, channel]` layout and write
//! it. Memory stays at a few chunks regardless of recording length.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

use serde_json::json;

use super::{attrs, typed_with};
use crate::error::{Error, Result};
use crate::model::{EventSeries, Recording, SampleType};
use crate::outputs::nwb::backend::{Attrs, Backend};
use crate::outputs::nwb::mapping::{EventPlan, SeriesPlan};
use crate::outputs::nwb::types::electrodes::TABLE_PATH;

/// Writes a continuous series; `done` counts samples written (all channels) for progress.
pub fn write_continuous(
    b: &dyn Backend,
    plan: &SeriesPlan,
    rec: &dyn Recording,
    chunk_seconds: f64,
    threads: usize,
    done: &AtomicU64,
) -> Result<()> {
    let info = rec.info();
    let path = format!("/acquisition/{}", plan.name);
    let neurodata_type = if plan.electrode_group.is_some() { "ElectricalSeries" } else { "TimeSeries" };
    b.group(&path, typed_with("core", neurodata_type, &[("description", json!(plan.description)), ("comments", json!("no comments"))]))?;

    let (rows, cols) = (info.samples, info.channel_count() as u64);
    let data_attrs = attrs(&[
        ("unit", json!(plan.unit)),
        ("conversion", json!(plan.conversion)),
        ("offset", json!(0.0)),
        ("resolution", json!(-1.0)),
    ]);
    let (shape, dims): (Vec<u64>, Vec<&str>) = if plan.electrode_group.is_some() {
        (vec![rows, cols], vec!["num_times", "num_channels"])
    } else if cols == 1 {
        (vec![rows], vec!["num_times"])
    } else {
        (vec![rows, cols], vec!["num_times", "num_DIM2"])
    };
    let chunk = ((chunk_seconds * info.sample_rate).round() as u64).max(1);
    // Unscaled integer sources keep their type (half the size of float32); `conversion` scales
    let native = info.stored_as != SampleType::F32
        && info.stored_as != SampleType::F64
        && info.channels.iter().all(|c| c.gain == 1.0 && c.offset == 0.0)
        && rec.read_stored(&[], 0..0, &mut []).unwrap_or(false);
    let ty = if native { info.stored_as } else { SampleType::F32 };
    let sink = b.stream(&format!("{path}/data"), &shape, chunk, ty, &dims, data_attrs)?;

    if let Some(_group) = plan.electrode_group {
        let region: Vec<i64> = (plan.first_electrode as i64..(plan.first_electrode as u64 + cols) as i64).collect();
        let reference = json!({ "_REFERENCE": { "source": ".", "path": TABLE_PATH } });
        let a = typed_with("hdmf-common", "DynamicTableRegion", &[("description", json!("electrodes of this series")), ("table", reference)]);
        b.i64s(&format!("{path}/electrodes"), &region, "num_rows", a)?;
    }
    b.f64_scalar(&format!("{path}/starting_time"), info.start_time, attrs(&[("rate", json!(info.sample_rate)), ("unit", json!("seconds"))]))?;

    // Parallel chunked copy
    let chunks = rows.div_ceil(chunk);
    let next = AtomicU64::new(0);
    let failed: Mutex<Option<Error>> = Mutex::new(None);
    let channels: Vec<usize> = (0..cols as usize).collect();
    let es = ty.bytes();
    std::thread::scope(|scope| {
        for _ in 0..threads.max(1) {
            scope.spawn(|| {
                let (mut floats, mut src, mut dst) = (Vec::<f32>::new(), Vec::<u8>::new(), Vec::<u8>::new());
                loop {
                    let k = next.fetch_add(1, Ordering::Relaxed);
                    if k >= chunks || failed.lock().unwrap().is_some() {
                        break;
                    }
                    let (s0, s1) = (k * chunk, ((k + 1) * chunk).min(rows));
                    let n = (s1 - s0) as usize;
                    let c = channels.len();
                    src.resize(n * c * es, 0);
                    let res = if native {
                        rec.read_stored(&channels, s0..s1, &mut src).map(|_| ())
                    } else {
                        floats.resize(n * c, 0.0);
                        rec.read(&channels, s0..s1, &mut floats).map(|()| {
                            for (d, v) in src.chunks_exact_mut(4).zip(&floats) {
                                d.copy_from_slice(&v.to_le_bytes());
                            }
                        })
                    };
                    let res = res.and_then(|()| {
                        let out: &[u8] = if c == 1 {
                            &src
                        } else {
                            // channel-major → time-major, element by element
                            dst.resize(src.len(), 0);
                            for ch in 0..c {
                                for t in 0..n {
                                    let (from, to) = ((ch * n + t) * es, (t * c + ch) * es);
                                    dst[to..to + es].copy_from_slice(&src[from..from + es]);
                                }
                            }
                            &dst
                        };
                        sink.write_rows(s0, s1 - s0, out)
                    });
                    match res {
                        Ok(()) => {
                            done.fetch_add((s1 - s0) * cols, Ordering::Relaxed);
                        }
                        Err(e) => {
                            *failed.lock().unwrap() = Some(e);
                            break;
                        }
                    }
                }
            });
        }
    });
    match failed.into_inner().unwrap() {
        Some(e) => Err(e),
        None => Ok(()),
    }
}

/// `(start, rate)` when `times` are evenly spaced (within 0.01 % of the mean interval).
pub fn regular_rate(times: &[f64]) -> Option<(f64, f64)> {
    if times.len() < 3 {
        return None;
    }
    let mean = (times[times.len() - 1] - times[0]) / (times.len() - 1) as f64;
    if mean <= 0.0 {
        return None;
    }
    let regular = times.windows(2).all(|w| ((w[1] - w[0]) - mean).abs() <= mean * 1e-4);
    regular.then(|| (times[0], 1.0 / mean))
}

/// Events without offsets and multi-channel scalars: a `TimeSeries` with explicit timestamps,
/// or `starting_time` + `rate` when the events are evenly spaced (e.g. 1 s ticks).
pub fn write_events(b: &dyn Backend, plan: &EventPlan, e: &EventSeries) -> Result<()> {
    let path = format!("/acquisition/{}", plan.name);
    b.group(&path, typed_with("core", "TimeSeries", &[("description", json!(plan.description)), ("comments", json!("no comments"))]))?;
    let data_attrs = attrs(&[("unit", json!("a.u.")), ("conversion", json!(1.0)), ("offset", json!(0.0)), ("resolution", json!(-1.0))]);
    let n = e.len() as u64;
    if e.channels > 1 {
        b.f64s(&format!("{path}/data"), &e.values, &[n, e.channels as u64], &["num_times", "num_DIM2"], data_attrs)?;
    } else {
        b.f64s(&format!("{path}/data"), &e.values, &[n], &["num_times"], data_attrs)?;
    }
    if let Some((start, rate)) = regular_rate(&e.onsets) {
        return b.f64_scalar(&format!("{path}/starting_time"), start, attrs(&[("rate", json!(rate)), ("unit", json!("seconds"))]));
    }
    let ts_attrs: Attrs = attrs(&[("interval", json!(1)), ("unit", json!("seconds"))]);
    b.f64s(&format!("{path}/timestamps"), &e.onsets, &[n], &["num_times"], ts_attrs)
}

#[cfg(test)]
mod tests {
    use super::regular_rate;

    #[test]
    fn test_regular_rate() {
        let ticks: Vec<f64> = (0..10).map(|i| 4.096e-5 + i as f64 * 1.000_002_56).collect();
        let (start, rate) = regular_rate(&ticks).unwrap();
        assert!((start - 4.096e-5).abs() < 1e-12 && (rate - 1.0 / 1.000_002_56).abs() < 1e-9);
        assert!(regular_rate(&[0.0, 1.0, 3.0]).is_none());
        assert!(regular_rate(&[0.0, 1.0]).is_none());
    }
}
