//! PyO3 bindings for sorter-to-sorter comparison and spike train matching.

use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyfunction};
use pyo3::types::{PyDict, PyList};

use crate::array::to_numpy;
use dsp_synapse::metrics::comparison::{
    compare_sortings as rust_compare_sortings,
    compare_spike_trains as rust_compare_spike_trains,
};
use dsp_synapse::metrics::{DEFAULT_AGREEMENT_THRESHOLD, DEFAULT_MATCH_DELTA_MS};

const MS_PER_S: f64 = 1e3;

use super::storage::PySortingOutput;

/// Matches two spike trains one to one within a tolerance (SpikeInterface's comparison).
///
/// Parameters
/// ----------
/// train_a, train_b : list of int
///     Spike times, recording samples (A: the reference, e.g. ground truth).
/// fs : float
///     Sampling rate, Hz.
/// delta_time_ms : float, default 0.4
///     Two spikes match when this close, ms.
///
/// Returns
/// -------
/// dict
///     `num_spikes_a`, `num_spikes_b`, `num_matches`; `precision` (matches / B), `recall` (matches / A),
///     `accuracy` (matches / (A + B − matches)), `f1_score`, `agreement_score`; `false_negatives`
///     (unmatched in A), `false_positives` (unmatched in B), `false_negative_rate`,
///     `false_positive_rate`.
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (train_a, train_b, *, fs, delta_time_ms=DEFAULT_MATCH_DELTA_MS))]
pub fn compare_spike_trains<'py>(
    py: Python<'py>,
    train_a: Vec<u64>,
    train_b: Vec<u64>,
    fs: f64,
    delta_time_ms: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let tolerance_samples = ((delta_time_ms / MS_PER_S * fs).round() as u64).max(1);
    let res = rust_compare_spike_trains(&train_a, &train_b, tolerance_samples);

    let f1 = if res.precision + res.recall > 0.0 {
        (2.0 * res.precision * res.recall / (res.precision + res.recall)) as f64
    } else {
        0.0
    };

    let dict = PyDict::new(py);
    dict.set_item("num_spikes_a", res.num_spikes_a)?;
    dict.set_item("num_spikes_b", res.num_spikes_b)?;
    dict.set_item("num_matches", res.num_matches)?;
    dict.set_item("precision", res.precision as f64)?;
    dict.set_item("recall", res.recall as f64)?;
    dict.set_item("accuracy", res.accuracy as f64)?;
    dict.set_item("f1_score", f1)?;
    dict.set_item("agreement_score", res.agreement_score as f64)?;
    dict.set_item("false_negatives", res.false_negatives)?;
    dict.set_item("false_positives", res.false_positives)?;
    dict.set_item("false_negative_rate", res.false_negative_rate as f64)?;
    dict.set_item("false_positive_rate", res.false_positive_rate as f64)?;
    Ok(dict)
}

/// Compares two sortings unit by unit: the agreement of every pair, then the best one-to-one pairing.
///
/// Parameters
/// ----------
/// sorting_a, sorting_b : SortingOutput
///     A: the reference (e.g. ground truth or another sorter).
/// delta_time_ms : float, default 0.4
///     Two spikes match when this close, ms.
/// agreement_threshold : float, default 0.5
///     Pairs below this agreement are not matched.
///
/// Returns
/// -------
/// dict
///     `sorter_a`, `sorter_b`; `agreement_matrix` (`[units A, units B]` float32), `unit_ids_a`,
///     `unit_ids_b`; `matches` (dicts: `unit_id_a`, `unit_id_b`, `agreement`, `accuracy`, `precision`,
///     `recall`, `f1_score`, `num_matches`, `num_spikes_a`, `num_spikes_b`); `well_detected_units_a`,
///     `missed_units_a`, `false_positive_units_b`; `mean_agreement`, `mean_precision`, `mean_recall`,
///     `mean_f1` over matched pairs (NaN when none matched).
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (sorting_a, sorting_b, *, delta_time_ms=DEFAULT_MATCH_DELTA_MS, agreement_threshold=DEFAULT_AGREEMENT_THRESHOLD))]
pub fn compare_sortings<'py>(
    py: Python<'py>,
    sorting_a: &PySortingOutput,
    sorting_b: &PySortingOutput,
    delta_time_ms: f64,
    agreement_threshold: f32,
) -> PyResult<Bound<'py, PyDict>> {
    let comp = rust_compare_sortings(
        sorting_a.inner(),
        sorting_b.inner(),
        delta_time_ms,
        agreement_threshold,
    );

    let n_a = comp.unit_ids_a.len();
    let n_b = comp.unit_ids_b.len();
    let matrix_arr = to_numpy(py, comp.agreement_matrix, &[n_a, n_b])?;

    let mut sum_prec = 0.0f64;
    let mut sum_rec = 0.0f64;
    let mut sum_f1 = 0.0f64;
    let n_matches = comp.matched_units.len();

    let matches_list = PyList::empty(py);
    for m in &comp.matched_units {
        let f1 = if m.metrics.precision + m.metrics.recall > 0.0 {
            (2.0 * m.metrics.precision * m.metrics.recall
                / (m.metrics.precision + m.metrics.recall)) as f64
        } else {
            0.0
        };
        sum_prec += m.metrics.precision as f64;
        sum_rec += m.metrics.recall as f64;
        sum_f1 += f1;

        let md = PyDict::new(py);
        md.set_item("unit_id_a", m.unit_id_a)?;
        md.set_item("unit_id_b", m.unit_id_b)?;
        md.set_item("agreement", m.metrics.agreement_score as f64)?;
        md.set_item("accuracy", m.metrics.accuracy as f64)?;
        md.set_item("precision", m.metrics.precision as f64)?;
        md.set_item("recall", m.metrics.recall as f64)?;
        md.set_item("f1_score", f1)?;
        md.set_item("num_matches", m.metrics.num_matches)?;
        md.set_item("num_spikes_a", m.metrics.num_spikes_a)?;
        md.set_item("num_spikes_b", m.metrics.num_spikes_b)?;
        matches_list.append(md)?;
    }

    let mean = |sum: f64| if n_matches > 0 { sum / n_matches as f64 } else { f64::NAN };
    let (mean_prec, mean_rec, mean_f1) = (mean(sum_prec), mean(sum_rec), mean(sum_f1));

    let dict = PyDict::new(py);
    dict.set_item("sorter_a", comp.sorter_a)?;
    dict.set_item("sorter_b", comp.sorter_b)?;
    dict.set_item("agreement_matrix", matrix_arr)?;
    dict.set_item("unit_ids_a", comp.unit_ids_a)?;
    dict.set_item("unit_ids_b", comp.unit_ids_b)?;
    dict.set_item("matches", matches_list)?;
    dict.set_item("well_detected_units_a", comp.well_detected_units_a)?;
    dict.set_item("missed_units_a", comp.missed_units_a)?;
    dict.set_item("false_positive_units_b", comp.false_positive_units_b)?;
    dict.set_item("mean_agreement", comp.mean_matched_agreement as f64)?;
    dict.set_item("mean_precision", mean_prec)?;
    dict.set_item("mean_recall", mean_rec)?;
    dict.set_item("mean_f1", mean_f1)?;
    Ok(dict)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compare_spike_trains, m)?)?;
    m.add_function(wrap_pyfunction!(compare_sortings, m)?)?;
    Ok(())
}
