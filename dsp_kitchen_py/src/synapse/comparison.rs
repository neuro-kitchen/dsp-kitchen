//! PyO3 bindings for sorter-to-sorter comparison and spike train matching.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::array::to_numpy;
use dsp_synapse::metrics::comparison::{
    compare_sortings as rust_compare_sortings,
    compare_spike_trains as rust_compare_spike_trains,
};

use super::storage::PySortingOutput;

/// Compares two chronological spike trains within a symmetric tolerance window `delta_time_ms`.
///
/// Returns a dictionary with:
/// - `num_spikes_a`: Total spikes in train A
/// - `num_spikes_b`: Total spikes in train B
/// - `num_matches`: Number of 1-to-1 paired spikes within tolerance
/// - `precision`: Matches / Spikes B
/// - `recall`: Matches / Spikes A
/// - `accuracy`: Matches / (Spikes A + Spikes B - Matches)
/// - `f1_score`: $2 \times P \times R / (P + R)$
/// - `agreement_score`: Intersection-over-Union accuracy
/// - `false_negatives`: Unmatched spikes in A
/// - `false_positives`: Unmatched spikes in B
#[pyfunction]
#[pyo3(signature = (train_a, train_b, sample_rate_hz=30000.0, delta_time_ms=0.4))]
pub fn compare_spike_trains<'py>(
    py: Python<'py>,
    train_a: Vec<u64>,
    train_b: Vec<u64>,
    sample_rate_hz: f64,
    delta_time_ms: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let tolerance_samples = ((delta_time_ms * 1e-3 * sample_rate_hz).round() as u64).max(1);
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

/// Computes pairwise agreement between two [`SortingOutput`] instances and performs
/// greedy best-match pairing above `agreement_threshold`.
///
/// Returns a dictionary with:
/// - `sorter_a`: Name of sorter A
/// - `sorter_b`: Name of sorter B
/// - `agreement_matrix`: 2D NumPy array `[num_units_a, num_units_b]` with accuracy scores
/// - `unit_ids_a`: List of unit IDs from sorting A
/// - `unit_ids_b`: List of unit IDs from sorting B
/// - `matches`: List of match dictionaries (`unit_id_a`, `unit_id_b`, `agreement`, `precision`, `recall`, `f1_score`, `matches`)
/// - `well_detected_units_a`: List of units in A with agreement >= `agreement_threshold`
/// - `missed_units_a`: List of units in A with no match above `agreement_threshold`
/// - `false_positive_units_b`: List of units in B with no match above `agreement_threshold`
/// - `mean_agreement`: Mean agreement score across matched pairs
/// - `mean_accuracy`: Alias for mean agreement
/// - `mean_precision`: Mean precision across matched pairs
/// - `mean_recall`: Mean recall across matched pairs
/// - `mean_f1`: Mean F1 score across matched pairs
#[pyfunction]
#[pyo3(signature = (sorting_a, sorting_b, delta_time_ms=0.4, agreement_threshold=0.5))]
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

    let mean_prec = if n_matches > 0 { sum_prec / n_matches as f64 } else { 0.0 };
    let mean_rec = if n_matches > 0 { sum_rec / n_matches as f64 } else { 0.0 };
    let mean_f1 = if n_matches > 0 { sum_f1 / n_matches as f64 } else { 0.0 };

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
    dict.set_item("mean_accuracy", comp.mean_matched_agreement as f64)?;
    dict.set_item("mean_precision", mean_prec)?;
    dict.set_item("mean_recall", mean_rec)?;
    dict.set_item("mean_f1", mean_f1)?;
    Ok(dict)
}
