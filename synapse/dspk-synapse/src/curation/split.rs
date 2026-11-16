//! Choosing spikes to split off a unit: those a lasso holds in a scatter view.

use super::SortingData;

/// Whether `p` lies inside `polygon` (ray casting; a polygon of fewer than 3 points holds nothing).
pub fn inside(p: (f32, f32), polygon: &[(f32, f32)]) -> bool {
    if polygon.len() < 3 {
        return false;
    }
    let mut on = false;
    let mut j = polygon.len() - 1;
    for i in 0..polygon.len() {
        let ((xi, yi), (xj, yj)) = (polygon[i], polygon[j]);
        if (yi > p.1) != (yj > p.1) && p.0 < (xj - xi) * (p.1 - yi) / (yj - yi) + xi {
            on = !on;
        }
        j = i;
    }
    on
}

/// The spike times of unit `unit` (an index into `sorting.units`) whose (time s, amplitude)
/// lies inside `polygon`: the amplitude view's lasso.
pub fn spikes_in_polygon(sorting: &SortingData, unit: usize, polygon: &[(f32, f32)]) -> Vec<u64> {
    if sorting.amplitudes.is_empty() {
        return Vec::new();
    }
    sorting.by_unit[unit]
        .iter()
        .map(|&k| k as usize)
        .filter(|&k| inside(((sorting.times[k] as f64 / sorting.sample_rate) as f32, sorting.amplitudes[k]), polygon))
        .map(|k| sorting.times[k])
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_lasso_holds_the_points_inside_it() {
        let square = [(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)];
        assert!(inside((1.0, 1.0), &square));
        assert!(!inside((3.0, 1.0), &square));
        assert!(!inside((1.0, -0.5), &square));
        assert!(!inside((1.0, 1.0), &square[..2]), "two points hold nothing");
        // A concave lasso: the notch is outside
        let notched = [(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (2.0, 1.0), (0.0, 4.0)];
        assert!(inside((1.0, 0.5), &notched));
        assert!(!inside((2.0, 3.0), &notched));
    }
}
