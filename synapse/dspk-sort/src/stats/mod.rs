//! Small statistics shared by the sorters.

/// NumPy's median (`v` reordered).
pub fn median(v: &mut [f32]) -> f64 {
    let n = v.len();
    if n == 0 {
        return 0.0;
    }
    let (lower, mid, _) = v.select_nth_unstable_by(n / 2, f32::total_cmp);
    let mid = *mid as f64;
    if n % 2 == 1 { mid } else { 0.5 * (lower.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64 + mid) }
}
