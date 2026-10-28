//! `scipy.signal.find_peaks` on the host, plus troughs and both polarities.

use num_traits::Float;

/// Which extrema count as peaks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Polarity {
    /// Local maxima of `x` (scipy).
    #[default]
    Positive,
    /// Local minima of `x` (maxima of `−x`).
    Negative,
    /// Both; conditions apply to the magnitude `|x|` of each extremum.
    Both,
}

/// How `distance` resolves peaks closer than it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DistanceRule {
    /// scipy `_select_by_peak_distance`: the largest peak is kept and every peak nearer than
    /// `distance` removed, then the next largest remaining, … A removed peak removes nothing, so a
    /// peak's fate can depend on peaks farther than `distance` (a chain).
    #[default]
    Scipy,
    /// A peak is kept when no larger peak lies nearer than `distance` (SpikeInterface
    /// `locally_exclusive`). Depends only on peaks within `distance`, so chunked processing with
    /// `distance` of context on each side equals the whole-signal result.
    LocallyExclusive,
}

/// `min ≤ value ≤ max`, either side optional (scipy's `(min, max)` arguments).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Interval<T> {
    pub min: Option<T>,
    pub max: Option<T>,
}

impl<T> Default for Interval<T> {
    fn default() -> Self {
        Self { min: None, max: None }
    }
}

impl<T: Float> Interval<T> {
    /// At least `min`.
    pub fn at_least(min: T) -> Self {
        Self { min: Some(min), max: None }
    }

    fn is_set(&self) -> bool {
        self.min.is_some() || self.max.is_some()
    }

    fn contains(&self, v: T) -> bool {
        self.min.is_none_or(|m| m <= v) && self.max.is_none_or(|m| v <= m)
    }
}

/// `scipy.signal.relative_height` default of [`PeakOptions::rel_height`].
pub const DEFAULT_REL_HEIGHT: f64 = 0.5;

/// Conditions of [`find_peaks`], as `scipy.signal.find_peaks` (all off by default). Heights,
/// thresholds and prominences are of the signed signal (`x` for maxima, `−x` for minima).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PeakOptions<T> {
    /// Peak height.
    pub height: Interval<T>,
    /// Vertical distance to both neighbouring samples (`min` against the nearer, `max` against the
    /// farther, as scipy).
    pub threshold: Interval<T>,
    /// Minimal sample distance between peaks (≥ 1), resolved by `distance_rule`.
    pub distance: Option<usize>,
    pub distance_rule: DistanceRule,
    pub prominence: Interval<T>,
    /// Width (samples) at `rel_height` of the prominence.
    pub width: Interval<T>,
    /// Window (samples) the prominence searches; `None` = whole signal.
    pub wlen: Option<usize>,
    pub rel_height: T,
}

impl<T: Float> Default for PeakOptions<T> {
    fn default() -> Self {
        Self {
            height: Interval::default(),
            threshold: Interval::default(),
            distance: None,
            distance_rule: DistanceRule::default(),
            prominence: Interval::default(),
            width: Interval::default(),
            wlen: None,
            rel_height: T::from(DEFAULT_REL_HEIGHT).expect("0.5 is representable"),
        }
    }
}

/// Peaks found by [`find_peaks`], with the properties the conditions needed.
#[derive(Debug, Clone, PartialEq)]
pub struct Peaks<T> {
    /// Sample of each peak, ascending (the middle sample of a flat peak, rounded down).
    pub indices: Vec<usize>,
    /// `+1` for a maximum, `−1` for a minimum.
    pub signs: Vec<i8>,
    /// Prominence and its bases (when `prominence` or `width` was asked).
    pub prominences: Option<Vec<T>>,
    pub left_bases: Option<Vec<usize>>,
    pub right_bases: Option<Vec<usize>>,
    /// Width at `rel_height` and its interpolated ends (when `width` was asked).
    pub widths: Option<Vec<T>>,
    pub left_ips: Option<Vec<T>>,
    pub right_ips: Option<Vec<T>>,
}

impl<T> Default for Peaks<T> {
    fn default() -> Self {
        Self {
            indices: Vec::new(),
            signs: Vec::new(),
            prominences: None,
            left_bases: None,
            right_bases: None,
            widths: None,
            left_ips: None,
            right_ips: None,
        }
    }
}

/// Local extrema of `x` of `polarity` as `(index, sign)` ascending: maxima of `sign · x` (flat
/// peaks: middle sample, rounded down), as scipy `_local_maxima_1d`; edges never count.
pub fn local_extrema<T: Float>(x: &[T], polarity: Polarity) -> Vec<(usize, i8)> {
    let mut out: Vec<(usize, i8)> = Vec::new();
    for (sign, s) in signs_of(polarity) {
        out.extend(local_maxima(x, s).into_iter().map(|p| (p, sign)));
    }
    out.sort_by_key(|&(p, _)| p);
    out
}

fn signs_of<T: Float>(polarity: Polarity) -> Vec<(i8, T)> {
    match polarity {
        Polarity::Positive => vec![(1, T::one())],
        Polarity::Negative => vec![(-1, -T::one())],
        Polarity::Both => vec![(1, T::one()), (-1, -T::one())],
    }
}

fn local_maxima<T: Float>(x: &[T], sign: T) -> Vec<usize> {
    let s = |i: usize| sign * x[i];
    let mut out = Vec::new();
    let n = x.len();
    if n < 3 {
        return out;
    }
    let i_max = n - 1;
    let mut i = 1;
    while i < i_max {
        if s(i - 1) < s(i) {
            let mut ahead = i + 1;
            while ahead < i_max && s(ahead) == s(i) {
                ahead += 1;
            }
            if s(ahead) < s(i) {
                out.push((i + ahead - 1) / 2);
                i = ahead;
            }
        }
        i += 1;
    }
    out
}

/// Which of `peaks` (ascending samples) to keep so none is nearer than `distance` to a larger one
/// by `priority`, under `rule` (equal priorities: the earlier peak counts as larger). Returns a
/// keep mask.
pub fn select_by_distance<T: Float>(peaks: &[usize], priority: &[T], distance: usize, rule: DistanceRule) -> Vec<bool> {
    let n = peaks.len();
    // `a` outranks `b`
    let beats = |a: usize, b: usize| priority[a] > priority[b] || (priority[a] == priority[b] && a < b);
    if rule == DistanceRule::LocallyExclusive {
        return (0..n)
            .map(|j| {
                let left = (0..j).rev().take_while(|&k| peaks[j] - peaks[k] < distance);
                let right = (j + 1..n).take_while(|&k| peaks[k] - peaks[j] < distance);
                !left.chain(right).any(|k| beats(k, j))
            })
            .collect();
    }
    let mut keep = vec![true; n];
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| if beats(a, b) { std::cmp::Ordering::Less } else if beats(b, a) { std::cmp::Ordering::Greater } else { std::cmp::Ordering::Equal });
    for j in order {
        if !keep[j] {
            continue;
        }
        let mut k = j;
        while k > 0 && peaks[j] - peaks[k - 1] < distance {
            keep[k - 1] = false;
            k -= 1;
        }
        let mut k = j + 1;
        while k < n && peaks[k] - peaks[j] < distance {
            keep[k] = false;
            k += 1;
        }
    }
    keep
}

/// Prominence of the peak at `p` of `s` and its left / right bases (scipy `_peak_prominences`).
fn prominence<T: Float>(s: &dyn Fn(usize) -> T, n: usize, p: usize, wlen: Option<usize>) -> (T, usize, usize) {
    let (mut i_min, mut i_max) = (0, n - 1);
    if let Some(w) = wlen.filter(|&w| w >= 2) {
        i_min = p.saturating_sub(w / 2);
        i_max = (p + w / 2).min(n - 1);
    }
    let (mut left_base, mut left_min) = (p, s(p));
    let mut i = p as isize;
    while i >= i_min as isize && s(i as usize) <= s(p) {
        if s(i as usize) < left_min {
            left_min = s(i as usize);
            left_base = i as usize;
        }
        i -= 1;
    }
    let (mut right_base, mut right_min) = (p, s(p));
    let mut i = p;
    while i <= i_max && s(i) <= s(p) {
        if s(i) < right_min {
            right_min = s(i);
            right_base = i;
        }
        i += 1;
    }
    (s(p) - left_min.max(right_min), left_base, right_base)
}

/// Width of the peak at `p` at `rel_height` of its prominence, and the interpolated left / right
/// crossing positions (scipy `_peak_widths`).
fn width<T: Float>(s: &dyn Fn(usize) -> T, p: usize, prom: T, left_base: usize, right_base: usize, rel_height: T) -> (T, T, T) {
    let height = s(p) - prom * rel_height;
    let mut i = p;
    while left_base < i && height < s(i) {
        i -= 1;
    }
    let mut left = T::from(i).expect("index");
    if s(i) < height {
        left = left + (height - s(i)) / (s(i + 1) - s(i));
    }
    let mut i = p;
    while i < right_base && height < s(i) {
        i += 1;
    }
    let mut right = T::from(i).expect("index");
    if s(i) < height {
        right = right - (height - s(i)) / (s(i - 1) - s(i));
    }
    (right - left, left, right)
}

/// Peaks of `x` meeting `options`, as `scipy.signal.find_peaks` (`polarity` [`Polarity::Positive`])
/// — conditions applied in scipy's order: height, threshold, distance, prominence, width. For
/// [`Polarity::Negative`] the minima of `x` are searched as maxima of `−x`; for [`Polarity::Both`]
/// both, with `distance` ranking peaks by magnitude.
pub fn find_peaks<T: Float>(x: &[T], polarity: Polarity, options: &PeakOptions<T>) -> Peaks<T> {
    let n = x.len();
    // (index, sign) of every local extremum passing height and threshold, ascending
    let mut cand: Vec<(usize, T)> = Vec::new();
    for (_, sign) in signs_of::<T>(polarity) {
        for p in local_maxima(x, sign) {
            let v = sign * x[p];
            if !options.height.contains(v) {
                continue;
            }
            if options.threshold.is_set() {
                let (l, r) = (v - sign * x[p - 1], v - sign * x[p + 1]);
                if !(options.threshold.min.is_none_or(|m| m <= l.min(r)) && options.threshold.max.is_none_or(|m| l.max(r) <= m)) {
                    continue;
                }
            }
            cand.push((p, sign));
        }
    }
    cand.sort_by_key(|&(p, _)| p);

    if let Some(d) = options.distance.filter(|&d| d > 1) {
        let idx: Vec<usize> = cand.iter().map(|&(p, _)| p).collect();
        let priority: Vec<T> = cand.iter().map(|&(p, s)| s * x[p]).collect();
        let keep = select_by_distance(&idx, &priority, d, options.distance_rule);
        cand = cand.into_iter().zip(keep).filter_map(|(c, k)| k.then_some(c)).collect();
    }

    let mut out = Peaks::default();
    let need_prominence = options.prominence.is_set() || options.width.is_set();
    let mut props: Vec<(T, usize, usize, T, T, T)> = Vec::new();
    let mut kept: Vec<(usize, T)> = Vec::new();
    for (p, sign) in cand {
        let s = |i: usize| sign * x[i];
        let (mut prom, mut lb, mut rb) = (T::zero(), p, p);
        if need_prominence {
            (prom, lb, rb) = prominence(&s, n, p, options.wlen);
            if !options.prominence.contains(prom) {
                continue;
            }
        }
        let (mut w, mut li, mut ri) = (T::zero(), T::zero(), T::zero());
        if options.width.is_set() {
            (w, li, ri) = width(&s, p, prom, lb, rb, options.rel_height);
            if !options.width.contains(w) {
                continue;
            }
        }
        kept.push((p, sign));
        props.push((prom, lb, rb, w, li, ri));
    }
    out.indices = kept.iter().map(|&(p, _)| p).collect();
    out.signs = kept.iter().map(|&(_, s)| if s > T::zero() { 1 } else { -1 }).collect();
    if need_prominence {
        out.prominences = Some(props.iter().map(|p| p.0).collect());
        out.left_bases = Some(props.iter().map(|p| p.1).collect());
        out.right_bases = Some(props.iter().map(|p| p.2).collect());
    }
    if options.width.is_set() {
        out.widths = Some(props.iter().map(|p| p.3).collect());
        out.left_ips = Some(props.iter().map(|p| p.4).collect());
        out.right_ips = Some(props.iter().map(|p| p.5).collect());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn opts() -> PeakOptions<f64> {
        PeakOptions::default()
    }

    #[test]
    fn local_maxima_and_plateaus() {
        // scipy: find_peaks([0, 2, 0, 3, 3, 3, 0, 1, 1, 0]) -> [1, 4, 7]
        let x = [0.0, 2.0, 0.0, 3.0, 3.0, 3.0, 0.0, 1.0, 1.0, 0.0];
        assert_eq!(find_peaks(&x, Polarity::Positive, &opts()).indices, vec![1, 4, 7]);
        // Edges are never peaks
        assert!(find_peaks(&[5.0, 1.0, 5.0], Polarity::Positive, &opts()).indices.is_empty());
    }

    #[test]
    fn height_threshold_distance() {
        let x = [0.0, 5.0, 0.0, 3.0, 0.0, 4.0, 3.5, 0.0];
        let o = PeakOptions { height: Interval::at_least(3.5), ..opts() };
        assert_eq!(find_peaks(&x, Polarity::Positive, &o).indices, vec![1, 5]);
        let o = PeakOptions { threshold: Interval::at_least(1.0), ..opts() };
        assert_eq!(find_peaks(&x, Polarity::Positive, &o).indices, vec![1, 3], "peak 5 is only 0.5 above sample 6");
        // scipy: find_peaks(x, distance=3) -> [1, 5] (5 before 3)
        let o = PeakOptions { distance: Some(3), ..opts() };
        assert_eq!(find_peaks(&x, Polarity::Positive, &o).indices, vec![1, 5]);
    }

    #[test]
    fn distance_rules_differ_on_chains() {
        // 10 at 0 removes 9 at 2 (scipy); 8 at 4 then survives under scipy (its larger neighbour 9
        // was removed) but not locally exclusive (9 is larger and nearer than 3)
        let (peaks, priority) = ([0usize, 2, 4], [10.0, 9.0, 8.0]);
        assert_eq!(select_by_distance(&peaks, &priority, 3, DistanceRule::Scipy), vec![true, false, true]);
        assert_eq!(select_by_distance(&peaks, &priority, 3, DistanceRule::LocallyExclusive), vec![true, false, false]);
        // Ties: the earlier peak wins under both rules
        for rule in [DistanceRule::Scipy, DistanceRule::LocallyExclusive] {
            assert_eq!(select_by_distance(&[5usize, 6], &[1.0, 1.0], 2, rule), vec![true, false]);
        }
    }

    #[test]
    fn prominence_and_width_match_scipy() {
        // scipy: x = [0, 1, 0, 4, 1, 2, 0]; find_peaks(x, prominence=0) -> peaks [1, 3, 5],
        // prominences [1, 4, 1], left_bases [0, 2, 4], right_bases [2, 6, 6]
        let x = [0.0, 1.0, 0.0, 4.0, 1.0, 2.0, 0.0];
        let o = PeakOptions { prominence: Interval::at_least(0.0), width: Interval::at_least(0.0), ..opts() };
        let p = find_peaks(&x, Polarity::Positive, &o);
        assert_eq!(p.indices, vec![1, 3, 5]);
        assert_eq!(p.prominences.unwrap(), vec![1.0, 4.0, 1.0]);
        assert_eq!(p.left_bases.unwrap(), vec![0, 2, 4]);
        assert_eq!(p.right_bases.unwrap(), vec![2, 6, 6]);
        // peak_widths at 0.5: peak 3 crosses height 2 at 2.5 and 4 − 1/3 -> width 7/6
        let w = p.widths.unwrap();
        assert!((w[1] - 7.0 / 6.0).abs() < 1e-12, "{w:?}");
    }

    #[test]
    fn troughs_and_both_polarities() {
        let x = [0.0, -5.0, 0.0, 4.0, 0.0, -1.0, 0.0];
        let neg = find_peaks(&x, Polarity::Negative, &PeakOptions { height: Interval::at_least(2.0), ..opts() });
        assert_eq!((neg.indices, neg.signs), (vec![1], vec![-1]));
        let both = find_peaks(&x, Polarity::Both, &PeakOptions { height: Interval::at_least(2.0), ..opts() });
        assert_eq!((both.indices, both.signs), (vec![1, 3], vec![-1, 1]));
        // distance ranks by magnitude: the trough (5) beats the peak (4)
        let o = PeakOptions { height: Interval::at_least(2.0), distance: Some(3), ..opts() };
        assert_eq!(find_peaks(&x, Polarity::Both, &o).indices, vec![1]);
    }
}
