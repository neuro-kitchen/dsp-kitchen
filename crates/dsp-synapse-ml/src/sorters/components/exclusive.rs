//! SpikeInterface's locally exclusive peak selection (`sortingcomponents/peak_detection`, MIT),
//! shared by its detectors: among candidate peaks sorted by sample, a peak is dropped when a
//! candidate on a neighbouring row within `±sweep` samples has a larger score (`|value| /
//! threshold`), or an equal score and an earlier sample (or, with `tiebreak`, an equal score, the
//! same sample and a smaller tie-break value: matched filtering's depth index).
//!
//! Upstream drops every peak within `sweep + 1` samples of its chunk's ends (they belong to the
//! chunk's margin); callers keep those peaks as competitors and report only the ones they own.

/// A candidate peak (module docs).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Candidate {
    pub sample: u64,
    /// Channel, or template row (matched filtering).
    pub row: u32,
    /// `|value| / threshold`.
    pub score: f64,
    /// Matched filtering's depth index (0 when unused).
    pub tiebreak: u32,
}

/// Which candidates survive (module docs). `cands` must be sorted by sample; `neighbours(a, b)`:
/// whether rows `a` and `b` compete.
pub fn locally_exclusive(cands: &[Candidate], sweep: u64, neighbours: impl Fn(u32, u32) -> bool, use_tiebreak: bool) -> Vec<bool> {
    debug_assert!(cands.windows(2).all(|w| w[0].sample <= w[1].sample), "candidates sorted by sample");
    let n = cands.len();
    let mut keep = vec![true; n];
    let mut start = 0usize;
    for i in 0..n {
        let ci = cands[i];
        while start < n && cands[start].sample + sweep < ci.sample {
            start += 1;
        }
        let mut j = start;
        while j < n && cands[j].sample <= ci.sample + sweep {
            if j != i && neighbours(ci.row, cands[j].row) {
                let cj = cands[j];
                let beats = cj.score > ci.score
                    || (cj.score == ci.score && ci.sample > cj.sample)
                    || (use_tiebreak && cj.score == ci.score && ci.sample == cj.sample && ci.tiebreak > cj.tiebreak);
                if beats {
                    keep[i] = false;
                    break;
                }
            }
            j += 1;
        }
    }
    keep
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Upstream `detect_peaks_numba_locally_exclusive_on_chunk` (negative peaks), as written.
    fn upstream(traces: &[f32], n: usize, m: usize, thr: &[f64], sweep: usize, mask: &[bool]) -> Vec<(usize, usize)> {
        let at = |s: usize, c: usize| traces[s * m + c];
        let mut peaks = Vec::new();
        for s in 1..n - 1 {
            for c in 0..m {
                if (at(s, c) as f64) <= -thr[c] && at(s, c) < at(s - 1, c) && at(s, c) <= at(s + 1, c) {
                    peaks.push((s, c));
                }
            }
        }
        let np = peaks.len();
        let mut keep = vec![true; np];
        let mut next_start = 0;
        for i in 0..np {
            let (si, ci) = peaks[i];
            if si < sweep + 1 || si >= n - sweep - 1 {
                keep[i] = false;
                continue;
            }
            for j in next_start..np {
                if i == j {
                    continue;
                }
                let (sj, cj) = peaks[j];
                if si + sweep < sj {
                    break;
                }
                if si > sj + sweep {
                    next_start = j;
                    continue;
                }
                if mask[ci * m + cj] && si.abs_diff(sj) <= sweep {
                    let vi = (at(si, ci) as f64).abs() / thr[ci];
                    let vj = (at(sj, cj) as f64).abs() / thr[cj];
                    if vj > vi || (vj == vi && si > sj) {
                        keep[i] = false;
                        break;
                    }
                }
            }
        }
        peaks.into_iter().zip(keep).filter(|(_, k)| *k).map(|(p, _)| p).collect()
    }

    #[test]
    fn matches_upstream_rule() {
        let (n, m, sweep) = (3000usize, 6usize, 10usize);
        let mut state = 99u64;
        let traces: Vec<f32> = (0..n * m)
            .map(|_| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                (((state >> 40) as f32 / (1u64 << 24) as f32) - 0.5) * 12.0
            })
            .collect();
        let thr: Vec<f64> = (0..m).map(|c| 3.0 + 0.3 * c as f64).collect();
        let mask: Vec<bool> = (0..m * m).map(|e| (e / m).abs_diff(e % m) <= 1).collect();
        let want = upstream(&traces, n, m, &thr, sweep, &mask);
        // Ours: the same candidates (as the device peak finder gives them), sorted by sample
        let at = |s: usize, c: usize| traces[s * m + c];
        let mut cands = Vec::new();
        for s in 1..n - 1 {
            for c in 0..m {
                if (at(s, c) as f64) <= -thr[c] && at(s, c) < at(s - 1, c) && at(s, c) <= at(s + 1, c) {
                    cands.push(Candidate { sample: s as u64, row: c as u32, score: (at(s, c) as f64).abs() / thr[c], tiebreak: 0 });
                }
            }
        }
        let keep = locally_exclusive(&cands, sweep as u64, |a, b| mask[a as usize * m + b as usize], false);
        let got: Vec<(usize, usize)> = cands
            .iter()
            .zip(keep)
            .filter(|(c, k)| *k && (c.sample as usize) >= sweep + 1 && (c.sample as usize) < n - sweep - 1)
            .map(|(c, _)| (c.sample as usize, c.row as usize))
            .collect();
        assert!(want.len() > 50, "{} peaks", want.len());
        assert_eq!(got, want);
    }
}
