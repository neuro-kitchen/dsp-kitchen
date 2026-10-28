//! SpikeInterface's `locally_exclusive` peak detection on a device trace: per channel, samples
//! beyond `detect_threshold · noise` that are local extrema (strictly beyond the previous sample, at
//! least the next: [`dsp_base::peaks::find_peak_candidates`] on the device), then
//! [`super::locally_exclusive`] among channels within `radius_um` and `±exclude_sweep` samples.
//! `peak_sign` both: extrema of `|x|` (upstream tests the negative and positive conditions
//! separately; they differ only where a sample is a negative and a positive extremum at once).

use std::ops::Range;

use cubecl::prelude::*;
use cubecl::server::Handle;
use dsp_base::core::buffer;
use dsp_base::peaks::{find_peak_candidates, Polarity};

use super::exclusive::{locally_exclusive, Candidate};

/// Peak sign (upstream `peak_sign`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PeakSign {
    #[default]
    Neg,
    Pos,
    Both,
}

impl PeakSign {
    fn polarity(self) -> Polarity {
        match self {
            PeakSign::Neg => Polarity::Negative,
            PeakSign::Pos => Polarity::Positive,
            PeakSign::Both => Polarity::Both,
        }
    }
}

/// A detected peak.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Peak {
    /// Global sample.
    pub sample: u64,
    pub channel: u32,
    /// The trace's value there.
    pub amplitude: f32,
}

/// The detector of one probe and noise estimate.
pub struct LocallyExclusiveDetector {
    thresholds: Vec<f64>,
    heights: Handle,
    /// `[channels, channels]`: within `radius_um`.
    neighbours: Vec<bool>,
    channels: usize,
    sweep: u64,
    sign: PeakSign,
}

impl LocallyExclusiveDetector {
    /// `positions`: `(x, y)` µm per channel; `noise_levels`: per channel, the trace's unit;
    /// `exclude_sweep`: samples.
    pub fn new(client: &Client, positions: &[[f32; 2]], noise_levels: &[f64], detect_threshold: f64, radius_um: f32, exclude_sweep: usize, sign: PeakSign) -> Self {
        let channels = positions.len();
        assert_eq!(noise_levels.len(), channels, "one noise level per channel");
        let thresholds: Vec<f64> = noise_levels.iter().map(|n| n * detect_threshold).collect();
        let heights: Vec<f32> = thresholds.iter().map(|&t| t as f32).collect();
        let neighbours = (0..channels * channels)
            .map(|e| {
                let (a, b) = (positions[e / channels], positions[e % channels]);
                (a[0] - b[0]).hypot(a[1] - b[1]) <= radius_um
            })
            .collect();
        Self {
            heights: buffer::upload(client, if heights.is_empty() { &[0.0f32][..] } else { &heights }),
            thresholds,
            neighbours,
            channels,
            sweep: exclude_sweep as u64,
            sign,
        }
    }

    /// Margin each side of a window's own samples (upstream `get_margin`): `sweep + 1`.
    pub fn margin(&self) -> usize {
        self.sweep as usize + 1
    }

    /// Peaks of the `[channels, samples]` device trace whose local sample lies in `emit`
    /// (`global_offset`: global sample of local sample 0), in sample order. Candidates within
    /// [`Self::margin`] of `emit` compete too.
    pub fn detect(&self, client: &Client, trace: &Handle, samples: usize, emit: Range<usize>, global_offset: u64) -> Vec<Peak> {
        let m = self.channels;
        if m == 0 || emit.is_empty() {
            return Vec::new();
        }
        let scan = emit.start.saturating_sub(self.margin())..(emit.end + self.margin()).min(samples);
        let found = find_peak_candidates::<f32>(client, trace, &self.heights, m, samples, scan, self.sign.polarity());
        let mut cands: Vec<(Candidate, f32)> = Vec::new();
        for ch in 0..m {
            let (idx, vals) = found.channel(ch);
            for (&t, &v) in idx.iter().zip(vals) {
                let c = Candidate { sample: t as u64, row: ch as u32, score: (v as f64).abs() / self.thresholds[ch], tiebreak: 0 };
                cands.push((c, v));
            }
        }
        // Upstream's order: by sample, then channel
        cands.sort_by_key(|(c, _)| (c.sample, c.row));
        let list: Vec<Candidate> = cands.iter().map(|(c, _)| *c).collect();
        let keep = locally_exclusive(&list, self.sweep, |a, b| self.neighbours[a as usize * m + b as usize], false);
        cands
            .into_iter()
            .zip(keep)
            .filter(|((c, _), k)| *k && emit.contains(&(c.sample as usize)))
            .map(|((c, v), _)| Peak { sample: global_offset + c.sample, channel: c.row, amplitude: v })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dsp_core::compute::ComputeTarget;

    /// Two windows with margins give the same peaks as one buffer; peaks are exclusive.
    #[test]
    fn windows_agree_and_neighbours_exclude() {
        let Ok(target) = ComputeTarget::from_env() else { return };
        let client = target.client().expect("client");
        let (m, n) = (4usize, 4000usize);
        let pos: Vec<[f32; 2]> = (0..m).map(|c| [0.0, 30.0 * c as f32]).collect();
        let mut state = 5u64;
        let x: Vec<f32> = (0..m * n)
            .map(|_| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                (((state >> 40) as f32 / (1u64 << 24) as f32) - 0.5) * 10.0
            })
            .collect();
        let trace = buffer::upload(&client, &x);
        let det = LocallyExclusiveDetector::new(&client, &pos, &[1.0; 4], 3.0, 40.0, 15, PeakSign::Neg);
        let all = det.detect(&client, &trace, n, 100..n - 100, 0);
        let mut split = det.detect(&client, &trace, n, 100..2000, 0);
        split.extend(det.detect(&client, &trace, n, 2000..n - 100, 0));
        assert!(all.len() > 20);
        assert_eq!(all, split);
        for (i, a) in all.iter().enumerate() {
            assert!(a.amplitude <= -3.0);
            for b in &all[i + 1..] {
                let near = (a.channel as i32 - b.channel as i32).abs() <= 1;
                assert!(!(near && b.sample - a.sample <= 15), "{a:?} and {b:?} compete");
            }
        }
    }
}
