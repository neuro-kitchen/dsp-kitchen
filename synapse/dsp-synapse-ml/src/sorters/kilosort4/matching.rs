//! Learned-template matching with matching pursuit (Kilosort4 paper, Methods: *Spike detection with
//! learned templates and matching pursuit*, *Extracting PC features with background subtraction*;
//! written from its description). Per preprocessed window, on the device:
//!
//! 1. **Scores.** Every channel is correlated with the `wPCA` rows (`[channels, n_pcs, samples]`),
//!    then one `matmul` with the unit-norm templates in PC form (`[templates, channels · n_pcs]`) gives
//!    every template's projection `c_j(t) = ŵ_jᵀ D(t)` at every sample. Templates are factorized over
//!    channels and PCs, so this is the paper's low-rank convolution.
//! 2. **Matching pursuit**, at most `max_peels` rounds: per sample, the template explaining the most
//!    variance at its average norm `μ_j`, `V = 2·μ_j·c_j − μ_j²`; spikes where `V` is the maximum over
//!    ±`nt` samples and `c ≥ th_learned`; each spike's contribution `c·ŵ` is subtracted from the data
//!    and, through the precomputed template products at every lag (`ctc`), from the scores, so no
//!    score is computed again. Spikes of one round are over `nt` apart; grouped by
//!    `⌊t/(nt+1)⌋ mod 3`, the spikes of a group are over `2·nt` apart, so their subtractions never
//!    touch the same samples: three launches per round, no atomics, no sorting.
//! 3. **Features** of each spike: the residual's PC projections on the channels of the centre nearest
//!    its template, plus the spike's own template contribution added back (the paper's background
//!    subtraction), so overlapping spikes do not leak into each other's features.
//!
//! Choices of ours where the paper is silent: a template's average norm is its mean waveform's norm;
//! spikes are reported at the waveform trough (as detection does) and placed at their template's
//! position (centre of mass of its channel energies).
//!
//! Data movement per window: the window is on the device already; per round the spike count, then
//! the round's spikes (time, template, amplitude); at the end their features. Everything else stays.

use cubecl::prelude::*;
use cubecl::server::Handle;
use dsp_base::core::buffer;
use dsp_base::linalg::{matmul, MatrixView};
use dsp_base::peaks::{find_peak_candidates_on_device, Polarity};
use dsp_core::compute::LaunchGeometry;
use dsp_core::{DspError, DspResult};

use super::detect::{TemplateCentres, UniversalSpike};
use super::kernels::detect::correlate_templates_kernel;
use super::kernels::matching::{
    best_template_kernel, copy_kernel, gather_features_kernel, gather_spikes_kernel, mark_peaks_kernel, subtract_data_kernel,
    subtract_scores_kernel, template_products_kernel,
};
use super::learned::LearnedTemplates;
use super::templates::UniversalTemplates;

/// Matching pursuit rounds per window (the paper's).
pub const MAX_PEELS: usize = 50;

/// `wtw[p, q, l] = Σ_t wpca[p, t] · wpca[q, t + lag]`, `lag = l − (nt − 1)`.
pub(crate) fn lagged_pc_products(wpca: &[f32], np: usize, nt: usize) -> Vec<f32> {
    let lags = 2 * nt - 1;
    let mut out = vec![0.0f32; np * np * lags];
    for p in 0..np {
        for q in 0..np {
            for l in 0..lags {
                let lag = l as isize - (nt as isize - 1);
                out[(p * np + q) * lags + l] = (0..nt as isize)
                    .filter(|&t| t + lag >= 0 && t + lag < nt as isize)
                    .map(|t| wpca[p * nt + t as usize] as f64 * wpca[q * nt + (t + lag) as usize] as f64)
                    .sum::<f64>() as f32;
            }
        }
    }
    out
}

/// Learned-template matching on the device, set up once per run (see the module docs).
pub struct TemplateMatcher {
    client: Client,
    n: usize,
    channels: usize,
    nt: usize,
    np: usize,
    nt0min: usize,
    max_samples: usize,
    th: f32,
    max_peels: usize,
    /// Average norm of each template; unit-norm templates in PC form (`[n, channels · np]`) and as
    /// waveforms (`[n, channels, nt]`); products at every lag (`[n, n, 2·nt − 1]`); `wPCA`.
    mu: Handle,
    u: Handle,
    w: Handle,
    ctc: Handle,
    wpca: Handle,
    /// Per template: its centre (nearest to its position), position, and that centre's channels.
    centre: Vec<usize>,
    position: Vec<[f32; 2]>,
    chans: Vec<u32>,
    n_chans: usize,
    /// Scratch: residual, PC projections, scores, per-sample best, peak marks.
    residual: Handle,
    b: Handle,
    s: Handle,
    vmax: Handle,
    best: Handle,
    amp: Handle,
    peak: Handle,
}

impl TemplateMatcher {
    pub fn new(
        client: &Client,
        learned: &LearnedTemplates,
        universal: &UniversalTemplates,
        centres: &TemplateCentres,
        max_samples: usize,
        th_learned: f32,
        nt0min: usize,
        max_peels: usize,
    ) -> DspResult<Self> {
        let (n, channels, np, nt) = (learned.n, learned.channels, learned.n_pcs, universal.nt);
        if n == 0 {
            return Err(DspError::InvalidConfig("template matching needs at least one learned template".into()));
        }
        let wpca = &universal.wpca;
        // Unit-norm templates: PC features / norm (wPCA rows are orthonormal, so the waveform norm is
        // the features' norm)
        let mut mu = vec![0.0f32; n];
        let mut u = vec![0.0f32; n * channels * np];
        let mut w = vec![0.0f32; n * channels * nt];
        let (n_centres, n_chans) = (centres.n_centres(), centres.n_chans);
        let (mut centre, mut position, mut chans) = (vec![0usize; n], vec![[0.0f32; 2]; n], vec![0u32; n * n_chans]);
        for j in 0..n {
            let f = &learned.features[j * channels * np..(j + 1) * channels * np];
            let norm = f.iter().map(|v| (*v as f64).powi(2)).sum::<f64>().sqrt();
            mu[j] = norm as f32;
            let scale = if norm > 0.0 { 1.0 / norm } else { 0.0 };
            for (dst, &v) in u[j * channels * np..(j + 1) * channels * np].iter_mut().zip(f) {
                *dst = (v as f64 * scale) as f32;
            }
            // Waveform, and the energy per channel for the template's position
            let (mut ex, mut ey, mut e) = (0.0f64, 0.0f64, 0.0f64);
            for ch in 0..channels {
                let mut energy = 0.0f64;
                for tau in 0..nt {
                    let v: f64 = (0..np).map(|p| u[(j * channels + ch) * np + p] as f64 * wpca[p * nt + tau] as f64).sum();
                    w[(j * channels + ch) * nt + tau] = v as f32;
                    energy += v * v;
                }
                if ch < centres.channel_x.len() {
                    ex += energy * centres.channel_x[ch] as f64;
                    ey += energy * centres.channel_y[ch] as f64;
                    e += energy;
                }
            }
            let pos = if e > 0.0 { [(ex / e) as f32, (ey / e) as f32] } else { [0.0, 0.0] };
            position[j] = pos;
            centre[j] = (0..n_centres)
                .min_by(|&a, &b| {
                    let d = |k: usize| (centres.positions[k][0] - pos[0]).powi(2) + (centres.positions[k][1] - pos[1]).powi(2);
                    d(a).total_cmp(&d(b))
                })
                .unwrap_or(0);
            for c in 0..n_chans {
                chans[j * n_chans + c] = centres.ic[c * n_centres + centre[j]];
            }
        }
        // Template products at every lag: (U·Uᵀ) combined with the PCs' lagged products
        let lags = 2 * nt - 1;
        let rows = n * np;
        let mut x = vec![0.0f32; rows * channels];
        for j in 0..n {
            for ch in 0..channels {
                for p in 0..np {
                    x[(j * np + p) * channels + ch] = u[(j * channels + ch) * np + p];
                }
            }
        }
        let x = buffer::upload(client, &x);
        let utu = buffer::empty::<f32>(client, rows * rows);
        let view = MatrixView::row_major(&x, rows * channels, rows, channels);
        matmul::<f32>(client, &view, &view.transposed(), &utu, rows * rows);
        let wtw = lagged_pc_products(wpca, np, nt);
        let ctc = buffer::empty::<f32>(client, n * n * lags);
        let geom = LaunchGeometry::elementwise(client, n * n * lags);
        // SAFETY: `utu` holds `rows²`, `wtw` `np² · lags`, `ctc` `n² · lags` values
        unsafe {
            template_products_kernel::launch::<f32>(
                client,
                geom.cube_count,
                geom.cube_dim,
                BufferArg::from_raw_parts(utu, rows * rows),
                BufferArg::from_raw_parts(buffer::upload(client, &wtw), wtw.len()),
                BufferArg::from_raw_parts(ctc.clone(), n * n * lags),
                n as u32,
                np as u32,
                lags as u32,
            );
        }
        dsp_core::compute::device_elements("matching scores [templates, samples]", &[n, max_samples])?;
        Ok(Self {
            client: client.clone(),
            n,
            channels,
            nt,
            np,
            nt0min,
            max_samples,
            th: th_learned,
            max_peels,
            mu: buffer::upload(client, &mu),
            u: buffer::upload(client, &u),
            w: buffer::upload(client, &w),
            ctc,
            wpca: buffer::upload(client, wpca),
            centre,
            position,
            chans,
            n_chans,
            residual: buffer::empty::<f32>(client, channels * max_samples),
            b: buffer::empty::<f32>(client, channels * np * max_samples),
            s: buffer::empty::<f32>(client, n * max_samples),
            vmax: buffer::empty::<f32>(client, max_samples),
            best: buffer::empty::<u32>(client, max_samples),
            amp: buffer::empty::<f32>(client, max_samples),
            peak: buffer::empty::<f32>(client, max_samples),
        })
    }

    /// PC projections of the residual (`b`, `[channels, np, samples]`).
    fn project(&self, samples: usize) {
        let client = &self.client;
        let geom = LaunchGeometry::channels_samples(client, self.channels, samples);
        // SAFETY: `residual` holds `channels · samples`, `wpca` `np · nt`, `b` `channels · np · samples`
        unsafe {
            correlate_templates_kernel::launch::<f32>(
                client,
                geom.cube_count,
                geom.cube_dim.clone(),
                BufferArg::from_raw_parts(self.residual.clone(), self.channels * samples),
                BufferArg::from_raw_parts(self.wpca.clone(), self.np * self.nt),
                BufferArg::from_raw_parts(self.b.clone(), self.channels * self.np * samples),
                self.channels as u32,
                samples as u32,
                self.np as u32,
                self.nt as u32,
                geom.cube_dim.x,
                geom.cube_dim.y,
            );
        }
    }

    /// The scores `c_j(t)` of the current residual (`s`, `[n, samples]`).
    pub fn scores(&self, samples: usize) {
        self.project(samples);
        let k = self.channels * self.np;
        let lhs = MatrixView::row_major(&self.u, self.n * k, self.n, k);
        let rhs = MatrixView::row_major(&self.b, k * samples, k, samples);
        matmul::<f32>(&self.client, &lhs, &rhs, &self.s, self.n * samples);
    }

    /// Spikes of one preprocessed `[channels, samples]` window `x` (left unchanged) whose times fall
    /// in `scan` (local samples; kept `nt` away from the window's ends), as [`UniversalSpike`]s whose
    /// `template` is the learned template and `amplitude` its projection `c` (see the module docs).
    pub fn match_window(&mut self, x: &Handle, samples: usize, scan: std::ops::Range<usize>) -> DspResult<Vec<UniversalSpike>> {
        if samples > self.max_samples {
            return Err(DspError::InvalidConfig(format!("window of {samples} samples exceeds the matcher's {}", self.max_samples)));
        }
        let (client, n, nt) = (self.client.clone(), self.n, self.nt);
        let lags = 2 * nt - 1;
        let (lo, hi) = (scan.start.max(nt), scan.end.min(samples.saturating_sub(nt)));
        if lo >= hi {
            return Ok(Vec::new());
        }
        let len = self.channels * samples;
        let geom = LaunchGeometry::elementwise(&client, len);
        // SAFETY: `x` and `residual` hold at least `channels · samples` values
        unsafe {
            copy_kernel::launch::<f32>(&client, geom.cube_count, geom.cube_dim, BufferArg::from_raw_parts(x.clone(), len), BufferArg::from_raw_parts(self.residual.clone(), len), len as u32);
        }
        self.scores(samples);

        let per_sample = LaunchGeometry::elementwise(&client, samples);
        let heights = buffer::upload(&client, &[f32::MIN_POSITIVE]);
        let (mut times, mut best, mut amp) = (Vec::new(), Vec::new(), Vec::new());
        let mut round_handles = Vec::new();
        for _ in 0..self.max_peels {
            // SAFETY: `s` holds `n · samples`, the per-sample buffers `samples` values
            unsafe {
                best_template_kernel::launch::<f32>(
                    &client,
                    per_sample.cube_count.clone(),
                    per_sample.cube_dim.clone(),
                    BufferArg::from_raw_parts(self.s.clone(), n * samples),
                    BufferArg::from_raw_parts(self.mu.clone(), n),
                    BufferArg::from_raw_parts(self.vmax.clone(), samples),
                    BufferArg::from_raw_parts(self.best.clone(), samples),
                    BufferArg::from_raw_parts(self.amp.clone(), samples),
                    n as u32,
                    samples as u32,
                );
                mark_peaks_kernel::launch::<f32>(
                    &client,
                    per_sample.cube_count.clone(),
                    per_sample.cube_dim.clone(),
                    BufferArg::from_raw_parts(self.vmax.clone(), samples),
                    BufferArg::from_raw_parts(self.amp.clone(), samples),
                    BufferArg::from_raw_parts(self.peak.clone(), samples),
                    samples as u32,
                    nt as u32,
                    lo as u32,
                    hi as u32,
                    self.th,
                );
            }
            // Compacted on the device (marks are isolated, so each is a local maximum); the count
            // comes back to size the launches
            let cand = find_peak_candidates_on_device::<f32>(&client, &self.peak, &heights, 1, samples, lo..hi, Polarity::Positive);
            let k = cand.total;
            if k == 0 {
                break;
            }
            let (out_best, out_amp) = (buffer::empty::<u32>(&client, k), buffer::empty::<f32>(&client, k));
            let per_spike = LaunchGeometry::elementwise(&client, k);
            let per_score = LaunchGeometry::elementwise(&client, k * n * lags);
            let per_data = LaunchGeometry::elementwise(&client, k * self.channels * nt);
            // SAFETY: `cand.indices` holds `k` local samples (< samples); other sizes as above
            unsafe {
                gather_spikes_kernel::launch::<f32>(
                    &client,
                    per_spike.cube_count,
                    per_spike.cube_dim,
                    BufferArg::from_raw_parts(cand.indices.clone(), k),
                    BufferArg::from_raw_parts(self.best.clone(), samples),
                    BufferArg::from_raw_parts(self.amp.clone(), samples),
                    BufferArg::from_raw_parts(out_best.clone(), k),
                    BufferArg::from_raw_parts(out_amp.clone(), k),
                    k as u32,
                );
                for class in 0..3u32 {
                    subtract_scores_kernel::launch::<f32>(
                        &client,
                        per_score.cube_count.clone(),
                        per_score.cube_dim.clone(),
                        BufferArg::from_raw_parts(self.s.clone(), n * samples),
                        BufferArg::from_raw_parts(self.ctc.clone(), n * n * lags),
                        BufferArg::from_raw_parts(cand.indices.clone(), k),
                        BufferArg::from_raw_parts(self.best.clone(), samples),
                        BufferArg::from_raw_parts(self.amp.clone(), samples),
                        k as u32,
                        n as u32,
                        samples as u32,
                        lags as u32,
                        nt as u32,
                        class,
                    );
                    subtract_data_kernel::launch::<f32>(
                        &client,
                        per_data.cube_count.clone(),
                        per_data.cube_dim.clone(),
                        BufferArg::from_raw_parts(self.residual.clone(), len),
                        BufferArg::from_raw_parts(self.w.clone(), n * self.channels * nt),
                        BufferArg::from_raw_parts(cand.indices.clone(), k),
                        BufferArg::from_raw_parts(self.best.clone(), samples),
                        BufferArg::from_raw_parts(self.amp.clone(), samples),
                        k as u32,
                        self.channels as u32,
                        samples as u32,
                        nt as u32,
                        nt as u32,
                        class,
                    );
                }
            }
            round_handles.push((cand.indices, out_best, out_amp, k));
        }
        // All rounds' spikes, read once the pursuit is done
        for (idx, b, a, k) in &round_handles {
            times.extend(buffer::download_prefix::<u32>(&client, idx.clone(), *k));
            best.extend(buffer::download_prefix::<u32>(&client, b.clone(), *k));
            amp.extend(buffer::download_prefix::<f32>(&client, a.clone(), *k));
        }
        let total = times.len();
        if total == 0 {
            return Ok(Vec::new());
        }

        // Features: residual PCs at the spike + its own template's contribution
        self.project(samples);
        let (nc, np) = (self.n_chans, self.np);
        let chans: Vec<u32> = best.iter().flat_map(|&j| self.chans[j as usize * nc..(j as usize + 1) * nc].iter().copied()).collect();
        let feat = buffer::empty::<f32>(&client, total * nc * np);
        let geom = LaunchGeometry::elementwise(&client, total * nc * np);
        // SAFETY: `b` holds `channels · np · samples`, `u` `n · channels · np`, the spike arrays `total`,
        // `chans` `total · nc` (all < channels), `feat` `total · nc · np` values
        unsafe {
            gather_features_kernel::launch::<f32>(
                &client,
                geom.cube_count,
                geom.cube_dim,
                BufferArg::from_raw_parts(self.b.clone(), self.channels * np * samples),
                BufferArg::from_raw_parts(self.u.clone(), n * self.channels * np),
                BufferArg::from_raw_parts(buffer::upload(&client, &times), total),
                BufferArg::from_raw_parts(buffer::upload(&client, &best), total),
                BufferArg::from_raw_parts(buffer::upload(&client, &amp), total),
                BufferArg::from_raw_parts(buffer::upload(&client, &chans), total * nc),
                BufferArg::from_raw_parts(feat.clone(), total * nc * np),
                total as u32,
                nc as u32,
                np as u32,
                self.channels as u32,
                samples as u32,
            );
        }
        let feat = buffer::download_prefix::<f32>(&client, feat, total * nc * np);
        let half = nt / 2;
        let mut spikes: Vec<UniversalSpike> = (0..total)
            .filter_map(|k| {
                let j = best[k] as usize;
                // Score time (window centre) → waveform trough, as detection reports it
                let sample = (times[k] as usize + self.nt0min).checked_sub(half)?;
                Some(UniversalSpike {
                    sample,
                    centre: self.centre[j],
                    amplitude: amp[k],
                    template: j,
                    size: 0,
                    x_um: self.position[j][0],
                    y_um: self.position[j][1],
                    features: feat[k * nc * np..(k + 1) * nc * np].to_vec(),
                })
            })
            .collect();
        spikes.sort_unstable_by_key(|s| (s.sample, s.template));
        Ok(spikes)
    }

    /// The current scores (`[n, samples]`), for tests.
    #[cfg(test)]
    fn read_scores(&self, samples: usize) -> Vec<f32> {
        buffer::download_prefix::<f32>(&self.client, self.s.clone(), self.n * samples)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dsp_core::compute::ComputeTarget;
    use dsp_io::neuro::probe::SensorLayout;

    /// A small probe, PCs (smooth bumps, orthonormalized) and two learned templates.
    fn setup(client: &Client) -> (TemplateCentres, UniversalTemplates, LearnedTemplates) {
        let channels = 8usize;
        let ids: Vec<usize> = (0..channels).collect();
        let positions: Vec<[f32; 2]> = (0..channels).map(|c| [(c % 2) as f32 * 32.0, (c / 2) as f32 * 20.0]).collect();
        let probe = SensorLayout::from_channel_arrays("8ch", &ids, &positions, &vec![0; channels]).expect("probe");
        let centres = TemplateCentres::new(&probe, &super::super::detect::CentreOptions::default()).expect("centres");
        let (np, nt) = (3usize, 21usize);
        // Gram–Schmidt of three bumps
        let mut wpca = vec![0.0f64; np * nt];
        for p in 0..np {
            for t in 0..nt {
                wpca[p * nt + t] = (-(((t as f64 - 7.0 - 3.0 * p as f64) / 2.5).powi(2))).exp();
            }
            for q in 0..p {
                let dot: f64 = (0..nt).map(|t| wpca[p * nt + t] * wpca[q * nt + t]).sum();
                for t in 0..nt {
                    wpca[p * nt + t] -= dot * wpca[q * nt + t];
                }
            }
            let norm = (0..nt).map(|t| wpca[p * nt + t].powi(2)).sum::<f64>().sqrt();
            (0..nt).for_each(|t| wpca[p * nt + t] /= norm);
        }
        let wpca: Vec<f32> = wpca.into_iter().map(|v| v as f32).collect();
        let universal = UniversalTemplates { nt, n_pcs: np, n_templates: 1, wtemp: wpca[..nt].to_vec(), wpca };
        // Template 0 on channels 0–3, template 1 on channels 4–7 (different PC mixtures)
        let mut features = vec![0.0f32; 2 * channels * np];
        for ch in 0..4 {
            features[ch * np] = 10.0 - ch as f32;
            features[ch * np + 1] = 2.0;
            features[(channels + ch + 4) * np + 2] = 9.0 - ch as f32;
            features[(channels + ch + 4) * np] = -3.0;
        }
        let _ = client;
        (centres, universal, LearnedTemplates { n: 2, channels, n_pcs: np, features, counts: vec![100, 100], merged: 0 })
    }

    /// Two overlapping spikes of different templates and one isolated spike are all found, with their
    /// templates and amplitudes; after peeling, the incrementally updated scores equal the scores of
    /// the residual computed from scratch (the lag convention of the score update).
    #[test]
    fn pursuit_finds_overlaps_and_updates_scores_exactly() {
        let Ok(target) = ComputeTarget::from_env() else { return };
        let client = target.client().expect("client");
        let (centres, universal, learned) = setup(&client);
        let (channels, nt, samples) = (learned.channels, universal.nt, 600usize);
        let mut matcher = TemplateMatcher::new(&client, &learned, &universal, &centres, samples, 5.0, 7, MAX_PEELS).expect("matcher");
        // Plant: template 0 at score time 200 (amp 40), template 1 at 205 (amp 25), template 0 at 400 (amp 30)
        let w = buffer::download_prefix::<f32>(&client, matcher.w.clone(), 2 * channels * nt);
        let mut x = vec![0.0f32; channels * samples];
        for &(t, j, a) in &[(200usize, 0usize, 40.0f32), (205, 1, 25.0), (400, 0, 30.0)] {
            for ch in 0..channels {
                for tau in 0..nt {
                    x[ch * samples + t - nt / 2 + tau] += a * w[(j * channels + ch) * nt + tau];
                }
            }
        }
        let handle = buffer::upload(&client, &x);
        let spikes = matcher.match_window(&handle, samples, 0..samples).expect("match");
        let got: Vec<(usize, usize, i32)> = spikes.iter().map(|s| (s.sample + nt / 2 - 7, s.template, s.amplitude.round() as i32)).collect();
        assert_eq!(got, vec![(200, 0, 40), (205, 1, 25), (400, 0, 30)], "{spikes:?}");
        // Incremental scores vs scores of the residual
        let updated = matcher.read_scores(samples);
        matcher.scores(samples);
        let fresh = matcher.read_scores(samples);
        let worst = updated.iter().zip(&fresh).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(worst < 1e-3, "score update differs from recomputation by {worst}");
        // Features: the planted spike's own template, scaled (the residual is ~0)
        let f = &spikes[0].features;
        assert!(f.iter().any(|v| v.abs() > 1.0), "features carry the template");
    }
}
