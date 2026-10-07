//! Kilosort4 universal-template spike detection on the device.
//!
//! 1. [`TemplateCentres`] (host, once per probe): a grid of virtual template centres per shank
//!    (`dmin / 2` vertically, `dminx / 2` horizontally), each with its `nearest_chans` contacts and
//!    Gaussian spatial weights of `template_sizes` widths (`min_template_size · (s + 1)`), L2-
//!    normalized; centres farther than `max_channel_distance` from every contact dropped; each
//!    centre's `nearest_templates` neighbouring centres.
//! 2. Per batch ([`UniversalDetector::detect`]): correlation of every channel with every `wTEMP` row;
//!    weighted sum over each centre's channels, maximum magnitude over sizes and templates;
//!    maximum over neighbouring centres; local maxima over ±`nt0min` samples above
//!    `Th_universal`, compacted on the device (`dsp_base::peaks::find_peak_candidates`); features
//!    (`wPCA` projections) and the centre's vertical position for each spike.
//!
//! The data must be preprocessed as Kilosort4 expects (common reference, high-pass, whitening:
//! thresholds are in whitened σ).

use cubecl::prelude::*;
use cubecl::server::Handle;
use dsp_base::core::buffer;
use dsp_base::peaks::{find_peak_candidates_on_device, Polarity};
use dsp_core::compute::{device_elements, LaunchGeometry};
use dsp_core::{DspError, DspResult};
use dsp_io::neuro::probe::SensorLayout;

use super::kernels::{
    centre_response_kernel, correlate_templates_kernel, local_peak_score_kernel, neighbour_max_kernel,
    spike_features_kernel,
};
use super::templates::UniversalTemplates;

/// Spatial settings of the template centres (Kilosort4 defaults in [`Default`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CentreOptions {
    /// Vertical centre spacing (µm); `None`: median vertical contact spacing.
    pub dmin: Option<f32>,
    /// Horizontal centre spacing (µm).
    pub dminx: f32,
    pub max_channel_distance: f32,
    pub min_template_size: f32,
    pub template_sizes: usize,
    pub nearest_chans: usize,
    pub nearest_templates: usize,
}

impl Default for CentreOptions {
    fn default() -> Self {
        Self {
            dmin: None,
            dminx: 32.0,
            max_channel_distance: 32.0,
            min_template_size: 10.0,
            template_sizes: 5,
            nearest_chans: 10,
            nearest_templates: 100,
        }
    }
}

/// Template centres and their channel tables (see the module docs).
#[derive(Debug, Clone, PartialEq)]
pub struct TemplateCentres {
    /// `(x, y)` µm per centre.
    pub positions: Vec<[f32; 2]>,
    /// `iC[c, centre]` (`[n_chans, n_centres]`): recording channel of the centre's `c`-th contact.
    pub ic: Vec<u32>,
    /// `[n_sizes, n_chans, n_centres]` spatial weights.
    pub weights: Vec<f32>,
    /// `iC2[j, centre]` (`[n_neighbours, n_centres]`): neighbouring centres.
    pub ic2: Vec<u32>,
    pub n_chans: usize,
    pub n_sizes: usize,
    pub n_neighbours: usize,
    /// Vertical position (µm) of each recording channel.
    pub channel_y: Vec<f32>,
}

impl TemplateCentres {
    pub fn n_centres(&self) -> usize {
        self.positions.len()
    }

    /// Centres of `layout` (channels indexed as in the recording). Needs at least
    pub fn new(layout: &SensorLayout, opts: &CentreOptions) -> DspResult<Self> {
        let sites: Vec<(usize, f32, f32, usize)> = layout
            .contacts
            .iter()
            .enumerate()
            .filter(|(_, s)| s.enabled)
            .map(|(idx, s)| (idx, s.position.x_um, s.position.y_um, s.shank_id))
            .collect();
        let n_chans = opts.nearest_chans.min(sites.len());
        if n_chans == 0 {
            return Err(DspError::InvalidConfig("probe has no enabled contacts".into()));
        }
        let dmin = match opts.dmin {
            Some(d) => d,
            None => {
                let mut ys: Vec<f32> = sites.iter().map(|s| s.2).collect();
                ys.sort_by(f32::total_cmp);
                ys.dedup();
                let mut diffs: Vec<f32> = ys.windows(2).map(|w| w[1] - w[0]).collect();
                diffs.sort_by(f32::total_cmp);
                if diffs.is_empty() { 1.0 } else { diffs[diffs.len() / 2] }
            }
        };
        // Grid per shank
        let mut shanks: Vec<usize> = sites.iter().map(|s| s.3).collect();
        shanks.sort_unstable();
        shanks.dedup();
        let (mut xs_all, mut ys_all) = (Vec::new(), Vec::new());
        for sh in shanks {
            let on: Vec<&(usize, f32, f32, usize)> = sites.iter().filter(|s| s.3 == sh).collect();
            let (xmin, xmax) = on.iter().fold((f32::INFINITY, f32::NEG_INFINITY), |(a, b), s| (a.min(s.1), b.max(s.1)));
            let (ymin, ymax) = on.iter().fold((f32::INFINITY, f32::NEG_INFINITY), |(a, b), s| (a.min(s.2), b.max(s.2)));
            let mut y = ymin;
            while y <= ymax + 0.01 {
                ys_all.push(y);
                y += dmin / 2.0;
            }
            let nx = (((xmax - xmin) / (opts.dminx / 2.0)).round() as usize) + 1;
            for i in 0..nx {
                xs_all.push(if nx == 1 { xmin } else { xmin + (xmax - xmin) * i as f32 / (nx - 1) as f32 });
            }
        }
        let uniq = |mut v: Vec<f32>| {
            v.sort_by(f32::total_cmp);
            v.dedup_by(|a, b| (*a - *b).abs() < 1e-6);
            v
        };
        let (xs, ys) = (uniq(xs_all), uniq(ys_all));
        let grid: Vec<[f32; 2]> = xs.iter().flat_map(|&x| ys.iter().map(move |&y| [x, y])).collect();

        // Nearest contacts of each centre; drop centres too far from every contact
        let max_d2 = opts.max_channel_distance.powi(2);
        let mut positions = Vec::new();
        let mut near: Vec<Vec<(usize, f32)>> = Vec::new();
        for g in grid {
            let mut d: Vec<(usize, f32)> = sites.iter().map(|s| (s.0, (s.1 - g[0]).powi(2) + (s.2 - g[1]).powi(2))).collect();
            d.sort_by(|a, b| a.1.total_cmp(&b.1));
            d.truncate(n_chans);
            if d[0].1 <= max_d2 {
                positions.push(g);
                near.push(d);
            }
        }
        let n_centres = positions.len();
        let mut ic = vec![0u32; n_chans * n_centres];
        let mut weights = vec![0.0f32; opts.template_sizes * n_chans * n_centres];
        for (k, d) in near.iter().enumerate() {
            for (c, &(ch, _)) in d.iter().enumerate() {
                ic[c * n_centres + k] = ch as u32;
            }
            for s in 0..opts.template_sizes {
                let sigma = opts.min_template_size * (s + 1) as f32;
                let w: Vec<f32> = d.iter().map(|&(_, d2)| (-d2 / (sigma * sigma)).exp()).collect();
                let norm = w.iter().map(|v| v * v).sum::<f32>().sqrt().max(f32::MIN_POSITIVE);
                for (c, v) in w.iter().enumerate() {
                    weights[(s * n_chans + c) * n_centres + k] = v / norm;
                }
            }
        }
        let n_neighbours = opts.nearest_templates.min(n_centres);
        let mut ic2 = vec![0u32; n_neighbours * n_centres];
        for k in 0..n_centres {
            let mut d: Vec<(usize, f32)> =
                (0..n_centres).map(|j| (j, (positions[j][0] - positions[k][0]).powi(2) + (positions[j][1] - positions[k][1]).powi(2))).collect();
            d.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
            for (j, &(other, _)) in d.iter().take(n_neighbours).enumerate() {
                ic2[j * n_centres + k] = other as u32;
            }
        }
        let n_channels = sites.iter().map(|s| s.0 + 1).max().unwrap_or(0);
        let mut channel_y = vec![0.0f32; n_channels];
        for s in &sites {
            channel_y[s.0] = s.2;
        }
        Ok(Self { positions, ic, weights, ic2, n_chans, n_sizes: opts.template_sizes, n_neighbours, channel_y })
    }
}

/// One detected spike.
#[derive(Debug, Clone, PartialEq)]
pub struct UniversalSpike {
    /// Sample in the batch.
    pub sample: usize,
    pub centre: usize,
    /// Response magnitude (whitened units).
    pub amplitude: f32,
    /// Universal template that matched best.
    pub template: usize,
    /// Template size index that matched best.
    pub size: usize,
    /// Response-weighted vertical position (µm).
    pub y_um: f32,
    /// `[n_chans, n_pcs]` features on the centre's channels.
    pub features: Vec<f32>,
}

/// Universal-template detection on the device, set up once for a probe and a batch size:
/// templates, centre tables and thresholds are uploaded once and the scratch buffers kept, so a
/// batch costs its kernels plus two reads (the candidate counts, then the spikes).
pub struct UniversalDetector {
    client: Client,
    channels: usize,
    max_samples: usize,
    nt: usize,
    n_templates: usize,
    n_pcs: usize,
    nt0min: usize,
    centres: TemplateCentres,
    wtemp: Handle,
    wpca: Handle,
    ic: Handle,
    ic2: Handle,
    weights: Handle,
    /// `Th_universal` per centre.
    heights: Handle,
    /// `[channels, n_templates, max_samples]` correlations.
    b: Handle,
    /// `[n_centres, max_samples]` centre responses, arg-max, neighbourhood maximum, score.
    a_s: Handle,
    arg: Handle,
    a_max: Handle,
    score: Handle,
}

impl UniversalDetector {
    /// Detector for `[channels, samples ≤ max_samples]` batches.
    pub fn new(
        client: &Client,
        channels: usize,
        max_samples: usize,
        centres: &TemplateCentres,
        templates: &UniversalTemplates,
        th_universal: f32,
        nt0min: usize,
    ) -> DspResult<Self> {
        if centres.ic.iter().any(|&c| c as usize >= channels) {
            return Err(DspError::InvalidConfig(format!("template centres use a channel outside the {channels}-channel batch")));
        }
        let (k, n_centres) = (templates.n_templates, centres.n_centres());
        // The correlation buffer is the largest: channels × templates × samples
        device_elements("detector correlations [channels, templates, samples]", &[channels, k, max_samples])?;
        let per_centre = device_elements("centre responses [centres, samples]", &[n_centres, max_samples])?;
        Ok(Self {
            client: client.clone(),
            channels,
            max_samples,
            nt: templates.nt,
            n_templates: k,
            n_pcs: templates.n_pcs,
            nt0min,
            centres: centres.clone(),
            wtemp: buffer::upload(client, &templates.wtemp),
            wpca: buffer::upload(client, &templates.wpca),
            ic: buffer::upload(client, &centres.ic),
            ic2: buffer::upload(client, &centres.ic2),
            weights: buffer::upload(client, &centres.weights),
            heights: buffer::upload(client, &vec![th_universal; n_centres.max(1)]),
            b: buffer::empty::<f32>(client, channels * k * max_samples),
            a_s: buffer::empty::<f32>(client, per_centre),
            arg: buffer::empty::<i32>(client, per_centre),
            a_max: buffer::empty::<f32>(client, per_centre),
            score: buffer::empty::<f32>(client, per_centre),
        })
    }

    /// Spikes of one preprocessed `[channels, samples]` device batch `x` (see the module docs),
    /// ordered by centre then sample.
    pub fn detect(&mut self, x: &Handle, samples: usize) -> DspResult<Vec<UniversalSpike>> {
        if samples > self.max_samples {
            return Err(DspError::InvalidConfig(format!("batch of {samples} samples exceeds the detector's {}", self.max_samples)));
        }
        let (client, channels) = (&self.client, self.channels);
        let (nt, k, n_pcs) = (self.nt, self.n_templates, self.n_pcs);
        let centres = &self.centres;
        let (n_centres, n_chans) = (centres.n_centres(), centres.n_chans);
        if n_centres == 0 || samples <= 2 * nt {
            return Ok(Vec::new());
        }
        let per_row = LaunchGeometry::channels_samples(client, channels * k, samples);
        let per_centre = LaunchGeometry::channels_samples(client, n_centres, samples);
        let (b_len, c_len) = (channels * k * samples, n_centres * samples);
        // SAFETY: every array is passed with at most the length it was created with
        unsafe {
            correlate_templates_kernel::launch::<f32>(
                client,
                per_row.cube_count,
                per_row.cube_dim,
                BufferArg::from_raw_parts(x.clone(), channels * samples),
                BufferArg::from_raw_parts(self.wtemp.clone(), k * nt),
                BufferArg::from_raw_parts(self.b.clone(), b_len),
                channels as u32,
                samples as u32,
                k as u32,
                nt as u32,
            );
            centre_response_kernel::launch::<f32>(
                client,
                per_centre.cube_count.clone(),
                per_centre.cube_dim.clone(),
                BufferArg::from_raw_parts(self.b.clone(), b_len),
                BufferArg::from_raw_parts(self.weights.clone(), centres.weights.len()),
                BufferArg::from_raw_parts(self.ic.clone(), centres.ic.len()),
                BufferArg::from_raw_parts(self.a_s.clone(), c_len),
                BufferArg::from_raw_parts(self.arg.clone(), c_len),
                samples as u32,
                k as u32,
                n_centres as u32,
                n_chans as u32,
                centres.n_sizes as u32,
            );
            neighbour_max_kernel::launch::<f32>(
                client,
                per_centre.cube_count.clone(),
                per_centre.cube_dim.clone(),
                BufferArg::from_raw_parts(self.a_s.clone(), c_len),
                BufferArg::from_raw_parts(self.ic2.clone(), centres.ic2.len()),
                BufferArg::from_raw_parts(self.a_max.clone(), c_len),
                samples as u32,
                n_centres as u32,
                centres.n_neighbours as u32,
                nt as u32,
            );
            local_peak_score_kernel::launch::<f32>(
                client,
                per_centre.cube_count,
                per_centre.cube_dim,
                BufferArg::from_raw_parts(self.a_s.clone(), c_len),
                BufferArg::from_raw_parts(self.a_max.clone(), c_len),
                BufferArg::from_raw_parts(self.score.clone(), c_len),
                samples as u32,
                n_centres as u32,
                self.nt0min as u32,
            );
        }

        // Local maxima of the score above Th_universal, compacted and kept on the device (read 1:
        // the counts per centre)
        let cand = find_peak_candidates_on_device::<f32>(client, &self.score, &self.heights, n_centres, samples, 0..samples, Polarity::Positive);
        let n = cand.total;
        if n == 0 {
            return Ok(Vec::new());
        }
        // Arg-max, features and per-channel template responses of every candidate, on the device
        let picked = buffer::empty::<i32>(client, n);
        let feat = buffer::empty::<f32>(client, n * n_chans * n_pcs);
        let amp = buffer::empty::<f32>(client, n * n_chans);
        let per_spike = LaunchGeometry::elementwise(client, n * n_chans);
        // SAFETY: as above; the candidate buffers hold `n` entries
        unsafe {
            spike_features_kernel::launch::<f32>(
                client,
                per_spike.cube_count,
                per_spike.cube_dim,
                BufferArg::from_raw_parts(x.clone(), channels * samples),
                BufferArg::from_raw_parts(self.b.clone(), b_len),
                BufferArg::from_raw_parts(self.wpca.clone(), n_pcs * nt),
                BufferArg::from_raw_parts(self.ic.clone(), centres.ic.len()),
                BufferArg::from_raw_parts(self.arg.clone(), c_len),
                BufferArg::from_raw_parts(cand.rows.clone(), n),
                BufferArg::from_raw_parts(cand.indices.clone(), n),
                BufferArg::from_raw_parts(picked.clone(), n),
                BufferArg::from_raw_parts(feat.clone(), n * n_chans * n_pcs),
                BufferArg::from_raw_parts(amp.clone(), n * n_chans),
                samples as u32,
                k as u32,
                n_centres as u32,
                n_chans as u32,
                n_pcs as u32,
                nt as u32,
                n as u32,
            );
        }
        // Read 2: the spikes (all kernels are queued; these reads wait for them once)
        let rows = buffer::download::<u32>(client, cand.rows);
        let times = buffer::download::<u32>(client, cand.indices);
        let values = buffer::download::<f32>(client, cand.values);
        let picked = buffer::download::<i32>(client, picked);
        let feat = buffer::download::<f32>(client, feat);
        let amp = buffer::download::<f32>(client, amp);

        let mut spikes: Vec<UniversalSpike> = (0..n)
            .map(|i| {
                let (centre, t, a) = (rows[i] as usize, times[i] as usize, picked[i]);
                let idx = (a.unsigned_abs() as usize).saturating_sub(1);
                let (template, size, sign) = (idx % k, idx / k, if a < 0 { -1.0 } else { 1.0 });
                // y: contact y weighted by the (sign-corrected, rectified) template response
                let w: Vec<f32> = (0..n_chans).map(|c| (amp[i * n_chans + c] * sign).max(0.0)).collect();
                let total: f32 = w.iter().sum();
                let y_um = if total > 0.0 {
                    (0..n_chans).map(|c| w[c] * centres.channel_y[centres.ic[c * n_centres + centre] as usize]).sum::<f32>() / total
                } else {
                    centres.positions[centre][1]
                };
                UniversalSpike {
                    sample: t,
                    centre,
                    amplitude: values[i],
                    template,
                    size,
                    y_um,
                    features: feat[i * n_chans * n_pcs..(i + 1) * n_chans * n_pcs].to_vec(),
                }
            })
            .collect();
        // The device lists are unordered within a centre
        spikes.sort_unstable_by_key(|s| (s.centre, s.sample));
        Ok(spikes)
    }
}

/// Universal-template detection of one preprocessed `[channels, samples]` batch on the device
/// (`x`), with a one-off [`UniversalDetector`]. Over many batches, keep a detector instead.
#[allow(clippy::too_many_arguments)]
pub fn detect_universal(
    client: &Client,
    x: &Handle,
    channels: usize,
    samples: usize,
    centres: &TemplateCentres,
    templates: &UniversalTemplates,
    th_universal: f32,
    nt0min: usize,
) -> DspResult<Vec<UniversalSpike>> {
    UniversalDetector::new(client, channels, samples, centres, templates, th_universal, nt0min)?.detect(x, samples)
}
