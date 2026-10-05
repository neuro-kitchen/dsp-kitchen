//! Kilosort4 universal-template spike detection on the device.
//!
//! 1. [`TemplateCentres`] (host, once per probe): a grid of virtual template centres per shank
//!    (`dmin / 2` vertically, `dminx / 2` horizontally), each with its `nearest_chans` contacts and
//!    Gaussian spatial weights of `template_sizes` widths (`min_template_size · (s + 1)`), L2-
//!    normalized; centres farther than `max_channel_distance` from every contact dropped; each
//!    centre's `nearest_templates` neighbouring centres.
//! 2. Per batch ([`detect_universal`]): correlation of every channel with every `wTEMP` row;
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
use dsp_base::peaks::{find_peak_candidates, Polarity};
use dsp_core::compute::LaunchGeometry;
use dsp_core::{DspError, DspResult};
use dsp_io::neuro::probe::SensorLayout;

use super::kernels::{
    centre_response_kernel, correlate_templates_kernel, gather_args_kernel, local_peak_score_kernel,
    neighbour_max_kernel, spike_features_kernel,
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
    /// `nearest_chans` enabled contacts.
    pub fn new(layout: &SensorLayout, opts: &CentreOptions) -> DspResult<Self> {
        let sites: Vec<(usize, f32, f32, usize)> =
            layout.contacts.iter().filter(|s| s.enabled).map(|s| (s.channel_id, s.position.x_um, s.position.y_um, s.shank_id)).collect();
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

/// Universal-template detection of one preprocessed `[channels, samples]` batch on the device
/// (`x`). See the module docs.
#[allow(clippy::too_many_arguments)]
pub fn detect_universal<R: Runtime>(
    client: &ComputeClient<R>,
    x: &Handle,
    channels: usize,
    samples: usize,
    centres: &TemplateCentres,
    templates: &UniversalTemplates,
    th_universal: f32,
    nt0min: usize,
) -> DspResult<Vec<UniversalSpike>> {
    let (nt, k) = (templates.nt, templates.n_templates);
    let n_centres = centres.n_centres();
    if n_centres == 0 || samples <= 2 * nt {
        return Ok(Vec::new());
    }
    if centres.ic.iter().any(|&c| c as usize >= channels) {
        return Err(DspError::InvalidConfig(format!("template centres use a channel outside the {channels}-channel batch")));
    }
    let wtemp = buffer::upload(client, &templates.wtemp);
    let b = buffer::empty::<R, f32>(client, channels * k * samples);
    let a_s = buffer::empty::<R, f32>(client, n_centres * samples);
    let arg = buffer::empty::<R, i32>(client, n_centres * samples);
    let a_max = buffer::empty::<R, f32>(client, n_centres * samples);
    let score = buffer::empty::<R, f32>(client, n_centres * samples);
    let ic = buffer::upload(client, &centres.ic);
    let ic2 = buffer::upload(client, &centres.ic2);
    let weights = buffer::upload(client, &centres.weights);
    let per_row = LaunchGeometry::channels_samples(client, channels * k, samples);
    let per_centre = LaunchGeometry::channels_samples(client, n_centres, samples);
    // SAFETY: every array is passed with the length it was created with
    unsafe {
        correlate_templates_kernel::launch::<f32, R>(
            client,
            per_row.cube_count,
            per_row.cube_dim,
            ArrayArg::from_raw_parts(x.clone(), channels * samples),
            ArrayArg::from_raw_parts(wtemp, k * nt),
            ArrayArg::from_raw_parts(b.clone(), channels * k * samples),
            channels as u32,
            samples as u32,
            k as u32,
            nt as u32,
        );
        centre_response_kernel::launch::<f32, R>(
            client,
            per_centre.cube_count.clone(),
            per_centre.cube_dim.clone(),
            ArrayArg::from_raw_parts(b.clone(), channels * k * samples),
            ArrayArg::from_raw_parts(weights, centres.weights.len()),
            ArrayArg::from_raw_parts(ic.clone(), centres.ic.len()),
            ArrayArg::from_raw_parts(a_s.clone(), n_centres * samples),
            ArrayArg::from_raw_parts(arg.clone(), n_centres * samples),
            samples as u32,
            k as u32,
            n_centres as u32,
            centres.n_chans as u32,
            centres.n_sizes as u32,
        );
        neighbour_max_kernel::launch::<f32, R>(
            client,
            per_centre.cube_count.clone(),
            per_centre.cube_dim.clone(),
            ArrayArg::from_raw_parts(a_s.clone(), n_centres * samples),
            ArrayArg::from_raw_parts(ic2, centres.ic2.len()),
            ArrayArg::from_raw_parts(a_max.clone(), n_centres * samples),
            samples as u32,
            n_centres as u32,
            centres.n_neighbours as u32,
            nt as u32,
        );
        local_peak_score_kernel::launch::<f32, R>(
            client,
            per_centre.cube_count,
            per_centre.cube_dim,
            ArrayArg::from_raw_parts(a_s.clone(), n_centres * samples),
            ArrayArg::from_raw_parts(a_max, n_centres * samples),
            ArrayArg::from_raw_parts(score.clone(), n_centres * samples),
            samples as u32,
            n_centres as u32,
            nt0min as u32,
        );
    }

    // Local maxima of the score above Th_universal, compacted on the device
    let heights = buffer::upload(client, &vec![th_universal; n_centres]);
    let cand = find_peak_candidates::<R, f32>(client, &score, &heights, n_centres, samples, 0..samples, Polarity::Positive);
    let mut spikes: Vec<(usize, usize, f32)> = Vec::new(); // (centre, t, amplitude)
    for centre in 0..n_centres {
        let (ts, vals) = cand.channel(centre);
        spikes.extend(ts.iter().zip(vals).map(|(&t, &v)| (centre, t as usize, v)));
    }
    if spikes.is_empty() {
        return Ok(Vec::new());
    }
    // Arg-max (template, size, sign) of each spike: gathered on the device, only those entries read
    let n = spikes.len();
    let centre_ids: Vec<u32> = spikes.iter().map(|s| s.0 as u32).collect();
    let times: Vec<u32> = spikes.iter().map(|s| s.1 as u32).collect();
    let (centres_h, times_h) = (buffer::upload(client, &centre_ids), buffer::upload(client, &times));
    let picked = buffer::empty::<R, i32>(client, n);
    let per_pick = LaunchGeometry::elementwise(client, n);
    // SAFETY: as above
    unsafe {
        gather_args_kernel::launch::<R>(
            client,
            per_pick.cube_count,
            per_pick.cube_dim,
            ArrayArg::from_raw_parts(arg, n_centres * samples),
            ArrayArg::from_raw_parts(centres_h.clone(), n),
            ArrayArg::from_raw_parts(times_h.clone(), n),
            ArrayArg::from_raw_parts(picked.clone(), n),
            samples as u32,
            n as u32,
        );
    }
    let decoded: Vec<(usize, usize, f32)> = buffer::download::<R, i32>(client, picked)
        .into_iter()
        .map(|a| {
            let idx = (a.unsigned_abs() as usize).saturating_sub(1);
            (idx % k, idx / k, if a < 0 { -1.0 } else { 1.0 })
        })
        .collect();

    // Features and per-channel template responses
    let (n_chans, n_pcs) = (centres.n_chans, templates.n_pcs);
    let feat = buffer::empty::<R, f32>(client, n * n_chans * n_pcs);
    let amp = buffer::empty::<R, f32>(client, n * n_chans);
    let per_spike = LaunchGeometry::elementwise(client, n * n_chans);
    let tmpl: Vec<u32> = decoded.iter().map(|d| d.0 as u32).collect();
    // SAFETY: as above
    unsafe {
        spike_features_kernel::launch::<f32, R>(
            client,
            per_spike.cube_count,
            per_spike.cube_dim,
            ArrayArg::from_raw_parts(x.clone(), channels * samples),
            ArrayArg::from_raw_parts(b, channels * k * samples),
            ArrayArg::from_raw_parts(buffer::upload(client, &templates.wpca), n_pcs * nt),
            ArrayArg::from_raw_parts(ic, centres.ic.len()),
            ArrayArg::from_raw_parts(centres_h, n),
            ArrayArg::from_raw_parts(times_h, n),
            ArrayArg::from_raw_parts(buffer::upload(client, &tmpl), n),
            ArrayArg::from_raw_parts(feat.clone(), n * n_chans * n_pcs),
            ArrayArg::from_raw_parts(amp.clone(), n * n_chans),
            samples as u32,
            k as u32,
            n_centres as u32,
            n_chans as u32,
            n_pcs as u32,
            nt as u32,
            n as u32,
        );
    }
    let feat = buffer::download::<R, f32>(client, feat);
    let amp = buffer::download::<R, f32>(client, amp);

    Ok(spikes
        .iter()
        .zip(&decoded)
        .enumerate()
        .map(|(i, (&(centre, t, amplitude), &(template, size, sign)))| {
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
                amplitude,
                template,
                size,
                y_um,
                features: feat[i * n_chans * n_pcs..(i + 1) * n_chans * n_pcs].to_vec(),
            }
        })
        .collect())
}
