use dsp_base::math::peak_to_peak;
use dsp_base::math::RunningMoments;
use dsp_io::neuro::templates::{DenseTemplates, TemplateAxisOrder};
use serde::{Deserialize, Serialize};
use super::snippets::WaveformSnippet;

/// Automated or curated unit quality classification (Allen / IBL / Phy standard).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, Default)]
pub enum UnitQualityLabel {
    /// Well-isolated single biological neuron (`"good"` in Phy)
    #[serde(alias = "Good", alias = "good", alias = "singleunit")]
    SingleUnit,
    /// Multi-unit activity or overlapping cluster (`"mua"` in Phy)
    #[serde(alias = "Mua", alias = "mua", alias = "multiunit")]
    MultiUnit,
    /// Non-biological electrical/motion artifact or thermal noise (`"noise"` in Phy)
    #[serde(alias = "noise")]
    Noise,
    /// Uncurated / unsorted cluster (`"unsorted"` in Phy)
    #[default]
    #[serde(alias = "unsorted")]
    Unsorted,
}

#[allow(non_upper_case_globals)]
impl UnitQualityLabel {
    /// Alias matching Phy's `"good"` group.
    pub const Good: Self = Self::SingleUnit;
    /// Alias matching Phy's `"mua"` group.
    pub const Mua: Self = Self::MultiUnit;

    pub fn as_str(self) -> &'static str {
        match self {
            Self::SingleUnit => "good",
            Self::MultiUnit => "mua",
            Self::Noise => "noise",
            Self::Unsorted => "unsorted",
        }
    }

    pub fn parse(s: &str) -> Self {
        match s.trim().to_ascii_lowercase().as_str() {
            "good" | "singleunit" | "sua" => Self::SingleUnit,
            "mua" | "multiunit" => Self::MultiUnit,
            "noise" => Self::Noise,
            _ => Self::Unsorted,
        }
    }
}

/// Mean action potential template, standard deviation (`std`), and standard error of the mean (`se`)
/// across a cluster of waveforms.
///
/// Row `r` of `mean` / `std` / `se` (each `num_samples` long) belongs to recording channel
/// `channel_ids[r]`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WaveformTemplate {
    /// Recording channel of each row.
    pub channel_ids: Vec<usize>,
    pub num_channels: usize,
    pub num_samples: usize,
    /// Number of waveforms accumulated into this template (`>= 1`).
    pub count: usize,
    /// Sample of the deepest trough (minimum of `mean` over all rows).
    pub trough_index: usize,
    pub mean: Vec<f32>,
    pub std: Vec<f32>,
    /// Standard Error of the Mean (`SE = SD / sqrt(count)`).
    pub se: Vec<f32>,
}

impl WaveformTemplate {
    /// Builds a template; `mean` and `std` are `[channel_ids.len(), num_samples]`.
    pub fn new(channel_ids: Vec<usize>, num_samples: usize, mean: Vec<f32>, std: Vec<f32>) -> Self {
        Self::with_count(channel_ids, num_samples, 1, mean, std)
    }

    /// Builds a template with explicit waveform `count`, automatically computing `se = std / sqrt(count)`.
    pub fn with_count(
        channel_ids: Vec<usize>,
        num_samples: usize,
        count: usize,
        mean: Vec<f32>,
        std: Vec<f32>,
    ) -> Self {
        let num_channels = channel_ids.len();
        assert_eq!(mean.len(), num_channels * num_samples, "mean shape");
        assert_eq!(std.len(), num_channels * num_samples, "std shape");
        let trough_index = mean
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.total_cmp(b.1))
            .map_or(0, |(i, _)| i % num_samples.max(1));
        let denom = (count.max(1) as f32).sqrt();
        let se = std.iter().map(|&s| s / denom).collect();
        Self {
            channel_ids,
            num_channels,
            num_samples,
            count: count.max(1),
            trough_index,
            mean,
            std,
            se,
        }
    }

    /// Mean waveform of row `r`.
    pub fn row(&self, r: usize) -> &[f32] {
        &self.mean[r * self.num_samples..(r + 1) * self.num_samples]
    }

    /// Standard deviation waveform of row `r`.
    pub fn std_row(&self, r: usize) -> &[f32] {
        &self.std[r * self.num_samples..(r + 1) * self.num_samples]
    }

    /// Standard error waveform of row `r`.
    pub fn se_row(&self, r: usize) -> &[f32] {
        &self.se[r * self.num_samples..(r + 1) * self.num_samples]
    }

    /// Mean waveform on recording channel `channel`, if the template covers it.
    pub fn channel_row(&self, channel: usize) -> Option<&[f32]> {
        self.channel_ids.iter().position(|&c| c == channel).map(|r| self.row(r))
    }

    /// Recording channel with the largest peak-to-peak swing in `mean`.
    pub fn best_channel(&self) -> Option<usize> {
        (0..self.num_channels)
            .max_by(|&a, &b| peak_to_peak(self.row(a)).total_cmp(&peak_to_peak(self.row(b))))
            .and_then(|r| self.channel_ids.get(r).copied())
    }

    /// Largest peak-to-peak swing across all channels in `mean`.
    pub fn amplitude(&self) -> f32 {
        (0..self.num_channels).map(|r| peak_to_peak(self.row(r))).fold(0.0, f32::max)
    }
}

// Dense template arrays (`DenseTemplates`, `TemplateAxisOrder`) are the file layout: dsp-io.

/// Packs `(slot, template)` pairs into dense `(mean, std, se)` buffers in `order` (Phy uses the
/// unit id as slot, Zarr / NWB the table row), on at least `num_channels` channels.
pub fn pack_templates<'a>(
    units: impl IntoIterator<Item = (usize, Option<&'a WaveformTemplate>)>,
    num_channels: usize,
    order: TemplateAxisOrder,
) -> Option<([usize; 3], Vec<f32>, Vec<f32>, Vec<f32>)> {
    let list: Vec<(usize, Option<&WaveformTemplate>)> = units.into_iter().collect();
    let count = list.iter().map(|&(slot, _)| slot + 1).max().unwrap_or(0);
    let samples = list.iter().filter_map(|(_, t)| t.map(|w| w.num_samples)).max().unwrap_or(0);
    let channels = num_channels.max(
        list.iter()
            .filter_map(|(_, t)| *t)
            .flat_map(|t| t.channel_ids.iter().copied())
            .max()
            .map_or(0, |c| c + 1),
    );
    if count == 0 || samples == 0 || channels == 0 {
        return None;
    }
    let total = count * samples * channels;
    let mut mean = vec![0.0f32; total];
    let mut std = vec![0.0f32; total];
    let mut se = vec![0.0f32; total];

    for (slot, tpl) in list {
        let Some(t) = tpl else { continue };
        for (r, &ch) in t.channel_ids.iter().enumerate() {
            if ch < channels {
                let n_s = t.num_samples.min(samples);
                let m_row = t.row(r);
                let sd_row = t.std_row(r);
                let se_row = t.se_row(r);
                for s in 0..n_s {
                    let idx = match order {
                        TemplateAxisOrder::SamplesChannels => (slot * samples + s) * channels + ch,
                        TemplateAxisOrder::ChannelsSamples => (slot * channels + ch) * samples + s,
                    };
                    mean[idx] = m_row[s];
                    std[idx] = sd_row[s];
                    se[idx] = se_row[s];
                }
            }
        }
    }
    let shape = match order {
        TemplateAxisOrder::SamplesChannels => [count, samples, channels],
        TemplateAxisOrder::ChannelsSamples => [count, channels, samples],
    };
    Some((shape, mean, std, se))
}

/// Unpacks slot `slot` from dense `(mean, std, se)` buffers in `order` into a [`WaveformTemplate`].
pub fn unpack_template(
    slot: usize,
    shape: [usize; 3],
    order: TemplateAxisOrder,
    mean_buf: &[f32],
    std_buf: Option<&[f32]>,
    se_buf: Option<&[f32]>,
    spike_count: usize,
) -> Option<WaveformTemplate> {
    let (count, channels, samples) = match order {
        TemplateAxisOrder::SamplesChannels => (shape[0], shape[2], shape[1]),
        TemplateAxisOrder::ChannelsSamples => (shape[0], shape[1], shape[2]),
    };
    if slot >= count || channels == 0 || samples == 0 {
        return None;
    }
    let at = |buf: &[f32], c: usize, s: usize| match order {
        TemplateAxisOrder::SamplesChannels => buf[(slot * samples + s) * channels + c],
        TemplateAxisOrder::ChannelsSamples => buf[(slot * channels + c) * samples + s],
    };
    let n = channels * samples;
    let mut mean = Vec::with_capacity(n);
    for c in 0..channels {
        for s in 0..samples {
            mean.push(at(mean_buf, c, s));
        }
    }
    let std = match std_buf {
        Some(sd) if sd.len() == mean_buf.len() => {
            let mut v = Vec::with_capacity(n);
            for c in 0..channels {
                for s in 0..samples {
                    v.push(at(sd, c, s));
                }
            }
            v
        }
        _ => vec![1.0; n],
    };
    let mut w = WaveformTemplate::with_count((0..channels).collect(), samples, spike_count.max(1), mean, std);
    if let Some(se) = se_buf.filter(|s| s.len() == mean_buf.len()) {
        let mut v = Vec::with_capacity(n);
        for c in 0..channels {
            for s in 0..samples {
                v.push(at(se, c, s));
            }
        }
        w.se = v;
    }
    Some(w)
}

/// Template `t` of `dense` as a [`WaveformTemplate`] on the channels where it is not zero.
pub fn dense_waveform(dense: &DenseTemplates, t: usize) -> WaveformTemplate {
    let channels: Vec<usize> = (0..dense.channels).filter(|&c| dense.trace(t, c).iter().any(|v| *v != 0.0)).collect();
    let mean: Vec<f32> = channels.iter().flat_map(|&c| dense.trace(t, c)).collect();
    let std = vec![0.0; mean.len()];
    WaveformTemplate::new(channels, dense.samples, mean, std)
}

/// Degrees-of-freedom correction of template standard deviations: 0 (population SD, numpy's
/// default), as Phy / SpikeInterface `templates_std`.
pub const TEMPLATE_STD_DDOF: u64 = 0;

/// Computes the mean waveform template, standard deviation ([`TEMPLATE_STD_DDOF`]) and standard
/// error (`SE = SD / sqrt(n)`) across a set of aligned waveforms.
/// Snippets on other channels (or with another shape) than the first one are skipped.
pub fn compute_mean_template(snippets: &[WaveformSnippet]) -> Option<WaveformTemplate> {
    let first = snippets.first()?;
    let n_s = first.num_samples;
    let total_elements = first.num_channels() * n_s;
    let used: Vec<&WaveformSnippet> = snippets
        .iter()
        .filter(|s| s.waveform.len() == total_elements && s.channel_ids == first.channel_ids)
        .collect();
    if used.is_empty() {
        return None;
    }
    let mut moments = RunningMoments::new(total_elements);
    used.iter().for_each(|s| moments.push(&s.waveform));
    let mean = moments.mean().iter().map(|&m| m as f32).collect();
    let std = moments.std(TEMPLATE_STD_DDOF).into_iter().map(|s| s as f32).collect();
    Some(WaveformTemplate::with_count(first.channel_ids.clone(), n_s, used.len(), mean, std))
}

