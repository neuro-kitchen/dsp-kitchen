//! Units most like a lead unit, by template.

use super::sorting::SortingData;

/// Samples of shift tried when comparing templates.
pub const SIMILARITY_LAG: usize = 5;

/// A unit's mean template as dspk-synapse compares templates: every channel its stored template
/// has signal on (else its extent channels).
fn waveform_template(sorting: &SortingData, u: usize) -> Option<crate::WaveformTemplate> {
    let unit = &sorting.units[u];
    if let (Some(t), Some(row)) = (&sorting.templates, unit.template_row) {
        let r = t.row(row);
        let channels: Vec<usize> = (0..t.channels).filter(|&c| r[c * t.samples..(c + 1) * t.samples].iter().any(|v| *v != 0.0)).collect();
        if !channels.is_empty() {
            let mean: Vec<f32> = channels.iter().flat_map(|&c| r[c * t.samples..(c + 1) * t.samples].iter().copied()).collect();
            let n = mean.len();
            return Some(crate::WaveformTemplate::new(channels, t.samples, mean, vec![0.0; n]));
        }
    }
    let samples = unit.template.first()?.len();
    let mean: Vec<f32> = unit.template.iter().flatten().copied().collect();
    Some(crate::WaveformTemplate::new(unit.extent.channels.clone(), samples, mean, vec![0.0; unit.template.len() * samples]))
}

/// Units most like `lead` by template: dspk-synapse's lag-maximised cosine similarity (rows
/// paired by channel, best shift within ±[`SIMILARITY_LAG`] samples), best first, at most `max`.
pub fn similar(sorting: &SortingData, lead: usize, max: usize) -> Vec<(usize, f32)> {
    let Some(a) = waveform_template(sorting, lead) else { return Vec::new() };
    let mut out: Vec<(usize, f32)> = (0..sorting.units.len())
        .filter(|&u| u != lead && sorting.units[u].merged_into.is_none())
        .filter_map(|u| {
            let b = waveform_template(sorting, u)?;
            let (sim, _) = crate::sorting::similarity::template_max_cosine_similarity(&a, &b, SIMILARITY_LAG);
            (sim != 0.0).then_some((u, sim))
        })
        .collect();
    out.sort_by(|x, y| y.1.total_cmp(&x.1));
    out.truncate(max);
    out
}
