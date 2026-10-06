use dsp_core::RecordingSource;
use dsp_io::neuro::probe::{find_k_nearest_neighbors, SensorLayout};
use crate::core::{DeduplicatedSpike, SnippetBatch, SpikeEvent, WaveformSnippet};
use dsp_base::math::parabolic_vertex_offset;
use dsp_base::resampler::fractional_delay;

use super::alignment::SINC_KERNEL_RADIUS;

/// Samples a snippet `[center − pre, center + post)` needs beyond its window on each side: the
/// sinc taps when realigning, else the ±1 neighbours of the parabolic trough fit.
pub fn extraction_margin(apply_sinc_shift: bool) -> usize {
    if apply_sinc_shift {
        SINC_KERNEL_RADIUS.max(1)
    } else {
        1
    }
}

/// Whether a snippet around `center` (plus [`extraction_margin`]) lies inside `0..samples`.
/// Extractors skip spikes that do not fit.
pub fn snippet_fits(
    center: usize,
    pre: usize,
    post: usize,
    samples: usize,
    apply_sinc_shift: bool,
) -> bool {
    let m = extraction_margin(apply_sinc_shift);
    center >= pre + m && center + post + m <= samples
}

/// Parabolic sub-sample trough offset `δ` on `row` at `center` (trough at `center + δ`).
pub(crate) fn trough_offset(row: &[f32], center: usize) -> f32 {
    parabolic_vertex_offset(row[center - 1], row[center], row[center + 1])
}

/// Copies `[center − pre, center − pre + out.len())` of `row` into `out`, realigned by `+δ` so the
/// trough at `center + δ` lands on `out[pre]` when `shift` is given.
pub(crate) fn cut_row(
    row: &[f32],
    center: usize,
    pre: usize,
    shift: Option<f32>,
    out: &mut [f32],
) {
    let start = center - pre;
    match shift {
        Some(delta) => fractional_delay(row, start, delta, SINC_KERNEL_RADIUS, out),
        None => out.copy_from_slice(&row[start..start + out.len()]),
    }
}

/// Extracts a contiguous [`SnippetBatch`] directly from multi-channel raw data without intermediate
/// per-spike heap allocations.
#[allow(clippy::too_many_arguments)]
pub fn extract_snippet_batch_multichannel(
    data: &[f32],
    _channels: usize,
    samples: usize,
    spikes: &[DeduplicatedSpike],
    layout: &SensorLayout,
    k_neighbors: usize,
    pre_samples: usize,
    post_samples: usize,
    apply_sinc_shift: bool,
) -> SnippetBatch {
    let snippet_len = pre_samples + post_samples;
    let total_ch = layout.total_channels();
    let k = k_neighbors.min(total_ch).max(1);
    let stride = k * snippet_len;
    let knn_per_ch: Vec<Vec<usize>> = (0..total_ch)
        .map(|ch| find_k_nearest_neighbors(layout, ch, k))
        .collect();

    let mut flat_data = Vec::with_capacity(spikes.len() * stride);
    let mut primary_channels = Vec::with_capacity(spikes.len());
    let mut center_samples = Vec::with_capacity(spikes.len());
    let mut subsample_offsets = Vec::with_capacity(spikes.len());
    let mut channel_ids = Vec::with_capacity(spikes.len() * k);

    for spike in spikes {
        let center = spike.sample_index as usize;
        let primary_ch = spike.primary_channel;

        if !snippet_fits(center, pre_samples, post_samples, samples, apply_sinc_shift) {
            continue;
        }

        let neighbor_channels = knn_per_ch
            .get(primary_ch)
            .cloned()
            .unwrap_or_else(|| find_k_nearest_neighbors(layout, primary_ch, k));
        if neighbor_channels.len() != k {
            continue;
        }

        let sub_offset = trough_offset(
            &data[primary_ch * samples..(primary_ch + 1) * samples],
            center,
        );
        let shift = apply_sinc_shift.then_some(sub_offset);

        let dest = flat_data.len();
        flat_data.resize(dest + stride, 0.0);
        for (row_idx, &ch) in neighbor_channels.iter().enumerate() {
            let row = &data[ch * samples..(ch + 1) * samples];
            let out =
                &mut flat_data[dest + row_idx * snippet_len..dest + (row_idx + 1) * snippet_len];
            cut_row(row, center, pre_samples, shift, out);
        }

        primary_channels.push(primary_ch);
        center_samples.push(spike.sample_index);
        subsample_offsets.push(sub_offset);
        channel_ids.extend_from_slice(&neighbor_channels);
    }

    let num_spikes = primary_channels.len();
    SnippetBatch {
        data: flat_data,
        num_spikes,
        num_channels: k,
        num_samples: snippet_len,
        peak_index: pre_samples,
        primary_channels,
        center_samples,
        subsample_offsets,
        channel_ids,
    }
}

/// Extracts multi-channel [`WaveformSnippet`]s across K-nearest neighbors for deduplicated spike events.
#[allow(clippy::too_many_arguments)]
pub fn extract_snippets_multichannel(
    data: &[f32],
    channels: usize,
    samples: usize,
    spikes: &[DeduplicatedSpike],
    layout: &SensorLayout,
    k_neighbors: usize,
    pre_samples: usize,
    post_samples: usize,
    apply_sinc_shift: bool,
) -> Vec<WaveformSnippet> {
    extract_snippet_batch_multichannel(
        data,
        channels,
        samples,
        spikes,
        layout,
        k_neighbors,
        pre_samples,
        post_samples,
        apply_sinc_shift,
    )
    .to_snippets()
}

/// Single-channel snippet extraction from raw `SpikeEvent`s.
pub fn extract_snippets_single_channel(
    data: &[f32],
    _channels: usize,
    samples: usize,
    spikes: &[SpikeEvent],
    pre_samples: usize,
    post_samples: usize,
) -> Vec<WaveformSnippet> {
    let snippet_len = pre_samples + post_samples;
    let mut snippets = Vec::with_capacity(spikes.len());

    for spike in spikes {
        let center = spike.sample_index as usize;
        let ch = spike.channel_id;

        if !snippet_fits(center, pre_samples, post_samples, samples, false) {
            continue;
        }

        let row = &data[ch * samples..(ch + 1) * samples];
        let slice = &row[center - pre_samples..center - pre_samples + snippet_len];
        let sub_offset = trough_offset(row, center);

        snippets.push(WaveformSnippet {
            primary_channel: ch,
            center_sample: spike.sample_index,
            subsample_offset: sub_offset,
            channel_ids: vec![ch],
            num_samples: snippet_len,
            waveform: slice.to_vec(),
        });
    }

    snippets
}

/// Reads the window `[center − pre, center + post)` on `channels` around each of `centers` from a
/// recording (files read in place, chunk by chunk, not loaded). Spikes whose window leaves the
/// recording, or whose read fails, are skipped. With `subtract_mean`, each channel's mean over the
/// window is removed (raw recordings carry per-channel offsets). `primary_channel` is
/// `channels[0]`.
pub fn read_snippets(
    source: &dyn RecordingSource,
    centers: &[u64],
    channels: &[usize],
    pre_samples: usize,
    post_samples: usize,
    subtract_mean: bool,
) -> Vec<WaveformSnippet> {
    let total = source.info().samples;
    let len = pre_samples + post_samples;
    if channels.is_empty() || len == 0 {
        return Vec::new();
    }
    let mut buf = vec![0.0f32; channels.len() * len];
    let mut out = Vec::with_capacity(centers.len());
    for &center in centers {
        let Some(start) = center.checked_sub(pre_samples as u64) else { continue };
        if start + len as u64 > total || source.read(channels, start..start + len as u64, &mut buf).is_err() {
            continue;
        }
        let mut waveform = buf.clone();
        if subtract_mean {
            for row in waveform.chunks_exact_mut(len) {
                let mean = row.iter().sum::<f32>() / len as f32;
                row.iter_mut().for_each(|v| *v -= mean);
            }
        }
        out.push(WaveformSnippet { primary_channel: channels[0], center_sample: center, subsample_offset: 0.0, channel_ids: channels.to_vec(), num_samples: len, waveform });
    }
    out
}

#[cfg(test)]
mod tests {

    #[test]
    fn test_read_snippets_from_a_recording() {
        // 2 channels, 100 samples: channel c, sample t holds 10c + t
        let data: Vec<f32> = (0..2).flat_map(|c| (0..100).map(move |t| (10 * c + t) as f32)).collect();
        let rec = dsp_core::MemoryRecording::new("r", data, 2, 1000.0).unwrap();
        let s = read_snippets(&rec, &[1, 50, 98], &[1, 0], 2, 3, false);
        assert_eq!(s.len(), 1, "windows leaving the recording are skipped");
        assert_eq!(s[0].channel_ids, vec![1, 0]);
        assert_eq!(&s[0].waveform[..5], &[58.0, 59.0, 60.0, 61.0, 62.0]);
        assert_eq!(&s[0].waveform[5..], &[48.0, 49.0, 50.0, 51.0, 52.0]);
        let centered = read_snippets(&rec, &[50], &[0], 2, 3, true);
        assert_eq!(centered[0].waveform, vec![-2.0, -1.0, 0.0, 1.0, 2.0]);
    }

    use super::*;
    use dsp_io::neuro::probe::tetrode;

    #[test]
    fn test_snippet_batch_extraction_and_indexing() {
        let layout = tetrode();
        let samples = 500;
        let channels = 4;
        let mut data = vec![0.0f32; channels * samples];

        for ch in 0..channels {
            data[ch * samples + 200] = -100.0 / (1.0 + ch as f32);
        }

        let spikes = vec![DeduplicatedSpike {
            primary_channel: 1,
            sample_index: 200,
            peak_amplitude_uv: -50.0,
            participating_channels: vec![0, 1, 2, 3],
        }];

        let batch = extract_snippet_batch_multichannel(
            &data, channels, samples, &spikes, &layout, 4, 10, 20, false,
        );

        assert_eq!(batch.shape(), [1, 4, 30]);
        assert_eq!(batch.snippet_slice(0).len(), 120);
        assert_eq!(batch.channel_slice(0, 0).len(), 30);

        let roundtrip = SnippetBatch::from_snippets(&batch.to_snippets(), batch.peak_index).unwrap();
        assert_eq!(roundtrip, batch);
    }
}
