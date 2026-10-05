//! Threshold detection on a filtered trace already in device memory: candidates found and
//! compacted on the device (`dsp_base::peaks::find_peak_candidates`), only they are downloaded,
//! then spaced on the host.

use std::ops::Range;

use cubecl::prelude::*;
use cubecl::server::Handle;
use dsp_base::peaks::find_peak_candidates;

use super::spacing::SpikeSpacing;
use super::threshold::{SpikeEvent, SpikePolarity};

/// Spikes of the `[channels, samples]` device trace `trace` whose local sample lies in `emit`,
/// with **global** sample indices (`global_offset` = global sample of local sample 0).
///
/// `heights` holds each channel's detection height (`[channels]` `f32`, see
/// [`super::detection_heights`]). Candidates are searched in `emit` widened by the spacing distance
/// on each side (within the buffer), so with [`dsp_base::peaks::DistanceRule::LocallyExclusive`]
/// consecutive windows give exactly the whole-recording result when each buffer holds that much
/// context around its `emit` range.
#[allow(clippy::too_many_arguments)]
pub fn execute_detect_spikes_in_vram<R: Runtime>(
    client: &ComputeClient<R>,
    trace: &Handle,
    heights: &Handle,
    channels: usize,
    samples: usize,
    emit: Range<usize>,
    global_offset: u64,
    polarity: SpikePolarity,
    spacing: SpikeSpacing,
) -> Vec<SpikeEvent> {
    if channels == 0 || emit.is_empty() {
        return Vec::new();
    }
    let d = spacing.distance();
    let scan = emit.start.saturating_sub(d)..(emit.end + d).min(samples);
    let candidates = find_peak_candidates::<R, f32>(client, trace, heights, channels, samples, scan, polarity.into());

    let mut events = Vec::new();
    for ch in 0..channels {
        let (indices, values) = candidates.channel(ch);
        let scored: Vec<(usize, f32, f32)> = indices.iter().zip(values).map(|(&t, &v)| (t as usize, v.abs(), v)).collect();
        events.extend(spacing.select(scored).into_iter().filter(|(t, _, _)| emit.contains(t)).map(|(t, _, v)| SpikeEvent {
            channel_id: ch,
            sample_index: global_offset + t as u64,
            peak_amplitude_uv: v,
        }));
    }
    events.sort_by_key(|s| (s.sample_index, s.channel_id));
    events
}
