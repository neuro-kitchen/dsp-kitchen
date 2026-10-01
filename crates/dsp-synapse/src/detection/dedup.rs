use dsp_core::SensorLayout;
use dsp_core::layout::Position3D;
pub use crate::core::{DeduplicatedSpike, SpikeEvent};

/// Deduplicates multi-channel spike events across space and time ("locally exclusive" rule, as
/// SpikeInterface's `locally_exclusive` peak detection).
///
/// A crossing survives if no *deeper* crossing lies within `radius_um` (site distance) and
/// `window_samples` (time) of it; ties go to the earlier sample, then the lower channel. The result
/// does not depend on input order. Each survivor lists the channels of all crossings within the
/// radius and window of it as `participating_channels`. Crossings on channels missing from `layout`
/// are ignored.
pub fn deduplicate_spikes_spatial(
    spikes: &[SpikeEvent],
    layout: &SensorLayout,
    radius_um: f32,
    window_samples: u64,
) -> Vec<DeduplicatedSpike> {
    let mut sorted = spikes.to_vec();
    sort_events(&mut sorted);
    let positions = SitePositions::new(layout);
    locally_exclusive(&sorted, &positions, radius_um, window_samples, 0..u64::MAX)
}

/// Streaming form of [`deduplicate_spikes_spatial`] for crossings arriving window by window.
///
/// [`Self::push`] crossings as they are detected (global sample indices, in window order), then
/// [`Self::finalize_before`] the detection frontier: crossings whose whole ±window neighbourhood has
/// been detected are decided exactly as in a whole-recording run, and each survivor is emitted once.
/// [`Self::finish`] emits the rest after the last window.
#[derive(Debug, Clone)]
pub struct StreamingDedup {
    positions: SitePositions,
    radius_um: f32,
    window_samples: u64,
    pending: Vec<SpikeEvent>,
    emitted_until: u64,
}

impl StreamingDedup {
    pub fn new(layout: &SensorLayout, radius_um: f32, window_samples: u64) -> Self {
        Self {
            positions: SitePositions::new(layout),
            radius_um,
            window_samples,
            pending: Vec::new(),
            emitted_until: 0,
        }
    }

    /// Adds newly detected crossings (global sample indices).
    pub fn push(&mut self, events: &[SpikeEvent]) {
        self.pending.extend_from_slice(events);
        sort_events(&mut self.pending);
    }

    /// Emits survivors among crossings earlier than `frontier − window`, given that every crossing
    /// earlier than `frontier` has been pushed.
    pub fn finalize_before(&mut self, frontier: u64) -> Vec<DeduplicatedSpike> {
        self.emit_until(frontier.saturating_sub(self.window_samples))
    }

    /// Emits all remaining survivors (after the last window).
    pub fn finish(&mut self) -> Vec<DeduplicatedSpike> {
        self.emit_until(u64::MAX)
    }

    fn emit_until(&mut self, end: u64) -> Vec<DeduplicatedSpike> {
        if end <= self.emitted_until {
            return Vec::new();
        }
        let out = locally_exclusive(
            &self.pending,
            &self.positions,
            self.radius_um,
            self.window_samples,
            self.emitted_until..end,
        );
        self.emitted_until = end;
        // Future candidates are ≥ `end`; they only need context from `end − window` on.
        let keep_from = end.saturating_sub(self.window_samples);
        self.pending.retain(|e| e.sample_index >= keep_from);
        out
    }
}

fn sort_events(events: &mut [SpikeEvent]) {
    events.sort_by_key(|e| (e.sample_index, e.channel_id));
}

/// Site positions indexed by channel id.
#[derive(Debug, Clone)]
struct SitePositions(Vec<Option<Position3D>>);

impl SitePositions {
    fn new(layout: &SensorLayout) -> Self {
        let max_id = layout.contacts.iter().map(|c| c.channel_id).max().map_or(0, |m| m + 1);
        let mut table = vec![None; max_id];
        for c in &layout.contacts {
            table[c.channel_id] = Some(c.position);
        }
        Self(table)
    }

    fn get(&self, channel: usize) -> Option<&Position3D> {
        self.0.get(channel).and_then(Option::as_ref)
    }
}

/// `true` when crossing `a` beats `b`: deeper, then earlier, then lower channel.
fn deeper(a: &SpikeEvent, b: &SpikeEvent) -> bool {
    (a.peak_amplitude_uv, a.sample_index, a.channel_id) < (b.peak_amplitude_uv, b.sample_index, b.channel_id)
}

/// Survivors among `sorted` crossings with `sample_index` in `emit`, judged against all of `sorted`.
fn locally_exclusive(
    sorted: &[SpikeEvent],
    positions: &SitePositions,
    radius_um: f32,
    window: u64,
    emit: std::ops::Range<u64>,
) -> Vec<DeduplicatedSpike> {
    let mut out = Vec::new();
    let mut lo = 0usize;
    for (i, cand) in sorted.iter().enumerate() {
        if !emit.contains(&cand.sample_index) {
            continue;
        }
        let Some(pos) = positions.get(cand.channel_id) else { continue };
        while sorted[lo].sample_index + window < cand.sample_index {
            lo += 1;
        }
        let mut survives = true;
        let mut participating = Vec::new();
        for (j, other) in sorted[lo..].iter().enumerate() {
            if other.sample_index > cand.sample_index + window {
                break;
            }
            let Some(other_pos) = positions.get(other.channel_id) else { continue };
            if pos.distance_to(other_pos) > radius_um {
                continue;
            }
            if lo + j != i && deeper(other, cand) {
                survives = false;
                break;
            }
            participating.push(other.channel_id);
        }
        if survives {
            participating.sort_unstable();
            participating.dedup();
            out.push(DeduplicatedSpike {
                primary_channel: cand.channel_id,
                sample_index: cand.sample_index,
                peak_amplitude_uv: cand.peak_amplitude_uv,
                participating_channels: participating,
            });
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use dsp_core::layout::{Position3D, SensorSite};

    #[test]
    fn test_spatial_deduplication() {
        let contacts = vec![
            SensorSite::new(0, Position3D::new(0.0, 0.0, 0.0), 0),
            SensorSite::new(1, Position3D::new(20.0, 0.0, 0.0), 0),
            SensorSite::new(2, Position3D::new(100.0, 0.0, 0.0), 0), // Far channel
        ];
        let layout = SensorLayout::new("Test", contacts);

        let spikes = vec![
            // Two nearby detections of the same spike at sample 100
            SpikeEvent { channel_id: 0, sample_index: 100, peak_amplitude_uv: -120.0 },
            SpikeEvent { channel_id: 1, sample_index: 102, peak_amplitude_uv: -65.0 },
            // Independent spike on far channel 2 at sample 101
            SpikeEvent { channel_id: 2, sample_index: 101, peak_amplitude_uv: -80.0 },
        ];

        let deduped = deduplicate_spikes_spatial(&spikes, &layout, 50.0, 5);
        assert_eq!(deduped.len(), 2);

        // First event should be primary channel 0 with participating [0, 1]
        let ev0 = &deduped[0];
        assert_eq!(ev0.primary_channel, 0);
        assert_eq!(ev0.peak_amplitude_uv, -120.0);
        assert_eq!(ev0.participating_channels, vec![0, 1]);

        // Second event should be independent channel 2
        let ev1 = &deduped[1];
        assert_eq!(ev1.primary_channel, 2);
        assert_eq!(ev1.peak_amplitude_uv, -80.0);
    }

    fn line_layout(n: usize, pitch: f32) -> SensorLayout {
        SensorLayout::new(
            "line",
            (0..n).map(|c| SensorSite::new(c, Position3D::new(0.0, c as f32 * pitch, 0.0), 0)).collect(),
        )
    }

    #[test]
    fn deepest_channel_wins_even_when_it_crosses_last() {
        let layout = line_layout(3, 20.0);
        // Channel 0 crosses first but shallow; the deepest (channel 1) crosses 4 samples later.
        let spikes = vec![
            SpikeEvent { channel_id: 0, sample_index: 100, peak_amplitude_uv: -60.0 },
            SpikeEvent { channel_id: 1, sample_index: 104, peak_amplitude_uv: -150.0 },
            SpikeEvent { channel_id: 2, sample_index: 105, peak_amplitude_uv: -70.0 },
        ];
        let out = deduplicate_spikes_spatial(&spikes, &layout, 50.0, 10);
        assert_eq!(out.len(), 1);
        assert_eq!((out[0].primary_channel, out[0].sample_index), (1, 104));
        assert_eq!(out[0].participating_channels, vec![0, 1, 2]);
    }

    #[test]
    fn result_does_not_depend_on_input_order() {
        let layout = line_layout(8, 20.0);
        let mut spikes: Vec<SpikeEvent> = (0..40)
            .map(|i| SpikeEvent {
                channel_id: (i * 3) % 8,
                sample_index: 100 + (i as u64 * 7) % 60,
                peak_amplitude_uv: -50.0 - ((i * 37) % 23) as f32,
            })
            .collect();
        let reference = deduplicate_spikes_spatial(&spikes, &layout, 30.0, 8);
        spikes.reverse();
        assert_eq!(deduplicate_spikes_spatial(&spikes, &layout, 30.0, 8), reference);
        spikes.rotate_left(13);
        assert_eq!(deduplicate_spikes_spatial(&spikes, &layout, 30.0, 8), reference);
    }

    #[test]
    fn streaming_dedup_matches_whole_run() {
        let layout = line_layout(8, 20.0);
        let spikes: Vec<SpikeEvent> = (0..400)
            .map(|i| SpikeEvent {
                channel_id: (i * 5) % 8,
                sample_index: (i as u64 * 13) % 2_000,
                peak_amplitude_uv: -40.0 - ((i * 31) % 29) as f32,
            })
            .collect();
        let whole = deduplicate_spikes_spatial(&spikes, &layout, 30.0, 10);
        for window in [7u64, 100, 333] {
            let mut stream = StreamingDedup::new(&layout, 30.0, 10);
            let mut out = Vec::new();
            let mut start = 0;
            while start < 2_000 {
                let end = (start + window).min(2_000);
                let chunk: Vec<_> = spikes.iter().filter(|s| (start..end).contains(&s.sample_index)).cloned().collect();
                stream.push(&chunk);
                out.extend(stream.finalize_before(end));
                start = end;
            }
            out.extend(stream.finish());
            assert_eq!(out, whole, "window {window}");
        }
    }
}
