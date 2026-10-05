use dsp_io::neuro::probe::{Position3D, SensorLayout};
pub use crate::core::{DeduplicatedSpike, SpikeEvent};

/// Deduplicates multi-channel spike events across space and time ("locally exclusive" rule, as
/// SpikeInterface's `locally_exclusive` peak detection).
///
/// A crossing survives if no *stronger* crossing (larger `|peak amplitude|`, so negative troughs
/// and positive peaks alike) lies within `radius_um` (site distance) and `window_samples` (time) of
/// it; ties go to the earlier sample, then the lower channel. The result
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

/// `true` when crossing `a` beats `b`: larger magnitude, then earlier, then lower channel.
fn stronger(a: &SpikeEvent, b: &SpikeEvent) -> bool {
    b.peak_amplitude_uv
        .abs()
        .total_cmp(&a.peak_amplitude_uv.abs())
        .then((a.sample_index, a.channel_id).cmp(&(b.sample_index, b.channel_id)))
        .is_lt()
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
            if lo + j != i && stronger(other, cand) {
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

/// Channels within a radius of each channel (itself included), ascending, as CSR rows; the
/// device form of the neighbourhood test in [`deduplicate_spikes_spatial`]. Build once per probe
/// and radius, then [`Self::deduplicate`] every batch.
#[derive(Debug, Clone)]
pub struct DedupNeighbours {
    positions: SitePositions,
    offsets: Vec<u32>,
    channels: Vec<u32>,
    /// `u32` words per crossing in the participating-channel bitmask.
    mask_words: usize,
    offsets_h: cubecl::server::Handle,
    channels_h: cubecl::server::Handle,
}

impl DedupNeighbours {
    pub fn new<R: cubecl::prelude::Runtime>(
        client: &cubecl::prelude::ComputeClient<R>,
        layout: &SensorLayout,
        radius_um: f32,
    ) -> Self {
        use dsp_base::core::buffer;
        let positions = SitePositions::new(layout);
        let mut offsets = vec![0u32];
        let mut channels = Vec::new();
        let mut widest = 1usize;
        for a in 0..positions.0.len() {
            let row_start = channels.len();
            if let Some(pa) = positions.get(a) {
                for b in 0..positions.0.len() {
                    if positions.get(b).is_some_and(|pb| pa.distance_to(pb) <= radius_um) {
                        channels.push(b as u32);
                    }
                }
            }
            widest = widest.max(channels.len() - row_start);
            offsets.push(channels.len() as u32);
        }
        // Kernels never read an empty buffer
        let channels_h = buffer::upload(client, if channels.is_empty() { &[0u32][..] } else { &channels });
        Self {
            offsets_h: buffer::upload(client, &offsets),
            channels_h,
            positions,
            offsets,
            channels,
            mask_words: widest.div_ceil(32),
        }
    }

    /// Radius neighbours of `channel` (ascending; empty for channels missing from the layout).
    pub fn of(&self, channel: usize) -> &[u32] {
        match (self.offsets.get(channel), self.offsets.get(channel + 1)) {
            (Some(&a), Some(&b)) => &self.channels[a as usize..b as usize],
            _ => &[],
        }
    }

    /// [`deduplicate_spikes_spatial`] on the device. Only the survival flags and participating
    /// bitmasks (`1 + mask_words` words per crossing) are downloaded.
    pub fn deduplicate<R: cubecl::prelude::Runtime>(
        &self,
        client: &cubecl::prelude::ComputeClient<R>,
        spikes: &[SpikeEvent],
        window_samples: u64,
    ) -> Vec<DeduplicatedSpike> {
        use cubecl::prelude::*;
        use dsp_base::core::buffer;
        use dsp_core::compute::LaunchGeometry;
        use super::kernels::spatial_dedup_survival_kernel;

        let mut sorted: Vec<SpikeEvent> =
            spikes.iter().filter(|e| self.positions.get(e.channel_id).is_some()).cloned().collect();
        if sorted.is_empty() {
            return Vec::new();
        }
        sort_events(&mut sorted);

        let n = sorted.len();
        let base_sample = sorted[0].sample_index;
        let span = sorted[n - 1].sample_index - base_sample;
        assert!(span + window_samples < u32::MAX as u64, "dedup batch spans more than u32 samples");
        let sample_indices: Vec<u32> = sorted.iter().map(|e| (e.sample_index - base_sample) as u32).collect();
        let channel_ids: Vec<u32> = sorted.iter().map(|e| e.channel_id as u32).collect();
        let magnitudes: Vec<f32> = sorted.iter().map(|e| e.peak_amplitude_uv.abs()).collect();

        let surv_h = buffer::empty::<R, u32>(client, n);
        let mask_h = buffer::empty::<R, u32>(client, n * self.mask_words);
        let geom = LaunchGeometry::elementwise(client, n);
        // SAFETY: every array is passed with the length it was created with
        unsafe {
            spatial_dedup_survival_kernel::launch::<R>(
                client,
                geom.cube_count,
                geom.cube_dim,
                ArrayArg::from_raw_parts(buffer::upload(client, &sample_indices), n),
                ArrayArg::from_raw_parts(buffer::upload(client, &channel_ids), n),
                ArrayArg::from_raw_parts(buffer::upload(client, &magnitudes), n),
                ArrayArg::from_raw_parts(self.offsets_h.clone(), self.offsets.len()),
                ArrayArg::from_raw_parts(self.channels_h.clone(), self.channels.len().max(1)),
                ArrayArg::from_raw_parts(surv_h.clone(), n),
                ArrayArg::from_raw_parts(mask_h.clone(), n * self.mask_words),
                n as u32,
                self.mask_words as u32,
                window_samples as u32,
            );
        }
        let survives = buffer::download::<R, u32>(client, surv_h);
        let masks = buffer::download::<R, u32>(client, mask_h);

        sorted
            .iter()
            .enumerate()
            .filter(|&(i, _)| survives[i] != 0)
            .map(|(i, cand)| {
                let mask = &masks[i * self.mask_words..(i + 1) * self.mask_words];
                // Neighbour rows are ascending, so the channels come out sorted and unique
                let participating_channels = self
                    .of(cand.channel_id)
                    .iter()
                    .enumerate()
                    .filter(|&(slot, _)| mask[slot / 32] & (1 << (slot % 32)) != 0)
                    .map(|(_, &ch)| ch as usize)
                    .collect();
                DeduplicatedSpike {
                    primary_channel: cand.channel_id,
                    sample_index: cand.sample_index,
                    peak_amplitude_uv: cand.peak_amplitude_uv,
                    participating_channels,
                }
            })
            .collect()
    }
}

/// One-off [`DedupNeighbours::deduplicate`]: builds the neighbour table for this call. Reuse a
/// [`DedupNeighbours`] when deduplicating more than one batch.
pub fn deduplicate_spikes_spatial_gpu<R: cubecl::prelude::Runtime>(
    client: &cubecl::prelude::ComputeClient<R>,
    spikes: &[SpikeEvent],
    layout: &SensorLayout,
    radius_um: f32,
    window_samples: u64,
) -> Vec<DeduplicatedSpike> {
    if spikes.is_empty() {
        return Vec::new();
    }
    DedupNeighbours::new(client, layout, radius_um).deduplicate(client, spikes, window_samples)
}

#[cfg(test)]
mod tests {
    use super::*;
    use dsp_io::neuro::probe::SensorSite;

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

    #[test]
    fn test_strongest_crossing_wins_for_either_polarity() {
        let layout = line_layout(2, 20.0);
        let positive = [
            SpikeEvent { channel_id: 0, sample_index: 100, peak_amplitude_uv: 60.0 },
            SpikeEvent { channel_id: 1, sample_index: 101, peak_amplitude_uv: 140.0 },
        ];
        let kept = deduplicate_spikes_spatial(&positive, &layout, 50.0, 5);
        assert_eq!((kept.len(), kept[0].peak_amplitude_uv), (1, 140.0));
        let mixed = [
            SpikeEvent { channel_id: 0, sample_index: 100, peak_amplitude_uv: -90.0 },
            SpikeEvent { channel_id: 1, sample_index: 101, peak_amplitude_uv: 70.0 },
        ];
        let kept = deduplicate_spikes_spatial(&mixed, &layout, 50.0, 5);
        assert_eq!((kept.len(), kept[0].peak_amplitude_uv), (1, -90.0));
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

    #[test]
    fn gpu_spatial_dedup_matches_cpu_exactly() {
        use cubecl::prelude::*;
        use dsp_core::compute::{ComputeTarget, ComputeTask};

        struct Task<'a>(&'a [SpikeEvent], &'a SensorLayout, f32, u64);
        impl ComputeTask for Task<'_> {
            type Output = Vec<DeduplicatedSpike>;
            fn run<R: Runtime>(self, client: ComputeClient<R>) -> Vec<DeduplicatedSpike> {
                deduplicate_spikes_spatial_gpu::<R>(&client, self.0, self.1, self.2, self.3)
            }
        }

        // 8 sites (one mask word) and 48 sites within one radius (two words)
        for (sites, pitch, radius) in [(8usize, 20.0f32, 35.0f32), (48, 1.0, 100.0)] {
            let layout = line_layout(sites, pitch);
            let spikes: Vec<SpikeEvent> = (0..120)
                .map(|i| SpikeEvent {
                    channel_id: (i * 5) % sites,
                    sample_index: 50 + (i as u64 * 9) % 500,
                    peak_amplitude_uv: -45.0 - ((i * 31) % 40) as f32,
                })
                .collect();
            let cpu_out = deduplicate_spikes_spatial(&spikes, &layout, radius, 10);
            let targets = ComputeTarget::available();
            assert!(!targets.is_empty(), "no CubeCL runtime compiled in");
            for target in targets {
                let gpu_out = target.run(Task(&spikes, &layout, radius, 10)).expect("runtime");
                assert_eq!(gpu_out, cpu_out, "{sites} sites");
            }
        }
    }
}
