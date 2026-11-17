//! Any recording replayed at its own rate, as if it were being acquired: a file as a live source
//! (`device.toml`: `kind = "replay"`). Tests a chain on real data before a sensor exists.

use std::sync::Arc;
use std::time::{Duration, Instant};

use dspk_core::{DspResult, RecordingSource, SampleFormat};

use super::{block_samples, encode_stored, sample_time_ns, unix_now_ns, LiveBlock, LiveEvent, LiveLayout, LiveSource};

pub struct Paced {
    layout: LiveLayout,
    source: Arc<dyn RecordingSource>,
    block: usize,
    next: u64,
    /// Start again at the end instead of ending.
    looped: bool,
    /// Samples of earlier passes (a looped replay keeps counting).
    offset: u64,
    started: Option<Instant>,
}

impl Paced {
    /// Replays `source` in blocks of `block_ms`. Sources with native stored reads keep their
    /// stored type; others are handed out as float32.
    pub fn new(source: Arc<dyn RecordingSource>, block_ms: f64, looped: bool) -> Self {
        let info = source.info();
        let native = matches!(info.format, SampleFormat::I16 | SampleFormat::I32 | SampleFormat::F32);
        let format = if native { info.format } else { SampleFormat::F32 };
        let mut channels = info.channels.clone();
        if !native {
            for c in &mut channels {
                (c.gain, c.offset) = (1.0, 0.0);
            }
        }
        let layout = LiveLayout { name: info.name.clone(), channels, sample_rate: info.sample_rate, format };
        Self { block: block_samples(block_ms, info.sample_rate), layout, source, next: 0, looped, offset: 0, started: None }
    }

    /// Names the source (a device's configuration names it, not the file).
    pub fn rename(&mut self, name: &str) {
        self.layout.name = name.to_string();
    }

    fn read_block(&self, first: u64, samples: usize) -> DspResult<Vec<u8>> {
        let all: Vec<usize> = (0..self.layout.channel_count()).collect();
        let range = first..first + samples as u64;
        if self.layout.format == self.source.info().format {
            let mut bytes = vec![0u8; self.layout.frame_bytes() * samples];
            if self.source.read_stored(&all, range.clone(), &mut bytes).is_ok() {
                return Ok(bytes);
            }
        }
        let mut values = vec![0.0f32; all.len() * samples];
        self.source.read(&all, range, &mut values)?;
        let mut plain = self.layout.channels.clone();
        for c in &mut plain {
            (c.gain, c.offset) = (1.0, 0.0);
        }
        encode_stored(&values, &plain, samples, SampleFormat::F32)
    }
}

impl LiveSource for Paced {
    fn layout(&self) -> &LiveLayout {
        &self.layout
    }

    fn next(&mut self, wait: Duration) -> DspResult<LiveEvent> {
        let total = self.source.info().samples;
        if self.next >= total {
            if !self.looped || total == 0 {
                return Ok(LiveEvent::Ended);
            }
            self.offset += total;
            self.next = 0;
        }
        let started = *self.started.get_or_insert_with(Instant::now);
        let samples = (self.block as u64).min(total - self.next) as usize;
        let index = self.offset + self.next;
        let due = started + Duration::from_nanos(sample_time_ns(index + samples as u64, self.layout.sample_rate));
        let now = Instant::now();
        if due > now {
            if due - now > wait {
                std::thread::sleep(wait);
                return Ok(LiveEvent::Idle);
            }
            std::thread::sleep(due - now);
        }
        let data = self.read_block(self.next, samples)?;
        self.next += samples as u64;
        Ok(LiveEvent::Block(LiveBlock {
            first_sample: index,
            samples,
            source_time_ns: sample_time_ns(index, self.layout.sample_rate),
            received_unix_ns: unix_now_ns(),
            lost_before: 0,
            data,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dspk_core::MemoryRecording;

    #[test]
    fn replays_then_ends_or_loops() {
        let data: Vec<f32> = (0..2).flat_map(|c| (0..25).map(move |s| (c * 100 + s) as f32)).collect();
        let rec: Arc<dyn RecordingSource> = Arc::new(MemoryRecording::new("m", data, 2, 1000.0).unwrap());
        let mut once = Paced::new(rec.clone(), 10.0, false);
        let mut ends = Vec::new();
        loop {
            match once.next(Duration::from_millis(50)).unwrap() {
                LiveEvent::Block(b) => ends.push(b.end_sample()),
                LiveEvent::Idle => {}
                LiveEvent::Ended => break,
            }
        }
        assert_eq!(ends, vec![10, 20, 25]);

        let mut looped = Paced::new(rec, 10.0, true);
        let mut firsts = Vec::new();
        while firsts.len() < 4 {
            if let LiveEvent::Block(b) = looped.next(Duration::from_millis(50)).unwrap() {
                assert_eq!(b.to_f32(looped.layout())[0], (b.first_sample % 25) as f32);
                firsts.push(b.first_sample);
            }
        }
        assert_eq!(firsts, vec![0, 10, 20, 25]);
    }
}
