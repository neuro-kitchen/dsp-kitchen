//! Live preview of a view's op chain: a [`SignalBackend`] that answers views with the recording
//! processed on the device, window by window, so an edit shows on what is on screen without
//! writing anything.
//!
//! - The view's channels (its recording: all channels of the source, or the subset a copied view
//!   reads) are processed together as one grouped array (references such as CMR span them).
//! - Each request reads the window plus the chain's settling halo on both sides, uploads the
//!   recording's **stored** values (int16 stays int16; scaled on the device), runs the chain and
//!   downloads the interior: equal to processing the whole recording. Float64 recordings are
//!   scaled on the host (wgpu has no f64), as in the pipeline executor.
//! - The processed block (with a margin of half a window on each side) is cached, so panning a
//!   little, hover readouts and redraws reuse it.
//! - Windows larger than [`PREVIEW_MAX_VALUES`] (channels × samples) are not processed: zoomed
//!   out that far the view draws the raw recording from its pyramid ([`ProcessedSignal::covers`]
//!   tells the UI to say so).

use std::ops::Range;
use std::sync::{Arc, Mutex};

use dspk_core::{ComputeTarget, DspError, DspResult, RecordingInfo, RecordingSource, SampleFormat};
use dspk_ops::core::buffer;
use dspk_ops::pipeline::PipelineWorkspace;
use dspk_ops::{Pipeline, PipelineStage};
use dspk_view::envelope::fold::finish;
use dspk_view::envelope::{fold_row, Columns, EMPTY};
use dspk_view::{Envelope, OnProgress, Progress, SignalBackend, View};

/// Most values (channels × samples, halo and margin included) one preview processes: 64 Mi
/// float32 = 256 MiB on the device (384 channels: about 5.8 s at 30 kHz).
pub const PREVIEW_MAX_VALUES: usize = 1 << 26;

/// One processed block of the view's channels, channel-major.
struct Block {
    range: Range<u64>,
    data: Vec<f32>,
}

struct State {
    workspace: Option<(PipelineWorkspace<f32>, usize)>,
    block: Option<Block>,
}

pub struct ProcessedSignal {
    raw: Arc<dyn SignalBackend>,
    source: Arc<dyn RecordingSource>,
    /// Channels processed together, in source order (indices of the source).
    group: Vec<usize>,
    pipeline: Pipeline,
    halo: (usize, usize),
    stored: bool,
    state: Mutex<State>,
}

impl ProcessedSignal {
    /// The preview of `stages` over `group` (the view's channels) of `source`; `raw` answers what
    /// is not processed (outside the group, or zoomed out past [`PREVIEW_MAX_VALUES`]).
    pub fn new(raw: Arc<dyn SignalBackend>, source: Arc<dyn RecordingSource>, group: Vec<usize>, stages: Vec<PipelineStage>) -> DspResult<Self> {
        let info = source.info();
        let pipeline = Pipeline::with_stages(stages);
        let halo = pipeline.settling(info.sample_rate_hz())?;
        let stored = info.format != SampleFormat::F64;
        Ok(Self { raw, source, group, pipeline, halo, stored, state: Mutex::new(State { workspace: None, block: None }) })
    }

    /// Whether a window of `samples` samples is previewed processed (else drawn raw).
    pub fn covers(&self, samples: u64) -> bool {
        self.block_values(samples) <= PREVIEW_MAX_VALUES
    }

    fn block_values(&self, samples: u64) -> usize {
        self.group.len() * (samples as usize + self.halo.0 + self.halo.1)
    }

    /// The processed block holding `range` (computed now when the cached one does not). `slack`:
    /// with half a window more on each side, so small pans reuse it (a view's reads); a branch
    /// reading this preview asks for exactly what it needs (its own block has the slack).
    fn with_block<R>(&self, range: Range<u64>, slack: bool, f: impl FnOnce(&Block) -> R) -> DspResult<R> {
        let mut state = self.state.lock().unwrap();
        if let Some(b) = state.block.as_ref().filter(|b| b.range.start <= range.start && range.end <= b.range.end) {
            return Ok(f(b));
        }
        let total = self.source.info().samples;
        // Half a window of margin each side while it fits the budget: small pans reuse the block
        let len = range.end - range.start;
        let margin = if slack && self.block_values(2 * len) <= PREVIEW_MAX_VALUES { len / 2 } else { 0 };
        let want = range.start.saturating_sub(margin)..(range.end + margin).min(total);
        let read = want.start.saturating_sub(self.halo.0 as u64)..(want.end + self.halo.1 as u64).min(total);
        let n = (read.end - read.start) as usize;
        let nch = self.group.len();

        let (ws, capacity) = match state.workspace.take() {
            Some((ws, cap)) if cap >= n => (ws, cap),
            _ => {
                let client = ComputeTarget::from_env().and_then(ComputeTarget::client).map_err(|e| DspError::ComputeError(e.to_string()))?;
                let rate = self.source.info().sample_rate_hz();
                let mut ws = PipelineWorkspace::<f32>::new(client, self.pipeline.clone(), nch, n, rate)?;
                if self.stored {
                    let info = self.source.info();
                    let gains: Vec<f32> = self.group.iter().map(|&c| info.channels[c].gain).collect();
                    let offsets: Vec<f32> = self.group.iter().map(|&c| info.channels[c].offset).collect();
                    ws.set_stored_scaling(&gains, &offsets);
                }
                (ws, n)
            }
        };
        let mut ws = ws;
        // Fixed launch choices: a preview never stops to benchmark kernels for a window size it
        // has not seen (zooming meets a new size at every step), and its numbers do not depend
        // on what a tuner picked
        let _pinned = dspk_core::compute::pin_tuned_choices();
        let format = self.source.info().format;
        let mut bytes = if self.stored { vec![0u8; nch * n * format.bytes()] } else { Vec::new() };
        // Sources that cannot give stored values (no native stored read) are scaled on the host
        let stored = self.stored && self.source.read_stored(&self.group, read.clone(), &mut bytes).is_ok();
        let processed = if stored {
            let handle = ws.process_owned_stored_chunk_in_vram(bytes, format, n)?;
            buffer::download_prefix::<f32>(ws.client(), handle, nch * n)
        } else {
            let mut values = vec![0.0f32; nch * n];
            self.source.read(&self.group, read.clone(), &mut values)?;
            let mut out = vec![0.0f32; nch * n];
            ws.process_chunk(&values, n, &mut out);
            out
        };
        state.workspace = Some((ws, capacity));
        // Keep the interior (`want`), channel-major
        let (a, m) = ((want.start - read.start) as usize, (want.end - want.start) as usize);
        let mut data = Vec::with_capacity(nch * m);
        for r in 0..nch {
            data.extend_from_slice(&processed[r * n + a..r * n + a + m]);
        }
        let block = Block { range: want, data };
        let out = f(&block);
        state.block = Some(block);
        Ok(out)
    }

    /// `channels` over `samples` (channel-major into `out`): processed rows for the group's
    /// channels, raw for the others.
    fn fill(&self, channels: &[usize], samples: Range<u64>, slack: bool, out: &mut [f32]) -> DspResult<()> {
        let n = (samples.end - samples.start) as usize;
        let rows: Vec<Option<usize>> = channels.iter().map(|c| self.group.iter().position(|g| g == c)).collect();
        if rows.iter().any(Option::is_some) {
            self.with_block(samples.clone(), slack, |b| {
                let (m, a) = ((b.range.end - b.range.start) as usize, (samples.start - b.range.start) as usize);
                for (k, row) in rows.iter().enumerate() {
                    if let Some(r) = row {
                        out[k * n..(k + 1) * n].copy_from_slice(&b.data[r * m + a..r * m + a + n]);
                    }
                }
            })?;
        }
        for (k, (&c, row)) in channels.iter().zip(&rows).enumerate() {
            if row.is_none() {
                self.raw.read(&[c], samples.clone(), &mut out[k * n..(k + 1) * n])?;
            }
        }
        Ok(())
    }
}

impl ProcessedSignal {
    /// `channels` over `samples`, processed whatever the size (a branch reading this preview
    /// as its input: its halo makes the request larger than a view's).
    pub fn read_processed(&self, channels: &[usize], samples: Range<u64>, out: &mut [f32]) -> DspResult<()> {
        self.fill(channels, samples, false, out)
    }
}

/// A preview read as a recording (float32, unit gain): the input of a branch that shares its
/// first ops with another, so they are computed once (the branch runs only its own ops on it).
pub struct PreviewSource {
    preview: Arc<ProcessedSignal>,
    info: RecordingInfo,
}

impl PreviewSource {
    pub fn new(preview: Arc<ProcessedSignal>) -> Self {
        let mut info = preview.raw.info().clone();
        info.format = SampleFormat::F32;
        for c in &mut info.channels {
            c.gain = 1.0;
            c.offset = 0.0;
        }
        Self { preview, info }
    }
}

impl RecordingSource for PreviewSource {
    fn info(&self) -> &RecordingInfo {
        &self.info
    }

    fn read(&self, channels: &[usize], samples: Range<u64>, out: &mut [f32]) -> DspResult<()> {
        self.preview.read_processed(channels, samples, out)
    }
}

impl SignalBackend for ProcessedSignal {
    fn info(&self) -> &RecordingInfo {
        self.raw.info()
    }

    fn view(&self, view: &View, out: &mut Envelope) -> DspResult<()> {
        let n = view.samples();
        if !self.covers(n) {
            return self.raw.view(view, out);
        }
        view.validate(self.info().channel_count(), self.info().samples)?;
        let rows = view.channels.len();
        let mut samples = vec![0.0f32; rows * n as usize];
        self.fill(&view.channels, view.start..view.end, true, &mut samples)?;
        if n <= view.width as u64 {
            *out = Envelope::Samples(samples);
            return Ok(());
        }
        let columns = Columns::Even { start: view.start, len: n, width: view.width };
        let mut values = vec![EMPTY; rows * view.width];
        for r in 0..rows {
            fold_row(&samples[r * n as usize..(r + 1) * n as usize], view.start, columns, &mut values[r * view.width..(r + 1) * view.width]);
        }
        finish(&mut values);
        *out = Envelope::Columns { values, complete: true };
        Ok(())
    }

    fn read(&self, channels: &[usize], samples: Range<u64>, out: &mut [f32]) -> DspResult<()> {
        if !self.covers(samples.end - samples.start) {
            return self.raw.read(channels, samples, out);
        }
        self.fill(channels, samples, true, out)
    }

    fn focus(&self, sample: u64) {
        self.raw.focus(sample);
    }

    fn prepare(&self, start: u64, end: u64) -> DspResult<()> {
        self.raw.prepare(start, end)
    }

    fn progress(&self) -> Progress {
        self.raw.progress()
    }

    fn watch(&self, on_progress: OnProgress) {
        self.raw.watch(on_progress);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dspk_core::MemoryRecording;
    use dspk_view::LocalSignal;

    fn recording(channels: usize, samples: usize) -> Arc<dyn RecordingSource> {
        let data: Vec<f32> = (0..channels * samples).map(|i| ((i % samples) as f32 * 0.37).sin() + (i / samples) as f32).collect();
        Arc::new(MemoryRecording::new("test", data, channels, 1000.0).unwrap())
    }

    #[test]
    fn test_scale_preview_matches_the_arithmetic() {
        let source = recording(3, 4000);
        let raw: Arc<dyn SignalBackend> = Arc::new(LocalSignal::open(source.clone(), None).unwrap());
        let p = ProcessedSignal::new(raw.clone(), source, vec![0, 2], vec![PipelineStage::Scale { alpha: 2.0, beta: 1.0 }]).unwrap();
        let (mut a, mut b) = (vec![0.0f32; 3 * 100], vec![0.0f32; 3 * 100]);
        p.read(&[0, 1, 2], 1000..1100, &mut a).unwrap();
        raw.read(&[0, 1, 2], 1000..1100, &mut b).unwrap();
        for i in 0..100 {
            assert!((a[i] - (2.0 * b[i] + 1.0)).abs() < 1e-5, "channel 0 is processed");
            assert_eq!(a[100 + i], b[100 + i], "channel 1 is outside the group: raw");
            assert!((a[200 + i] - (2.0 * b[200 + i] + 1.0)).abs() < 1e-5);
        }
    }

    #[test]
    fn test_a_branch_on_a_shared_prefix_equals_its_whole_chain() {
        let source = recording(2, 6000);
        let raw: Arc<dyn SignalBackend> = Arc::new(LocalSignal::open(source.clone(), None).unwrap());
        let prefix = Arc::new(ProcessedSignal::new(raw.clone(), source.clone(), vec![0, 1], vec![PipelineStage::bandpass(20.0, 200.0)]).unwrap());
        let branch = ProcessedSignal::new(raw.clone(), Arc::new(PreviewSource::new(prefix)), vec![0, 1], vec![PipelineStage::Scale { alpha: 3.0, beta: 0.0 }]).unwrap();
        let whole = ProcessedSignal::new(raw, source, vec![0, 1], vec![PipelineStage::bandpass(20.0, 200.0), PipelineStage::Scale { alpha: 3.0, beta: 0.0 }]).unwrap();
        let (mut a, mut b) = (vec![0.0f32; 2 * 400], vec![0.0f32; 2 * 400]);
        branch.read(&[0, 1], 2000..2400, &mut a).unwrap();
        whole.read(&[0, 1], 2000..2400, &mut b).unwrap();
        for i in 0..800 {
            assert!((a[i] - b[i]).abs() < 1e-3 * b[i].abs().max(1.0), "{i}: {} vs {}", a[i], b[i]);
        }
    }

    #[test]
    fn test_window_equals_processing_the_whole_recording() {
        let source = recording(2, 6000);
        let raw: Arc<dyn SignalBackend> = Arc::new(LocalSignal::open(source.clone(), None).unwrap());
        let stages = vec![PipelineStage::bandpass(20.0, 200.0), PipelineStage::CommonMedianReference];
        let p = ProcessedSignal::new(raw.clone(), source.clone(), vec![0, 1], stages.clone()).unwrap();
        let mut window = vec![0.0f32; 2 * 500];
        p.read(&[0, 1], 3000..3500, &mut window).unwrap();
        // The whole recording through the same chain, on the same device
        let whole = ProcessedSignal::new(raw, source, vec![0, 1], stages).unwrap();
        let mut all = vec![0.0f32; 2 * 6000];
        whole.read(&[0, 1], 0..6000, &mut all).unwrap();
        for c in 0..2 {
            for i in 0..500 {
                let (w, a) = (window[c * 500 + i], all[c * 6000 + 3000 + i]);
                assert!((w - a).abs() < 1e-3 * a.abs().max(1.0), "ch {c} sample {i}: {w} vs {a}");
            }
        }
    }
}
