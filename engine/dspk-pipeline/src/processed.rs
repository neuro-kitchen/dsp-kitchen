//! Live preview of a view's op chain as a [`SignalBackend`]: the views are answered with the
//! recording processed on the device by `dspk_pipeline::preview` (window by window, cached), the
//! raw signal for channels outside the view's group and for windows zoomed out past
//! [`PREVIEW_MAX_VALUES`] ([`ProcessedSignal::covers`] tells the UI to say so).

use std::ops::Range;
use std::sync::Arc;

use dspk_core::{DspResult, RecordingInfo, RecordingSource, SampleFormat};
use dspk_ops::PipelineStage;
use crate::preview::Preview;
use dspk_view::envelope::fold::finish;
use dspk_view::envelope::{fold_row, Columns, EMPTY};
use dspk_view::{Envelope, OnProgress, Progress, SignalBackend, View};

pub use crate::preview::PREVIEW_MAX_VALUES;

/// A view's signal with its op chain applied: [`Preview`] (dspk-pipeline) for the channels the
/// chain processes, the raw signal for the others and for windows too large to preview.
pub struct ProcessedSignal {
    raw: Arc<dyn SignalBackend>,
    preview: Preview,
}

impl ProcessedSignal {
    /// The preview of `stages` over `group` (the view's channels) of `source`; `raw` answers what
    /// is not processed (outside the group, or zoomed out past [`PREVIEW_MAX_VALUES`]).
    pub fn new(raw: Arc<dyn SignalBackend>, source: Arc<dyn RecordingSource>, group: Vec<usize>, stages: Vec<PipelineStage>) -> DspResult<Self> {
        Ok(Self { raw, preview: Preview::new(source, group, stages)? })
    }

    /// The source channel each output row stands for (the read channels through the reorders).
    pub fn out_channels(&self) -> &[usize] {
        self.preview.out_channels()
    }

    /// Whether a window of `samples` samples is previewed processed (else drawn raw).
    pub fn covers(&self, samples: u64) -> bool {
        self.preview.covers(samples)
    }

    /// Makes `view` ready without blocking on the device (a browser's engine worker, which may
    /// not wait for WebGPU): afterwards [`SignalBackend::view`] of it is served from the block
    /// this computed. Views drawn raw (zoomed out past the preview) need nothing.
    pub async fn warm_view_async(&self, view: &View) -> DspResult<()> {
        if !self.covers(view.samples()) {
            return Ok(());
        }
        self.preview.warm_async(view.start..view.end, true).await
    }

    /// `channels` over `samples` (channel-major into `out`): processed rows for the group's
    /// channels, raw for the others.
    fn fill(&self, channels: &[usize], samples: Range<u64>, slack: bool, out: &mut [f32]) -> DspResult<()> {
        let n = (samples.end - samples.start) as usize;
        for k in self.preview.read(channels, samples.clone(), slack, out)? {
            self.raw.read(&[channels[k]], samples.clone(), &mut out[k * n..(k + 1) * n])?;
        }
        Ok(())
    }
}

impl ProcessedSignal {
    /// The processed signal of a view from its op chain (`ops`, the enabled ones in order) over
    /// `channels` (the view's subset, or every channel), as [`crate::preview::view_plan`] lays
    /// it out. Whitenings are fitted here, on the signal of the steps before them: this reads
    /// the recording and takes seconds (call it off the drawing thread).
    pub fn of_ops(raw: Arc<dyn SignalBackend>, source: Arc<dyn RecordingSource>, ops: &[&crate::Node], channels: Vec<usize>, base_dir: &std::path::Path) -> Result<Self, String> {
        let plan = crate::preview::view_plan(ops, channels, base_dir)?;
        let mut stages: Vec<PipelineStage> = plan.steps.into_iter().map(|(_, stage)| stage).collect();
        // A whitening's fit reads the device back as it goes, which a browser cannot wait for
        if cfg!(target_family = "wasm") && !plan.fits.is_empty() {
            return Err("whitening is fitted on a server or in the desktop app, not in a browser yet".into());
        }
        for (at, fit) in plan.fits {
            let target = dspk_core::ComputeTarget::from_env().map_err(|e| e.to_string())?;
            let fitted = crate::fit_whitening(target, source.as_ref(), &plan.group, &stages[..at], &fit).map_err(|e| e.to_string())?;
            stages[at] = PipelineStage::SpatialWhitening(fitted);
        }
        Self::new(raw, source, plan.group, stages).map_err(|e| e.to_string())
    }

    /// [`Self::of_ops`] for ops sent as JSON (pipeline nodes, in order) over `channels` (none =
    /// all): a view's processing as a client that does not compute asks a server for it.
    /// `base_dir` is where files the ops name (a remap's channel map) are read from.
    pub fn of_json_ops(raw: Arc<dyn SignalBackend>, source: Arc<dyn RecordingSource>, ops: &[String], channels: &[usize], base_dir: &std::path::Path) -> Result<Self, String> {
        let nodes: Vec<crate::Node> = ops.iter().map(|op| serde_json::from_str(op).map_err(|e| format!("an op that is not a pipeline node: {e}"))).collect::<Result<_, _>>()?;
        let channels: Vec<usize> = if channels.is_empty() { (0..source.info().channel_count()).collect() } else { channels.to_vec() };
        Self::of_ops(raw, source, &nodes.iter().collect::<Vec<_>>(), channels, base_dir)
    }

    /// `channels` over `samples`, processed whatever the size (a branch reading this preview
    /// as its input: its halo makes the request larger than a view's).
    pub fn read_processed(&self, channels: &[usize], samples: Range<u64>, out: &mut [f32]) -> DspResult<()> {
        self.fill(channels, samples, false, out)
    }
}

/// Processed signals kept per recording.
const CHAINS_KEPT: usize = 8;

/// The processed signals a client's views asked for, by chain (channels and ops), so a view can
/// be made ready asynchronously ([`ProcessedSignal::warm_view_async`]) before it is answered: what
/// `dspk serve` and a browser's engine hold for a recording they serve. Files the ops name (a
/// remap's channel map) are read from `base_dir`.
pub struct ProcessedChains {
    base_dir: std::path::PathBuf,
    made: std::sync::Mutex<Vec<((Vec<usize>, Vec<String>), Arc<ProcessedSignal>)>>,
}

impl ProcessedChains {
    pub fn new(base_dir: impl Into<std::path::PathBuf>) -> Self {
        Self { base_dir: base_dir.into(), made: Default::default() }
    }

    /// The signal of `ops` over `channels` (none = all) of `source` (`raw` answers what is not
    /// processed). Made now (whitenings are fitted: seconds) and kept, most recent last.
    pub fn make(&self, raw: Arc<dyn SignalBackend>, source: Arc<dyn RecordingSource>, channels: &[usize], ops: &[String]) -> Result<Arc<ProcessedSignal>, String> {
        let signal = Arc::new(ProcessedSignal::of_json_ops(raw, source, ops, channels, &self.base_dir)?);
        let mut made = self.made.lock().unwrap_or_else(|e| e.into_inner());
        made.retain(|(key, _)| key.0 != channels || key.1 != ops);
        if made.len() >= CHAINS_KEPT {
            made.remove(0);
        }
        made.push(((channels.to_vec(), ops.to_vec()), signal.clone()));
        Ok(signal)
    }

    /// Makes `view` of the chain of `ops` over `channels` ready without blocking on the device;
    /// nothing to do for a chain not made here.
    pub async fn warm(&self, channels: &[usize], ops: &[String], view: &View) -> DspResult<()> {
        let signal = self.made.lock().unwrap_or_else(|e| e.into_inner()).iter().find(|(key, _)| key.0 == channels && key.1 == ops).map(|(_, s)| s.clone());
        match signal {
            Some(signal) => signal.warm_view_async(view).await,
            None => Ok(()),
        }
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

    /// A source that gives only scaled values (no stored reads, as float64 recordings and some
    /// formats): previewed through float32 scaled on the host.
    struct ScaledOnly(Arc<dyn RecordingSource>);
    impl RecordingSource for ScaledOnly {
        fn info(&self) -> &dspk_core::RecordingInfo {
            self.0.info()
        }
        fn read(&self, channels: &[usize], samples: Range<u64>, out: &mut [f32]) -> DspResult<()> {
            self.0.read(channels, samples, out)
        }
        fn read_stored(&self, _channels: &[usize], _samples: Range<u64>, _out: &mut [u8]) -> DspResult<()> {
            Err(dspk_core::DspError::UnsupportedFormat("stored reads".into()))
        }
    }

    #[test]
    fn a_source_without_stored_reads_previews_as_one_with_them() {
        let source = recording(2, 6000);
        let scaled: Arc<dyn RecordingSource> = Arc::new(ScaledOnly(source.clone()));
        let stages = vec![PipelineStage::bandpass(20.0, 200.0), PipelineStage::Scale { alpha: 2.0, beta: 0.5 }];
        let stored = ProcessedSignal::new(Arc::new(LocalSignal::open(source.clone(), None).unwrap()), source, vec![0, 1], stages.clone()).unwrap();
        let host = ProcessedSignal::new(Arc::new(LocalSignal::open(scaled.clone(), None).unwrap()), scaled, vec![0, 1], stages).unwrap();
        let (mut a, mut b) = (vec![0.0f32; 2 * 400], vec![0.0f32; 2 * 400]);
        stored.read(&[0, 1], 2000..2400, &mut a).unwrap();
        host.read(&[0, 1], 2000..2400, &mut b).unwrap();
        for i in 0..800 {
            assert!((a[i] - b[i]).abs() < 1e-4 * a[i].abs().max(1.0), "{i}: {} vs {}", a[i], b[i]);
        }
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
