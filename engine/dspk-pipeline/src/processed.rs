//! Live preview of a view's op chain as a [`SignalBackend`]: the views are answered with the
//! recording processed on the device by `dspk_pipeline::preview` (window by window, cached), the
//! raw signal for channels outside the view's group and for windows zoomed out past
//! [`PREVIEW_MAX_VALUES`] ([`ProcessedSignal::covers`] tells the UI to say so).

use std::ops::Range;
use std::sync::Arc;

use dspk_core::{DspResult, RecordingInfo, RecordingSource, SampleFormat};
use dspk_ops::PipelineStage;
use crate::preview::Preview;
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

    /// The same with blocks and tiles of at most `max_values` values ([`Preview::with_max_values`]).
    pub fn with_max_values(mut self, max_values: usize) -> Self {
        self.preview = self.preview.with_max_values(max_values);
        self
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
        // Zoomed out: reduced on the device, only the columns read back
        if view.samples() > view.width as u64 {
            return self.preview.columns_async(view.start..view.end, view.width).await.map(|_| ());
        }
        self.preview.warm_async(view.start..view.end, true).await
    }

    /// A zoomed-out `view` from the device's columns ([`Preview::columns`]): the group's rows
    /// processed, the other channels' rows from the raw signal's view.
    fn columns_view(&self, view: &View, out: &mut Envelope) -> DspResult<()> {
        let width = view.width;
        let group = self.preview.columns(view.start..view.end, width)?;
        let outs = self.preview.out_channels();
        let mut values = vec![[f32::NAN; 2]; view.channels.len() * width];
        let mut missing = Vec::new();
        for (k, c) in view.channels.iter().enumerate() {
            match outs.iter().position(|g| g == c) {
                Some(r) => values[k * width..(k + 1) * width].copy_from_slice(&group[r * width..(r + 1) * width]),
                None => missing.push(k),
            }
        }
        let mut complete = true;
        if !missing.is_empty() {
            let raw_view = View { channels: missing.iter().map(|&k| view.channels[k]).collect(), ..view.clone() };
            let mut raw = Envelope::Samples(Vec::new());
            self.raw.view(&raw_view, &mut raw)?;
            match raw {
                Envelope::Columns { values: rows, complete: c } => {
                    complete = c;
                    for (i, &k) in missing.iter().enumerate() {
                        values[k * width..(k + 1) * width].copy_from_slice(&rows[i * width..(i + 1) * width]);
                    }
                }
                Envelope::Samples(_) => return Err(dspk_core::DspError::ComputeError("a zoomed-out raw view answered samples".into())),
            }
        }
        *out = Envelope::Columns { values, complete };
        Ok(())
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

    /// Forgets the chains kept (a batch job takes the GPU: their device memory goes with them).
    pub fn clear(&self) {
        self.made.lock().unwrap_or_else(|e| e.into_inner()).clear();
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
        if n > view.width as u64 {
            return self.columns_view(view, out);
        }
        // Zoomed in: the samples themselves
        let mut samples = vec![0.0f32; rows * n as usize];
        self.fill(&view.channels, view.start..view.end, true, &mut samples)?;
        *out = Envelope::Samples(samples);
        Ok(())
    }

    fn read(&self, channels: &[usize], samples: Range<u64>, out: &mut [f32]) -> DspResult<()> {
        if !self.preview.fits_one_block(samples.end - samples.start) {
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

    fn built(&self, columns: usize) -> Vec<bool> {
        self.raw.built(columns)
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

    /// A zoomed-out processed view is reduced on the device: its columns are exactly the min and
    /// max of the processed samples of the same block (read back here for the check).
    #[test]
    fn zoomed_out_columns_are_the_extremes_of_the_processed_samples() {
        let source = recording(3, 6000);
        let raw: Arc<dyn SignalBackend> = Arc::new(LocalSignal::open(source.clone(), None).unwrap());
        let p = ProcessedSignal::new(raw, source, vec![0, 2], vec![PipelineStage::bandpass(20.0, 200.0), PipelineStage::Scale { alpha: 2.0, beta: 0.5 }]).unwrap();
        let view = View { channels: vec![0, 1, 2], start: 1000, end: 5000, width: 97 };
        let mut env = Envelope::Samples(Vec::new());
        p.view(&view, &mut env).unwrap();
        let Envelope::Columns { values, complete } = env else { panic!("zoomed out: columns") };
        assert!(complete);
        let mut samples = vec![0.0f32; 3 * 4000];
        p.read(&[0, 1, 2], 1000..5000, &mut samples).unwrap();
        let columns = dspk_view::Columns::Even { start: 1000, len: 4000, width: 97 };
        for (k, row) in samples.chunks_exact(4000).enumerate() {
            let mut expected = vec![dspk_view::envelope::EMPTY; 97];
            dspk_view::envelope::fold_row(row, 1000, columns, &mut expected);
            dspk_view::envelope::fold::finish(&mut expected);
            assert_eq!(&values[k * 97..(k + 1) * 97], &expected[..], "channel {k} (1: outside the group, raw)");
        }
    }

    /// A window wider than one block is processed in tiles (each with the chain's halo, reduced
    /// over its interior), and gives the columns of the window processed whole.
    #[test]
    fn tiled_columns_equal_the_window_processed_whole() {
        // Channels unlike each other (the median reference leaves their differences)
        let samples = 40_000;
        let data: Vec<f32> = (0..2 * samples).map(|i| {
            let (c, t) = ((i / samples) as f32, (i % samples) as f32);
            (t * 0.31 * (c + 1.0)).sin() * (c + 1.0) * 40.0 + (t * 0.07).sin() * 10.0 + c * 3.0
        }).collect();
        let source: Arc<dyn RecordingSource> = Arc::new(MemoryRecording::new("test", data, 2, 1000.0).unwrap());
        let raw: Arc<dyn SignalBackend> = Arc::new(LocalSignal::open(source.clone(), None).unwrap());
        let stages = vec![PipelineStage::bandpass(20.0, 200.0), PipelineStage::CommonMedianReference];
        let whole = ProcessedSignal::new(raw.clone(), source.clone(), vec![0, 1], stages.clone()).unwrap();
        // Blocks of 2 × 6000 values: the 30,000-sample window takes several tiles
        let tiled = ProcessedSignal::new(raw, source, vec![0, 1], stages).unwrap().with_max_values(2 * 6000);
        let view = View { channels: vec![0, 1], start: 5000, end: 35_000, width: 211 };
        assert!(!tiled.preview.fits_one_block(30_000) && tiled.covers(30_000), "tiles, not raw");
        let (mut a, mut b) = (Envelope::Samples(Vec::new()), Envelope::Samples(Vec::new()));
        whole.view(&view, &mut a).unwrap();
        tiled.view(&view, &mut b).unwrap();
        let (Envelope::Columns { values: a, .. }, Envelope::Columns { values: b, .. }) = (a, b) else { panic!("columns") };
        let scale = a.iter().flat_map(|v| [v[0].abs(), v[1].abs()]).fold(0.0f32, f32::max);
        assert!(scale > 1.0, "a signal to compare, not rounding noise ({scale})");
        for (x, (p, q)) in a.iter().zip(&b).enumerate() {
            assert!((p[0] - q[0]).abs() <= 1e-3 * scale && (p[1] - q[1]).abs() <= 1e-3 * scale, "column {x}: {p:?} vs {q:?}");
        }
    }

    /// Where a processed view's time goes (the Author's chain, a 10 s window of 32 channels at
    /// 24.4 kHz): `cargo test -p dspk-pipeline --features view --profile validate --
    /// --ignored where_a_processed_view --nocapture`.
    #[test]
    #[ignore = "a measurement, not a check"]
    fn where_a_processed_view_spends_its_time() {
        let (channels, rate, seconds) = (32usize, 24_414.0625f64, 20.0f64);
        let samples = (rate * seconds) as usize;
        let data: Vec<f32> = (0..channels * samples).map(|i| ((i % samples) as f32 * 0.013).sin() * 50.0 + ((i * 7919) % 1013) as f32 * 0.1).collect();
        let source: Arc<dyn RecordingSource> = Arc::new(MemoryRecording::new("bench", data, channels, rate).unwrap());
        let raw: Arc<dyn SignalBackend> = Arc::new(LocalSignal::open(source.clone(), None).unwrap());
        let stages = vec![PipelineStage::bandpass(300.0, 6000.0), PipelineStage::lowpass(300.0), PipelineStage::teager_kaiser(), PipelineStage::fir(vec![0.2; 5])];
        let all: Vec<usize> = (0..channels).collect();
        let p = ProcessedSignal::new(raw, source, all.clone(), stages).unwrap();
        let window = (rate * 10.0) as u64;
        let time = |label: &str, f: &mut dyn FnMut()| {
            let t = std::time::Instant::now();
            f();
            println!("{label:<52} {:8.1} ms", t.elapsed().as_secs_f64() * 1e3);
        };
        let mut env = Envelope::Samples(Vec::new());
        let view = |start: u64| View { channels: all.clone(), start, end: start + window, width: 1200 };
        time("first zoomed-out view (read, upload, chain, reduce)", &mut || p.view(&view(0), &mut env).unwrap());
        time("the same view again (kept columns)", &mut || p.view(&view(0), &mut env).unwrap());
        time("a pan inside the kept block (reduce only)", &mut || p.view(&view(window / 8), &mut env).unwrap());
        let mut out = vec![0.0f32; channels * 2000];
        time("a zoomed-in read of 2000 samples (download)", &mut || p.read(&all, window / 2..window / 2 + 2000, &mut out).unwrap());
        // The browser's block size: the 20 s window in tiles of 16 Mi values
        let tiled = p.with_max_values(dspk_view::PROCESSED_MAX_VALUES_BROWSER);
        let wide = View { channels: all.clone(), start: 0, end: 2 * window, width: 1200 };
        time("a 20 s view in tiles of 16 Mi values", &mut || tiled.view(&wide, &mut env).unwrap());
        // The same with a one-step chain (a scale: almost no kernel time): the rest is reading
        // and uploading
        let source2 = tiled.preview.source();
        let one = ProcessedSignal::new(Arc::new(LocalSignal::open(source2.clone(), None).unwrap()), source2, all.clone(), vec![PipelineStage::Scale { alpha: 1.0, beta: 0.0 }]).unwrap().with_max_values(dspk_view::PROCESSED_MAX_VALUES_BROWSER);
        time("the same 20 s with a one-step chain (a scale; first)", &mut || one.view(&wide, &mut env).unwrap());
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
