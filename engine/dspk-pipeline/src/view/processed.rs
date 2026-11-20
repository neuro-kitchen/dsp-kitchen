//! Live preview of a view's op chain as a [`SignalBackend`]: the views are answered with the
//! recording processed on the device by [`Preview`](super::Preview) (window by window, cached), the
//! raw signal for channels outside the view's group and for windows zoomed out past
//! [`PREVIEW_MAX_VALUES`](super::PREVIEW_MAX_VALUES) ([`ProcessedSignal::covers`] tells the UI to say so).

use std::ops::Range;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use dspk_core::{DspResult, RecordingInfo, RecordingSource, SampleFormat};
use dspk_ops::PipelineStage;
use super::budget::DeviceBudget;
use super::preview::{InputCache, Preview, PreviewTimings};
use dspk_view::{Envelope, OnProgress, Progress, Pyramid, SignalBackend, View};


/// A view's signal with its op chain applied: [`Preview`] (dspk-pipeline) for the channels the
/// chain processes, the raw signal for the others and for windows too large to preview.
pub struct ProcessedSignal {
    raw: Arc<dyn SignalBackend>,
    preview: Arc<Preview>,
}

/// `preview` shared, entered in its device budget.
fn shared(preview: Preview) -> Arc<Preview> {
    let preview = Arc::new(preview);
    Preview::register(&preview);
    preview
}

impl ProcessedSignal {
    /// The preview of `stages` over `group` (the view's channels) of `source`; `raw` answers what
    /// is not processed (outside the group, or zoomed out past [`PREVIEW_MAX_VALUES`](super::PREVIEW_MAX_VALUES)).
    pub fn new(raw: Arc<dyn SignalBackend>, source: Arc<dyn RecordingSource>, group: Vec<usize>, stages: Vec<PipelineStage>) -> DspResult<Self> {
        Ok(Self { raw, preview: shared(Preview::new(source, group, stages)?) })
    }

    /// [`Self::new`] charged to `budget` instead of the process's device budget.
    pub fn with_budget(raw: Arc<dyn SignalBackend>, source: Arc<dyn RecordingSource>, group: Vec<usize>, stages: Vec<PipelineStage>, cache: Arc<InputCache>, budget: Arc<DeviceBudget>) -> DspResult<Self> {
        Ok(Self { raw, preview: shared(Preview::new(source, group, stages)?.with_input_cache(cache).with_budget(budget)) })
    }

    /// Attaches a resumable min/max [`Pyramid`] (mmapped `drv/<drv_uuid>_drv_<source>.minmax` or
    /// in-memory) to hold the final op's zoomed-out overview.
    pub fn attach_pyramid(&self, pyramid: Arc<Pyramid>) {
        self.preview.attach_pyramid(pyramid);
    }

    /// Whether this chain's processed pyramid is built completely.
    pub fn frozen(&self) -> bool {
        self.preview.is_pyramid_complete()
    }

    /// The device budget this signal is charged to.
    pub fn budget(&self) -> &Arc<DeviceBudget> {
        self.preview.budget()
    }

    /// [`Self::new`] reading the recording through `cache` (shared by a recording's chains): an
    /// edited chain over a window already uploaded does not read and upload it again.
    pub fn with_input_cache(raw: Arc<dyn SignalBackend>, source: Arc<dyn RecordingSource>, group: Vec<usize>, stages: Vec<PipelineStage>, cache: Arc<InputCache>) -> DspResult<Self> {
        Ok(Self { raw, preview: shared(Preview::new(source, group, stages)?.with_input_cache(cache)) })
    }

    /// A branch: `stages` run on `prefix`'s output on the device (the prefix is computed once for
    /// both, and nothing is read back in between). Its rows are the prefix's output rows.
    pub fn after(prefix: &ProcessedSignal, stages: Vec<PipelineStage>) -> DspResult<Self> {
        Ok(Self { raw: prefix.raw.clone(), preview: shared(Preview::after(prefix.preview.clone(), stages)?) })
    }

    /// Where the last view's time went (read, upload, chain, read back; per step in measuring
    /// mode, `DSPK_PREVIEW_TIMING=1`).
    pub fn last_timings(&self) -> PreviewTimings {
        self.preview.last_timings()
    }

    /// The same with blocks and tiles of at most `max_values` values ([`Preview::with_max_values`]).
    pub fn with_max_values(self, max_values: usize) -> Self {
        let preview = Arc::try_unwrap(self.preview).unwrap_or_else(|_| panic!("with_max_values before the signal is shared"));
        Self { raw: self.raw, preview: shared(preview.with_max_values(max_values)) }
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
            self.preview.want_overview(&(view.start..view.end));
            return Ok(());
        }
        // Zoomed out: reduced on the device, only the columns read back
        if view.samples() > view.width as u64 {
            return self.preview.columns_async(view.start..view.end, view.width).await.map(|_| ());
        }
        self.preview.warm_async(view.start..view.end, true).await
    }

    /// A zoomed-out `view` from the device's columns ([`Preview::columns_cancellable`]): the
    /// group's rows processed, the other channels' rows from the raw signal's view.
    fn columns_view(&self, view: &View, out: &mut Envelope, cancel: Option<&AtomicBool>) -> DspResult<()> {
        let width = view.width;
        let group = self.preview.columns_cancellable(view.start..view.end, width, cancel)?;
        let outs = self.preview.out_channels();
        let mut values = vec![[f32::NAN; 2]; view.channels.len() * width];
        let mut missing = Vec::new();
        for (k, c) in view.channels.iter().enumerate() {
            match outs.iter().position(|g| g == c) {
                Some(r) => values[k * width..(k + 1) * width].copy_from_slice(&group[r * width..(r + 1) * width]),
                None => missing.push(k),
            }
        }
        // A pyramid still being built leaves columns to fill in: the viewer asks again on progress
        let mut complete = self.preview.columns_complete();
        if !missing.is_empty() {
            let raw_view = View { channels: missing.iter().map(|&k| view.channels[k]).collect(), ..view.clone() };
            let mut raw = Envelope::Samples(Vec::new());
            self.raw.view(&raw_view, &mut raw)?;
            match raw {
                Envelope::Columns { values: rows, complete: c } => {
                    complete &= c;
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
        self.view_with_cancel(view, out, None)
    }

    fn view_cancellable(&self, view: &View, out: &mut Envelope, cancel: &AtomicBool) -> DspResult<()> {
        self.view_with_cancel(view, out, Some(cancel))
    }

    fn read(&self, channels: &[usize], samples: Range<u64>, out: &mut [f32]) -> DspResult<()> {
        if !self.preview.fits_one_block(samples.end - samples.start) {
            return self.raw.read(channels, samples, out);
        }
        self.fill(channels, samples, true, out)
    }

    fn focus(&self, sample: u64) {
        self.raw.focus(sample);
        self.preview.focus_pyramid(sample);
    }

    fn prepare(&self, start: u64, end: u64) -> DspResult<()> {
        self.raw.prepare(start, end)
    }

    fn progress(&self) -> Progress {
        self.preview.pyramid_progress().unwrap_or_else(|| self.raw.progress())
    }

    fn built(&self, columns: usize) -> Vec<bool> {
        self.preview.pyramid_built(columns).unwrap_or_else(|| self.raw.built(columns))
    }

    fn watch(&self, on_progress: OnProgress) {
        if self.preview.pyramid().is_some() {
            self.preview.watch(on_progress);
        } else {
            self.raw.watch(on_progress);
        }
    }
}

impl ProcessedSignal {
    fn view_with_cancel(&self, view: &View, out: &mut Envelope, cancel: Option<&AtomicBool>) -> DspResult<()> {
        if cancel.is_some_and(|c| c.load(Ordering::Relaxed)) {
            return Ok(());
        }
        let n = view.samples();
        if !self.covers(n) {
            self.preview.want_overview(&(view.start..view.end));
            return self.raw.view(view, out);
        }
        view.validate(self.info().channel_count(), self.info().samples)?;
        let rows = view.channels.len();
        if n > view.width as u64 {
            return self.columns_view(view, out, cancel);
        }
        // Zoomed in: the samples themselves
        let mut samples = vec![0.0f32; rows * n as usize];
        self.fill(&view.channels, view.start..view.end, true, &mut samples)?;
        *out = Envelope::Samples(samples);
        Ok(())
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

    /// An edited chain over the window on screen reuses the window already uploaded (it is not
    /// read again), and its view equals the same chain made from nothing, bit for bit.
    #[test]
    fn an_edit_reuses_the_uploaded_window_and_equals_a_fresh_chain() {
        use crate::view::{Made, ProcessedChains, ViewPlan};
        let source = recording(2, 8000);
        let raw: Arc<dyn SignalBackend> = Arc::new(LocalSignal::open(source.clone(), None).unwrap());
        let plan = |alpha: f32| ViewPlan {
            group: vec![0, 1],
            steps: vec![("bandpass".into(), PipelineStage::bandpass(20.0, 200.0)), (format!("scale {alpha}"), PipelineStage::Scale { alpha, beta: 0.0 }), ("tk".into(), PipelineStage::teager_kaiser())],
            shown: vec![0, 1],
            fits: Vec::new(),
        };
        let signal = |chains: &ProcessedChains, alpha: f32, editing: Option<usize>| match chains.signal_editing(raw.clone(), source.clone(), "main", &plan(alpha), editing).unwrap() {
            Made::Ready(s) => s,
            Made::NeedsFit(_) => panic!("no whitening"),
        };
        let view = View { channels: vec![0, 1], start: 2000, end: 6000, width: 64 };
        let columns = |s: &ProcessedSignal| {
            let mut env = Envelope::Samples(Vec::new());
            s.view(&view, &mut env).unwrap();
            let Envelope::Columns { values, .. } = env else { panic!("columns") };
            values
        };
        let chains = ProcessedChains::new("");
        let first = signal(&chains, 2.0, Some(1));
        let _ = columns(&first);
        assert!(!first.last_timings().input_reused, "the first chain reads the window");
        // The scale dragged: the chain up to it is kept, the window is not read again
        let edited = signal(&chains, 3.0, Some(1));
        let got = columns(&edited);
        // The same chain made from nothing (the same structure: a prefix, then the branch)
        let fresh = signal(&ProcessedChains::new(""), 3.0, Some(1));
        assert_eq!(got, columns(&fresh), "an edit computed from what was kept equals the chain made from nothing");
        // And the whole chain in one block, within the window processing's halo tolerance
        let whole = columns(&signal(&ProcessedChains::new(""), 3.0, None));
        let scale = whole.iter().flat_map(|v| [v[0].abs(), v[1].abs()]).fold(0.0f32, f32::max).max(1.0);
        for (x, (p, q)) in got.iter().zip(&whole).enumerate() {
            assert!((p[0] - q[0]).abs() <= 1e-3 * scale && (p[1] - q[1]).abs() <= 1e-3 * scale, "column {x}: {p:?} vs {q:?}");
        }
    }

    /// Chains and windows past the device budget: what the views keep stays within it (others
    /// are released, off screen first), and a released chain made again equals what it gave.
    #[test]
    fn views_stay_within_the_device_budget_and_a_released_chain_comes_back_the_same() {
        use crate::view::{DeviceBudget, Made, ProcessedChains, ViewPlan};
        let source = recording(4, 400_000);
        let raw: Arc<dyn SignalBackend> = Arc::new(LocalSignal::open(source.clone(), None).unwrap());
        let limit = 64u64 << 20;
        let chains = ProcessedChains::with_budget("", DeviceBudget::with_limit(limit));
        let plan = |alpha: f32| ViewPlan {
            group: vec![0, 1, 2, 3],
            steps: vec![("bandpass".into(), PipelineStage::bandpass(20.0, 200.0)), (format!("scale {alpha}"), PipelineStage::Scale { alpha, beta: 0.0 }), ("tk".into(), PipelineStage::teager_kaiser())],
            shown: vec![0, 1, 2, 3],
            fits: Vec::new(),
        };
        let columns = |alpha: f32, start: u64, end: u64| {
            let Made::Ready(s) = chains.signal(raw.clone(), source.clone(), "main", &plan(alpha)).unwrap() else { panic!("no whitening") };
            let mut env = Envelope::Samples(Vec::new());
            s.view(&View { channels: vec![0, 1, 2, 3], start, end, width: 128 }, &mut env).unwrap();
            let Envelope::Columns { values, .. } = env else { panic!("columns") };
            values
        };
        let first = columns(1.0, 0, 50_000);
        // Six chains (fewer than are kept), each over a wider window
        for (i, alpha) in [2.0f32, 3.0, 4.0, 5.0, 6.0].into_iter().enumerate() {
            let _ = columns(alpha, 0, 80_000 * (i as u64 + 1));
            let kept = chains.budget().charged();
            assert!(kept <= limit, "chain {alpha}: {kept} bytes kept over a budget of {limit}");
        }
        assert!(chains.budget().released() > 0, "the wider windows needed room: some chain was released");
        // The first chain was released long ago; asked again, it is made again, the same
        assert_eq!(columns(1.0, 0, 50_000), first, "a chain released and made again gives what it gave");
    }

    /// A chain left unchanged is frozen: the whole recording processed once into buckets, and a
    /// zoomed-out view then drawn from them holds what the view processed directly holds (each
    /// column reaches at most one bucket past its edges), with the same extremes overall.
    #[test]
    fn a_chain_left_unchanged_is_frozen_and_its_summary_holds_the_processed_view() {
        use crate::view::{Made, ProcessedChains, ViewPlan, SUMMARY_BUCKET};
        let source = recording(3, 400_000);
        let raw: Arc<dyn SignalBackend> = Arc::new(LocalSignal::open(source.clone(), None).unwrap());
        let chains = ProcessedChains::new("");
        let plan = ViewPlan {
            group: vec![0, 1, 2],
            steps: vec![("bandpass".into(), PipelineStage::bandpass(20.0, 200.0)), ("abs".into(), PipelineStage::Elementwise(dspk_ops::math::Elementwise::Abs))],
            shown: vec![0, 1, 2],
            fits: Vec::new(),
        };
        let Made::Ready(signal) = chains.signal(raw.clone(), source.clone(), "main", &plan).unwrap() else { panic!("no whitening") };
        let view = View { channels: vec![0, 1, 2], start: 10_000, end: 300_000, width: 200 };
        let columns = |s: &ProcessedSignal| {
            let mut env = Envelope::Samples(Vec::new());
            s.view(&view, &mut env).unwrap();
            let Envelope::Columns { values, .. } = env else { panic!("columns") };
            values
        };
        let direct = columns(&signal);
        // Left as it is, no other chain asked for: frozen in the background
        let t = std::time::Instant::now();
        while !signal.frozen() {
            assert!(t.elapsed().as_secs() < 120, "not frozen within two minutes");
            std::thread::sleep(std::time::Duration::from_millis(200));
            // Drawn again in the first seconds (other tests ask for other chains meanwhile): still
            // the one on screen when the freeze delay ends; then left alone, as a view left as it is
            if t.elapsed().as_secs_f64() < 2.0 {
                let _ = columns(&signal);
            }
        }
        assert!((view.end - view.start) / view.width as u64 >= SUMMARY_BUCKET);
        let frozen = columns(&signal);
        let scale = direct.iter().flat_map(|v| [v[0].abs(), v[1].abs()]).fold(0.0f32, f32::max).max(1.0);
        let tol = 1e-3 * scale;
        // Pyramid::envelope aligns column edges to buckets (moving by < 1 column so every sample
        // falls in exactly one column): each direct column's extremes lie within its column or a
        // neighbour in the pyramid.
        let w = view.width;
        for ch in 0..3 {
            let d_row = &direct[ch * w..(ch + 1) * w];
            let f_row = &frozen[ch * w..(ch + 1) * w];
            for x in 0..w {
                let lo = f_row[x.saturating_sub(1)..=(x + 1).min(w - 1)].iter().map(|v| v[0]).fold(f32::INFINITY, f32::min);
                let hi = f_row[x.saturating_sub(1)..=(x + 1).min(w - 1)].iter().map(|v| v[1]).fold(f32::NEG_INFINITY, f32::max);
                assert!(lo <= d_row[x][0] + tol && hi >= d_row[x][1] - tol, "ch {ch} col {x}: pyramid [{lo}, {hi}] does not hold {:?}", d_row[x]);
            }
        }
        let extremes = |v: &[[f32; 2]]| v.iter().fold([f32::INFINITY, f32::NEG_INFINITY], |a, b| [a[0].min(b[0]), a[1].max(b[1])]);
        let (a, b) = (extremes(&direct), extremes(&frozen));
        assert!((a[0] - b[0]).abs() <= tol && (a[1] - b[1]).abs() <= tol, "{a:?} vs {b:?}");
    }

    /// Chains of one recording share the window uploaded: a chain with another first op (an op
    /// added, removed or changed at the start) over the window on screen reads nothing again.
    #[test]
    fn a_new_chain_over_the_window_on_screen_reuses_its_upload() {
        use crate::view::{Made, ProcessedChains, ViewPlan};
        let source = recording(2, 20_000);
        let raw: Arc<dyn SignalBackend> = Arc::new(LocalSignal::open(source.clone(), None).unwrap());
        let chains = ProcessedChains::new("");
        let plan = |steps: Vec<(String, PipelineStage)>| ViewPlan { group: vec![0, 1], steps, shown: vec![0, 1], fits: Vec::new() };
        let view = View { channels: vec![0, 1], start: 4000, end: 12_000, width: 64 };
        let timings = |p: ViewPlan| {
            let Made::Ready(s) = chains.signal(raw.clone(), source.clone(), "main", &p).unwrap() else { panic!("no whitening") };
            let mut env = Envelope::Samples(Vec::new());
            s.view(&view, &mut env).unwrap();
            s.last_timings()
        };
        let first = timings(plan(vec![("bandpass".into(), PipelineStage::bandpass(20.0, 200.0))]));
        assert!(!first.input_reused);
        let added = timings(plan(vec![("scale".into(), PipelineStage::Scale { alpha: 2.0, beta: 0.0 }), ("bandpass".into(), PipelineStage::bandpass(20.0, 200.0))]));
        assert!(added.input_reused, "a chain with another first op over the same window reuses its upload: {added:?}");
        assert_eq!(added.read, 0.0, "nothing read again");
    }

    /// Where adding an op to the view on screen spends its time (32 channels, 10 s at 24.4 kHz).
    #[test]
    #[ignore = "a measurement, not a check"]
    fn where_adding_an_op_spends_its_time() {
        use crate::view::{Made, ProcessedChains, ViewPlan};
        let (channels, rate) = (32usize, 24_414.0625f64);
        let samples = (rate * 60.0) as usize;
        let data: Vec<f32> = (0..channels * samples).map(|i| ((i % samples) as f32 * 0.013).sin() * 50.0 + ((i * 7919) % 1013) as f32 * 0.1).collect();
        let source: Arc<dyn RecordingSource> = Arc::new(MemoryRecording::new("bench", data, channels, rate).unwrap());
        let raw: Arc<dyn SignalBackend> = Arc::new(LocalSignal::open(source.clone(), None).unwrap());
        let chains = ProcessedChains::new("");
        let all: Vec<usize> = (0..channels).collect();
        let view = View { channels: all.clone(), start: 0, end: (rate * 10.0) as u64, width: 1200 };
        let mut steps: Vec<(String, PipelineStage)> = Vec::new();
        for (name, stage) in [("bandpass", PipelineStage::bandpass(300.0, 6000.0)), ("scale", PipelineStage::Scale { alpha: 2.0, beta: 0.0 }), ("tk", PipelineStage::teager_kaiser()), ("abs", PipelineStage::Elementwise(dspk_ops::math::Elementwise::Abs))] {
            steps.push((name.into(), stage));
            let plan = ViewPlan { group: all.clone(), steps: steps.clone(), shown: all.clone(), fits: Vec::new() };
            let t = std::time::Instant::now();
            let Made::Ready(s) = chains.signal(raw.clone(), source.clone(), "main", &plan).unwrap() else { panic!("no whitening") };
            let mut env = Envelope::Samples(Vec::new());
            s.view(&view, &mut env).unwrap();
            println!("{:<44} {:>8.1} ms", format!("+ {name} ({} ops)", steps.len()), t.elapsed().as_secs_f64() * 1e3);
            println!("    {:?}", s.last_timings());
            let t = std::time::Instant::now();
            s.view(&view, &mut env).unwrap();
            println!("{:<44} {:>8.3} ms", "  the same view again", t.elapsed().as_secs_f64() * 1e3);
        }
    }

    /// The Author's chain (band-pass, Teager-Kaiser, a 20 Hz low-pass) over a one-minute window of
    /// 32 channels at 24.4 kHz: drawn, not blank (a page showed it white).
    #[test]
    #[ignore = "a GPU-sized check: run with DSPK_RUNTIME=wgpu"]
    fn a_one_minute_window_of_the_authors_chain_has_values() {
        use crate::view::{Made, ProcessedChains, ViewPlan};
        let (channels, rate) = (32usize, 24_414.0625f64);
        let samples = (rate * 120.0) as usize;
        let data: Vec<f32> = (0..channels * samples).map(|i| ((i % samples) as f32 * 0.013).sin() * 50.0 + ((i * 7919) % 1013) as f32 * 0.1).collect();
        let source: Arc<dyn RecordingSource> = Arc::new(MemoryRecording::new("minute", data, channels, rate).unwrap());
        let raw: Arc<dyn SignalBackend> = Arc::new(LocalSignal::open(source.clone(), None).unwrap());
        let chains = ProcessedChains::new("");
        let all: Vec<usize> = (0..channels).collect();
        let steps = vec![("bp".into(), PipelineStage::bandpass(300.0, 6000.0)), ("tk".into(), PipelineStage::teager_kaiser()), ("lp".into(), PipelineStage::lowpass(20.0))];
        let plan = ViewPlan { group: all.clone(), steps, shown: all.clone(), fits: Vec::new() };
        let Made::Ready(s) = chains.signal(raw, source, "main", &plan).unwrap() else { panic!("no whitening") };
        let start = (rate * 30.0) as u64;
        let view = View { channels: (0..8).collect(), start, end: start + (rate * 60.0) as u64, width: 1200 };
        println!("covers {}, one block {}", s.covers(view.samples()), s.preview.fits_one_block(view.samples()));
        let t = std::time::Instant::now();
        let mut env = Envelope::Samples(Vec::new());
        s.view(&view, &mut env).unwrap();
        println!("{:.1} ms {:?}", t.elapsed().as_secs_f64() * 1e3, s.last_timings());
        let Envelope::Columns { values, complete } = env else { panic!("columns") };
        let finite = values.iter().filter(|v| v[0].is_finite() && v[1].is_finite()).count();
        println!("{finite} of {} columns hold values, complete {complete}", values.len());
        assert!(finite * 10 >= values.len() * 9, "{finite} of {} columns hold values", values.len());
    }

    /// A chain ending in a low-pass runs it at a lower rate; what it shows is the chain run at the
    /// full rate: zoomed in, within 2 % of the scale (samples interpolated from the low rate);
    /// zoomed out, each column's extremes within 3 low-rate samples of the column's edges.
    #[test]
    fn a_low_pass_run_at_a_lower_rate_equals_it_at_the_full_rate() {
        let (channels, rate) = (3usize, 24_414.0625f64);
        let samples = (rate * 40.0) as usize;
        // Bursts of broadband activity, modulated at 1.3 Hz: an envelope to follow
        let data: Vec<f32> = (0..channels * samples)
            .map(|i| {
                let (c, t) = (i / samples, (i % samples) as f64 / rate);
                let carrier = ((i as u64).wrapping_mul(2654435761) % 1000) as f64 / 500.0 - 1.0;
                let envelope = 1.0 + (2.0 * std::f64::consts::PI * 1.3 * t + c as f64).sin();
                (carrier * envelope * 40.0 + (2.0 * std::f64::consts::PI * 1500.0 * t).sin() * 20.0 * envelope) as f32
            })
            .collect();
        let source: Arc<dyn RecordingSource> = Arc::new(MemoryRecording::new("envelope", data, channels, rate).unwrap());
        let raw: Arc<dyn SignalBackend> = Arc::new(LocalSignal::open(source.clone(), None).unwrap());
        let stages = vec![PipelineStage::bandpass(300.0, 6000.0), PipelineStage::teager_kaiser(), PipelineStage::lowpass(20.0)];
        let low = ProcessedSignal::new(raw.clone(), source.clone(), vec![0, 1, 2], stages.clone()).unwrap();
        let full = ProcessedSignal { raw, preview: shared(Preview::new_full_rate(source, vec![0, 1, 2], stages).unwrap()) };
        assert_eq!(low.preview.rate_factor(), 128, "20 Hz at 24.4 kHz: the low-pass at 191 Hz");
        assert_eq!(full.preview.rate_factor(), 1);
        let compare = |view: View, what: &str| {
            let (mut a, mut b) = (Envelope::Samples(Vec::new()), Envelope::Samples(Vec::new()));
            low.view(&view, &mut a).unwrap();
            full.view(&view, &mut b).unwrap();
            let flat = |e: Envelope| match e {
                Envelope::Columns { values, .. } => values.into_iter().flatten().collect::<Vec<f32>>(),
                Envelope::Samples(v) => v,
            };
            let (a, b) = (flat(a), flat(b));
            let scale = b.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-6);
            let (at, worst) = a.iter().zip(&b).map(|(p, q)| (p - q).abs()).enumerate().fold((0, 0.0f32), |m, (i, d)| if d > m.1 { (i, d) } else { m });
            let near = |v: &[f32]| v[at.saturating_sub(4)..(at + 5).min(v.len())].to_vec();
            assert!(worst <= 0.02 * scale, "{what}: {worst} of {scale} at value {at} of {}: lower rate {:?}, full rate {:?}", a.len(), near(&a), near(&b));
        };
        let all: Vec<usize> = (0..channels).collect();
        compare(View { channels: all.clone(), start: (rate * 10.0) as u64, end: (rate * 10.0) as u64 + 2000, width: 2000 }, "zoomed in");
        // Zoomed out, a column of the lower rate holds the full-rate signal's extremes over the
        // column to within 3 low-rate samples of its edges (its edges in the low rate's samples
        // fall up to 2 of them early, and its last sample up to one more before its edge): between
        // the full rate's [min, max] over the column shrunk by 3·M on each side and grown by 3·M
        let (start, end, width) = ((rate * 5.0) as u64, (rate * 35.0) as u64, 300usize);
        let mut env = Envelope::Samples(Vec::new());
        low.view(&View { channels: all.clone(), start, end, width }, &mut env).unwrap();
        let Envelope::Columns { values: cols, .. } = env else { panic!("columns") };
        let reach = 3 * low.preview.rate_factor() as u64;
        let (from, to) = (start - reach, end + reach);
        let mut whole = vec![0f32; channels * (to - from) as usize];
        full.read_processed(&all, from..to, &mut whole).unwrap();
        let len = (to - from) as usize;
        let scale = whole.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let extremes = |c: usize, a: u64, b: u64| {
            let row = &whole[c * len + (a - from) as usize..c * len + (b - from) as usize];
            row.iter().fold([f32::INFINITY, f32::NEG_INFINITY], |m, v| [m[0].min(*v), m[1].max(*v)])
        };
        for c in 0..channels {
            for x in 0..width {
                let (c0, c1) = (start + x as u64 * (end - start) / width as u64, start + (x as u64 + 1) * (end - start) / width as u64);
                let (outer, inner) = (extremes(c, c0 - reach, c1 + reach), extremes(c, c0 + reach, c1 - reach));
                let [lo, hi] = cols[c * width + x];
                let tol = 0.005 * scale;
                assert!(lo >= outer[0] - tol && lo <= inner[0] + tol, "channel {c} column {x}: min {lo} outside [{}, {}]", outer[0], inner[0]);
                assert!(hi <= outer[1] + tol && hi >= inner[1] - tol, "channel {c} column {x}: max {hi} outside [{}, {}]", inner[1], outer[1]);
            }
        }
    }

    /// Two views of one chain (traces of some channels, a heatmap of all) asking for the same
    /// tiled window at once: neither stops the other, and both get the window's columns.
    #[test]
    fn two_views_of_one_chain_asking_at_once_both_get_their_columns() {
        let samples = 60_000;
        let data: Vec<f32> = (0..3 * samples).map(|i| ((i % samples) as f32 * 0.21).sin() * 30.0 + (i / samples) as f32 * 5.0).collect();
        let source: Arc<dyn RecordingSource> = Arc::new(MemoryRecording::new("two views", data, 3, 1000.0).unwrap());
        let raw: Arc<dyn SignalBackend> = Arc::new(LocalSignal::open(source.clone(), None).unwrap());
        let stages = vec![PipelineStage::bandpass(20.0, 200.0), PipelineStage::teager_kaiser()];
        let whole = ProcessedSignal::new(raw.clone(), source.clone(), vec![0, 1, 2], stages.clone()).unwrap();
        // Blocks of 3 × 20,000 values: the 50,000-sample window takes 3 tiles (within the tiles
        // processed exactly), where a stale request stops
        let tiled = Arc::new(ProcessedSignal::new(raw, source, vec![0, 1, 2], stages).unwrap().with_max_values(3 * 20_000));
        let range = (5_000u64, 55_000u64);
        assert!(!tiled.preview.fits_one_block(range.1 - range.0) && tiled.covers(range.1 - range.0));
        let views = [vec![0usize, 1], vec![0, 1, 2]];
        let answers: Vec<Vec<[f32; 2]>> = std::thread::scope(|scope| {
            let handles: Vec<_> = views
                .iter()
                .map(|channels| {
                    let (tiled, channels) = (tiled.clone(), channels.clone());
                    scope.spawn(move || {
                        let mut env = Envelope::Samples(Vec::new());
                        tiled.view(&View { channels, start: range.0, end: range.1, width: 160 }, &mut env).expect("neither view stops the other");
                        let Envelope::Columns { values, .. } = env else { panic!("columns") };
                        values
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });
        let mut env = Envelope::Samples(Vec::new());
        whole.view(&View { channels: vec![0, 1, 2], start: range.0, end: range.1, width: 160 }, &mut env).unwrap();
        let Envelope::Columns { values: expected, .. } = env else { panic!("columns") };
        let scale = expected.iter().flat_map(|v| [v[0].abs(), v[1].abs()]).fold(0.0f32, f32::max);
        for (k, got) in answers.iter().enumerate() {
            let rows = views[k].len();
            for (x, (p, q)) in got.iter().zip(&expected[..rows * 160]).enumerate() {
                assert!((p[0] - q[0]).abs() <= 1e-3 * scale && (p[1] - q[1]).abs() <= 1e-3 * scale, "view {k} column {x}: {p:?} vs {q:?}");
            }
        }
    }

    /// A branch read on the device equals its whole chain (as the host branch does).
    #[test]
    fn a_branch_on_the_device_equals_its_whole_chain() {
        let source = recording(2, 6000);
        let raw: Arc<dyn SignalBackend> = Arc::new(LocalSignal::open(source.clone(), None).unwrap());
        let prefix = ProcessedSignal::new(raw.clone(), source.clone(), vec![0, 1], vec![PipelineStage::bandpass(20.0, 200.0), PipelineStage::Elementwise(dspk_ops::math::Elementwise::Abs)]).unwrap();
        let branch = ProcessedSignal::after(&prefix, vec![PipelineStage::Scale { alpha: 3.0, beta: 0.0 }]).unwrap();
        let whole = ProcessedSignal::new(raw, source, vec![0, 1], vec![PipelineStage::bandpass(20.0, 200.0), PipelineStage::Elementwise(dspk_ops::math::Elementwise::Abs), PipelineStage::Scale { alpha: 3.0, beta: 0.0 }]).unwrap();
        let (mut a, mut b) = (vec![0.0f32; 2 * 400], vec![0.0f32; 2 * 400]);
        branch.read(&[0, 1], 2000..2400, &mut a).unwrap();
        whole.read(&[0, 1], 2000..2400, &mut b).unwrap();
        for i in 0..800 {
            assert!((a[i] - b[i]).abs() < 1e-3 * b[i].abs().max(1.0), "{i}: {} vs {}", a[i], b[i]);
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
        // Blocks of 2 × 12,000 values: the 30,000-sample window takes a few tiles (within the
        // tiles processed exactly, `PREVIEW_MAX_TILES`)
        let tiled = ProcessedSignal::new(raw, source, vec![0, 1], stages).unwrap().with_max_values(2 * 12_000);
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
        println!("    {:?}", p.last_timings());
        time("the same view again (kept columns)", &mut || p.view(&view(0), &mut env).unwrap());
        time("a pan inside the kept block (reduce only)", &mut || p.view(&view(window / 8), &mut env).unwrap());
        println!("    {:?}", p.last_timings());
        let mut out = vec![0.0f32; channels * 2000];
        time("a zoomed-in read of 2000 samples (download)", &mut || p.read(&all, window / 2..window / 2 + 2000, &mut out).unwrap());
        // The browser's block size: the 20 s window in tiles of 16 Mi values
        let tiled = p.with_max_values(crate::view::PREVIEW_MAX_VALUES_BROWSER);
        let wide = View { channels: all.clone(), start: 0, end: 2 * window, width: 1200 };
        time("a 20 s view in tiles of 16 Mi values", &mut || tiled.view(&wide, &mut env).unwrap());
        println!("    {:?}", tiled.last_timings());
        // The same with a one-step chain (a scale: almost no kernel time): the rest is reading
        // and uploading
        let source2 = tiled.preview.source();
        let one = ProcessedSignal::new(Arc::new(LocalSignal::open(source2.clone(), None).unwrap()), source2, all.clone(), vec![PipelineStage::Scale { alpha: 1.0, beta: 0.0 }]).unwrap().with_max_values(crate::view::PREVIEW_MAX_VALUES_BROWSER);
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

    /// A named derived chain (`signal_for_drv`) runs flat (constant VRAM even with 4+ ops, keeping
    /// LowRate active), persists its mmapped pyramid under `<rec>.cache/drv/<uuid>_drv_<source>.minmax`,
    /// reopens complete on the next session without recomputing, and resets in place when edited.
    #[test]
    fn derived_chain_persists_drv_minmax_and_resets_in_place_on_edit() {
        use crate::view::{Made, ProcessedChains, ViewPlan};
        let dir = std::env::temp_dir().join(format!("dspk_drv_test_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let rec_path = dir.join("rec.bin");
        std::fs::write(&rec_path, b"placeholder").unwrap();

        let source = recording(2, 120_000);
        let raw: Arc<dyn SignalBackend> = Arc::new(LocalSignal::open(source.clone(), None).unwrap());
        let drv_id = dspk_view::new_uuid();
        let drv_file = dspk_view::drv_pyramid_path(&rec_path, "main", &drv_id);

        let make_plan = |alpha: f32| ViewPlan {
            group: vec![0, 1],
            steps: vec![
                ("bp".into(), PipelineStage::bandpass(20.0, 200.0)),
                ("cmr".into(), PipelineStage::CommonMedianReference),
                ("tk".into(), PipelineStage::teager_kaiser()),
                (format!("scale {alpha}"), PipelineStage::Scale { alpha, beta: 0.0 }),
            ],
            shown: vec![0, 1],
            fits: Vec::new(),
        };

        // Session 1: build the 4-op derived chain and wait for its background pyramid to finish
        let chains1 = ProcessedChains::new(&dir);
        let Made::Ready(sig1) = chains1.signal_for_drv(&drv_id, Some(&rec_path), raw.clone(), source.clone(), "main", &make_plan(2.0)).unwrap() else {
            panic!("no whitening");
        };
        assert!(drv_file.exists(), "drv minmax file created at {}", drv_file.display());
        sig1.focus(60_000);
        let t0 = std::time::Instant::now();
        while !sig1.frozen() {
            assert!(t0.elapsed().as_secs() < 60, "derived pyramid did not finish in time");
            std::thread::sleep(std::time::Duration::from_millis(50));
        }
        let view = View { channels: vec![0, 1], start: 0, end: 120_000, width: 100 };
        let mut env1 = Envelope::Samples(Vec::new());
        sig1.view(&view, &mut env1).unwrap();
        drop(sig1);
        drop(chains1);

        // Session 2: reopen the same drv_id + same ops -> already complete immediately!
        let chains2 = ProcessedChains::new(&dir);
        let Made::Ready(sig2) = chains2.signal_for_drv(&drv_id, Some(&rec_path), raw.clone(), source.clone(), "main", &make_plan(2.0)).unwrap() else {
            panic!("no whitening");
        };
        assert!(sig2.frozen(), "reopened drv pyramid is immediately complete without recomputing");
        let mut env2 = Envelope::Samples(Vec::new());
        sig2.view(&view, &mut env2).unwrap();
        assert_eq!(env1, env2);

        // Edit ops on the same drv_id -> resets the SAME file in place (no second file created)
        let Made::Ready(sig3) = chains2.signal_for_drv(&drv_id, Some(&rec_path), raw.clone(), source.clone(), "main", &make_plan(5.0)).unwrap() else {
            panic!("no whitening");
        };
        assert!(!sig3.frozen(), "edited chain invalidated and reset the drv pyramid in place");
        let drv_files: Vec<_> = std::fs::read_dir(drv_file.parent().unwrap()).unwrap().filter_map(Result::ok).collect();
        assert_eq!(drv_files.len(), 1, "editing reuses the single drv file instead of accumulating files");

        // Cooperative cancellation: a pre-cancelled view returns immediately
        let cancelled = AtomicBool::new(true);
        let mut env_cancel = Envelope::Samples(Vec::new());
        sig3.view_cancellable(&view, &mut env_cancel, &cancelled).unwrap();

        drop(sig3);
        drop(sig2);
        drop(chains2);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
