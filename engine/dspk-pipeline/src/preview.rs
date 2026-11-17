//! Live preview of a chain of operations: the recording processed on the device window by
//! window, so an edit shows on what is on screen without writing anything.
//!
//! - The channels of `group` are processed together as one grouped array (references such as CMR
//!   span them).
//! - Each request reads the window plus the chain's settling halo on both sides, uploads the
//!   recording's **stored** values (int16 stays int16; scaled on the device), runs the chain and
//!   downloads the interior: equal to processing the whole recording. Float64 recordings are
//!   scaled on the host (wgpu has no f64), as in the pipeline executor.
//! - The processed block (with a margin of half a window on each side) is cached, so panning a
//!   little, hover readouts and redraws reuse it.
//! - Windows larger than [`PREVIEW_MAX_VALUES`] (channels × samples) are not meant to be
//!   processed: [`Preview::covers`] tells a viewer to draw the raw recording instead.
//!
//! This is the one place a viewer computes on the device while it draws. A viewer wraps it in
//! its own signal type (what to show for channels outside the group, how to fold samples into
//! columns); nothing here knows about views.

use std::ops::Range;
use std::sync::{Arc, Mutex};

use dspk_core::{ComputeTarget, DspError, DspResult, RecordingInfo, RecordingSource, SampleFormat};
use dspk_ops::core::buffer;
use dspk_ops::pipeline::PipelineWorkspace;
use dspk_ops::{Pipeline, PipelineStage};

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

/// The preview of a chain over a group of channels of a recording.
pub struct Preview {
    source: Arc<dyn RecordingSource>,
    /// Channels processed together, in the order they are read (indices of the source).
    group: Vec<usize>,
    /// The source channel each output row stands for: `group` through the chain's reorders.
    out_channels: Vec<usize>,
    pipeline: Pipeline,
    halo: (usize, usize),
    stored: bool,
    state: Mutex<State>,
    /// Held while a block is made for a reader that waits ([`Self::read`]).
    making: Mutex<()>,
}

impl Preview {
    /// The preview of `stages` over `group` (channels of `source`).
    pub fn new(source: Arc<dyn RecordingSource>, group: Vec<usize>, stages: Vec<PipelineStage>) -> DspResult<Self> {
        let info = source.info();
        let pipeline = Pipeline::with_stages(stages);
        let halo = pipeline.settling(info.sample_rate_hz())?;
        // Stored values go to the device as they are (int16 stays int16, scaled there) when the
        // source can give them: asked once, on one sample. Float64 (wgpu has none) and sources
        // without stored reads are scaled on the host and sent as float32
        let mut probe = vec![0u8; group.len().min(1) * info.format.bytes()];
        let stored = !matches!(info.format, SampleFormat::F16 | SampleFormat::F64) && (info.samples == 0 || source.read_stored(&group[..group.len().min(1)], 0..1, &mut probe).is_ok());
        let mut out_channels = group.clone();
        for stage in pipeline.stages() {
            if let PipelineStage::ReorderChannels { order } = stage {
                if order.len() != out_channels.len() || order.iter().any(|&i| i >= out_channels.len()) {
                    return Err(DspError::InvalidConfig(format!("a reorder of {} channels over {}", order.len(), out_channels.len())));
                }
                out_channels = order.iter().map(|&i| out_channels[i]).collect();
            }
        }
        Ok(Self { source, group, out_channels, pipeline, halo, stored, state: Mutex::new(State { workspace: None, block: None }), making: Mutex::new(()) })
    }

    /// The recording being previewed.
    pub fn info(&self) -> &RecordingInfo {
        self.source.info()
    }

    /// The source channel each output row stands for (the read channels through the reorders).
    pub fn out_channels(&self) -> &[usize] {
        &self.out_channels
    }

    /// Whether a window of `samples` samples fits a preview (else a viewer draws it raw).
    pub fn covers(&self, samples: u64) -> bool {
        self.block_values(samples) <= PREVIEW_MAX_VALUES
    }

    fn block_values(&self, samples: u64) -> usize {
        self.group.len() * (samples as usize + self.halo.0 + self.halo.1)
    }

    /// The block to compute for `range`: the samples kept (`want`, with half a window of margin
    /// each side when `slack` and it fits the budget, so small pans reuse it) and the samples
    /// read (`want` and the chain's settling halo).
    fn plan(&self, range: Range<u64>, slack: bool) -> (Range<u64>, Range<u64>) {
        let total = self.source.info().samples;
        let len = range.end - range.start;
        let margin = if slack && self.block_values(2 * len) <= PREVIEW_MAX_VALUES { len / 2 } else { 0 };
        let want = range.start.saturating_sub(margin)..(range.end + margin).min(total);
        let read = want.start.saturating_sub(self.halo.0 as u64)..(want.end + self.halo.1 as u64).min(total);
        (want, read)
    }

    /// The workspace kept from an earlier block when it holds `n` samples, else a new one.
    fn workspace(&self, state: &mut State, n: usize) -> DspResult<(PipelineWorkspace<f32>, usize)> {
        if let Some((ws, cap)) = state.workspace.take().filter(|(_, cap)| *cap >= n) {
            return Ok((ws, cap));
        }
        let client = ComputeTarget::from_env().and_then(ComputeTarget::client).map_err(|e| DspError::ComputeError(e.to_string()))?;
        let rate = self.source.info().sample_rate_hz();
        let mut ws = PipelineWorkspace::<f32>::new(client, self.pipeline.clone(), self.group.len(), n, rate)?;
        // Stored values are scaled on the device; values already scaled on the host pass through
        let info = self.source.info();
        let (gains, offsets): (Vec<f32>, Vec<f32>) =
            if self.stored { self.group.iter().map(|&c| (info.channels[c].gain, info.channels[c].offset)).unzip() } else { (vec![1.0; self.group.len()], vec![0.0; self.group.len()]) };
        ws.set_stored_scaling(&gains, &offsets);
        Ok((ws, n))
    }

    /// The interior `want` of `processed` (the rows over `read`), channel-major.
    fn keep(&self, want: Range<u64>, read: &Range<u64>, processed: &[f32]) -> Block {
        let n = (read.end - read.start) as usize;
        let (a, m) = ((want.start - read.start) as usize, (want.end - want.start) as usize);
        let mut data = Vec::with_capacity(self.group.len() * m);
        for r in 0..self.group.len() {
            data.extend_from_slice(&processed[r * n + a..r * n + a + m]);
        }
        Block { range: want, data }
    }

    /// The processed block holding `range` (made now when the cached one does not hold it, by
    /// [`Self::warm_async`], waited for here). `slack`: with half a window more on each side, so
    /// small pans reuse it (a view's reads); a branch reading this preview asks for exactly what
    /// it needs (its own block has the slack). In a browser nothing may wait for the device:
    /// there a block that was not made ready first is an error.
    fn with_block<R>(&self, range: Range<u64>, slack: bool, f: impl FnOnce(&Block) -> R) -> DspResult<R> {
        let holds = |b: &Block| b.range.start <= range.start && range.end <= b.range.end;
        // One block made at a time per preview (two threads asking for one window make it once)
        let _making = self.making.lock().unwrap();
        if let Some(b) = self.state.lock().unwrap().block.as_ref().filter(|b| holds(b)) {
            return Ok(f(b));
        }
        cubecl::future::reader::try_read_sync(self.warm_async(range.clone(), slack))
            .ok_or_else(|| DspError::ComputeError("the processed block was not made ready before it was read (a browser cannot wait for the device)".into()))??;
        let state = self.state.lock().unwrap();
        let b = state.block.as_ref().filter(|b| holds(b)).ok_or_else(|| DspError::ComputeError("the processed block was replaced while it was read".into()))?;
        Ok(f(b))
    }

    /// Makes the block holding `range` ready, reading the device's answer back without blocking
    /// (a browser hands it back only asynchronously). The one place a block is made: the desktop
    /// and a server wait for it ([`Self::read`]), a browser's engine awaits it before reading.
    pub async fn warm_async(&self, range: Range<u64>, slack: bool) -> DspResult<()> {
        let (want, read, mut ws, capacity) = {
            let mut state = self.state.lock().unwrap();
            if state.block.as_ref().is_some_and(|b| b.range.start <= range.start && range.end <= b.range.end) {
                return Ok(());
            }
            let (want, read) = self.plan(range, slack);
            let (ws, capacity) = self.workspace(&mut state, (read.end - read.start) as usize)?;
            (want, read, ws, capacity)
        };
        let (n, nch) = ((read.end - read.start) as usize, self.group.len());
        // Stored values as they are, else float32 scaled on the host (scaling 1, 0 on the device)
        let (bytes, format) = if self.stored {
            let format = self.source.info().format;
            let mut bytes = vec![0u8; nch * n * format.bytes()];
            self.source.read_stored(&self.group, read.clone(), &mut bytes).map(|_| (bytes, format))
        } else {
            let mut values = vec![0.0f32; nch * n];
            self.source.read(&self.group, read.clone(), &mut values).map(|_| (values.iter().flat_map(|v| v.to_le_bytes()).collect(), SampleFormat::F32))
        }?;
        let launched = {
            // Fixed launch choices: a preview never stops to benchmark kernels for a window size
            // it has not seen (zooming meets a new size at every step), and its numbers do not
            // depend on what a tuner picked
            let _pinned = dspk_core::compute::pin_tuned_choices();
            ws.process_owned_stored_chunk_in_vram(bytes, format, n).map_err(DspError::from)
        };
        let processed = match launched {
            Ok(handle) => buffer::download_prefix_async::<f32>(ws.client(), handle, nch * n).await.map_err(DspError::ComputeError),
            Err(e) => Err(e),
        };
        let mut state = self.state.lock().unwrap();
        state.workspace = Some((ws, capacity));
        state.block = Some(self.keep(want, &read, &processed?));
        Ok(())
    }

    /// The processed rows of `channels` over `samples`, channel-major into `out` (row `k` of
    /// `out` is `channels[k]`). Channels the chain does not give (outside the group) are left
    /// untouched in `out` and their positions `k` returned, for the caller to fill.
    ///
    /// `slack`: process half a window more on each side, so small pans reuse the block (a
    /// view's reads); a reader that asks for exactly what it needs passes `false`.
    pub fn read(&self, channels: &[usize], samples: Range<u64>, slack: bool, out: &mut [f32]) -> DspResult<Vec<usize>> {
        let n = (samples.end - samples.start) as usize;
        let rows: Vec<Option<usize>> = channels.iter().map(|c| self.out_channels.iter().position(|g| g == c)).collect();
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
        Ok(rows.iter().enumerate().filter(|(_, r)| r.is_none()).map(|(k, _)| k).collect())
    }
}

/// How a view's op chain is previewed, as the pipeline runs it.
pub struct ViewPlan {
    /// Source channels read, in order.
    pub group: Vec<usize>,
    /// Device steps and their keys (for sharing computed prefixes).
    pub steps: Vec<(String, PipelineStage)>,
    /// Source channels shown, in order.
    pub shown: Vec<usize>,
    /// Whitening steps (by index in `steps`, identities until fitted) and how to fit them.
    pub fits: Vec<(usize, crate::WhitenFit)>,
}

/// The plan of `ops` (a view's enabled ops, in order) over `group` (the channels the view
/// starts with): `group`, the channels read (through the remaps that open the chain: they fold
/// into the read and may drop); `steps`, the processing on the device with each step's key (a
/// remap between processing steps reorders there and keeps every channel); `shown`, the channels
/// it ends with (through the remaps after the last processing step: they may drop). Resampling
/// is not previewed. `base_dir`: where a remap's channel-map file is looked for.
pub fn view_plan(ops: &[&crate::Node], mut group: Vec<usize>, base_dir: &std::path::Path) -> Result<ViewPlan, String> {
    let is_remap = |n: &crate::Node| matches!(n, crate::Node::Remap { .. });
    let map_of = |n: &crate::Node| n.channel_map(Some(base_dir)).map_err(|e| e.to_string()).map(|m| m.expect("a remap has a map"));
    let lead = ops.iter().take_while(|n| is_remap(n)).count();
    for n in &ops[..lead] {
        let order = map_of(n)?.resolve(group.len()).map_err(|e| e.to_string())?;
        group = order.into_iter().map(|i| group[i]).collect();
    }
    let last_processing = ops.iter().rposition(|n| !is_remap(n));
    let mut steps = Vec::new();
    let mut fits = Vec::new();
    let mut current = group.clone();
    for (i, n) in ops.iter().enumerate().skip(lead) {
        let key = serde_json::to_string(n).unwrap_or_default();
        if is_remap(n) {
            let order = map_of(n)?.resolve(current.len()).map_err(|e| e.to_string())?;
            if last_processing.is_some_and(|last| i > last) {
                // After the last processing step: the channels shown (may drop)
                current = order.into_iter().map(|j| current[j]).collect();
                continue;
            }
            let mut seen = order.clone();
            seen.sort_unstable();
            if seen != (0..current.len()).collect::<Vec<_>>() {
                return Err(format!("a remap between processing steps keeps every channel ({}; it reorders them): drop channels in a remap first or last", current.len()));
            }
            current = order.iter().map(|&j| current[j]).collect();
            steps.push((key, PipelineStage::ReorderChannels { order }));
        } else if let Some(fit) = n.whiten_fit() {
            n.stage().map_err(|e| e.to_string())?;
            fits.push((steps.len(), fit));
            steps.push((key, PipelineStage::SpatialWhitening(crate::unfitted_whitening(current.len()))));
        } else if let Some(stage) = n.stage().map_err(|e| e.to_string())? {
            // A grid's operator is made for its electrodes
            if let PipelineStage::SurfaceLaplacian(lap) = &stage
                && lap.num_channels != current.len()
            {
                return Err(format!("the Laplacian's grid has {} electrodes; the view has {} channels", lap.num_channels, current.len()));
            }
            steps.push((key, stage));
        }
    }
    Ok(ViewPlan { group, steps, shown: current, fits })
}
