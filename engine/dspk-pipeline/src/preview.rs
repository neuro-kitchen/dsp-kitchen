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
}

impl Preview {
    /// The preview of `stages` over `group` (channels of `source`).
    pub fn new(source: Arc<dyn RecordingSource>, group: Vec<usize>, stages: Vec<PipelineStage>) -> DspResult<Self> {
        let info = source.info();
        let pipeline = Pipeline::with_stages(stages);
        let halo = pipeline.settling(info.sample_rate_hz())?;
        let stored = info.format != SampleFormat::F64;
        let mut out_channels = group.clone();
        for stage in pipeline.stages() {
            if let PipelineStage::ReorderChannels { order } = stage {
                if order.len() != out_channels.len() || order.iter().any(|&i| i >= out_channels.len()) {
                    return Err(DspError::InvalidConfig(format!("a reorder of {} channels over {}", order.len(), out_channels.len())));
                }
                out_channels = order.iter().map(|&i| out_channels[i]).collect();
            }
        }
        Ok(Self { source, group, out_channels, pipeline, halo, stored, state: Mutex::new(State { workspace: None, block: None }) })
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
