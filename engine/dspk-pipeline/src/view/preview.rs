//! Live preview of a chain of operations: the recording processed on the device window by
//! window, so an edit shows on what is on screen without writing anything.
//!
//! - The channels of `group` are processed together as one grouped array (references such as CMR
//!   span them).
//! - A block reads its window plus the chain's settling halo on both sides (rounded out to
//!   [`READ_GRID`]), uploads the recording's **stored** values (int16 stays int16; scaled on the
//!   device) and runs the chain: equal to processing the whole recording. Float64 recordings are
//!   scaled on the host (wgpu has no f64), as in the pipeline executor.
//! - **What stays on the device:** the processed block (with half a window of margin each side, so
//!   pans reuse it) and the uploaded input ([`InputCache`]: an edited chain over the same window does
//!   not read and upload it again). A branch reads its prefix's block there ([`Preview::after`]).
//! - **What comes back:** a zoomed-out view's `[min, max]` columns only (reduced on the device, the
//!   chain's trailing pointwise ops folded into the reduction), or the samples read only. A window
//!   wider than one block is processed in tiles, each launched before the last one's columns are
//!   read back.
//! - Where the time went: [`Preview::last_timings`].
//!
//! This is the one place a viewer computes on the device while it draws. A viewer wraps it in
//! its own signal type (what to show for channels outside the group); nothing here knows about
//! views.

use std::ops::Range;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex, OnceLock, Weak};

use cubecl::prelude::Client;
use cubecl::server::Handle;
use dspk_core::{ComputeTarget, DspError, DspResult, RecordingInfo, RecordingSource, SampleFormat};
use dspk_ops::core::buffer;
use dspk_ops::filter::{FilterBand, FilterDesign};
use dspk_ops::math::{pointwise_op, PointwiseRun};
use dspk_ops::resampler::{decimate_len, execute_resample_poly, firwin, FirWindow, ResampleFilter, RESAMPLE_POLY_DEFAULT_EDGE};
use dspk_ops::pipeline::PipelineWorkspace;
use dspk_ops::{Pipeline, PipelineStage};
use dspk_view::envelope::fold::{finish, merge, EMPTY};
use dspk_view::{envelope_launch_mapped, Columns, PointwiseMap};
use web_time::Instant;

use super::budget::{next_id, size_class, DeviceBudget, Holder};
use super::{PREVIEW_MAX_TILES, PREVIEW_MAX_VALUES};

/// Samples a block's read range is rounded out to (both ends): two chains over one window with
/// slightly different settling halos read the same samples, so the second reuses the first's
/// uploaded input ([`InputCache`]). Applied to every block, so a block is the same whether its input
/// came from the cache or was read again.
const READ_GRID: u64 = 512;

/// Uploaded inputs kept per recording (each one window of the group on the device).
const INPUTS_KEPT: usize = 2;

/// Samples per bucket of a frozen summary ([`Summary`]): views with at least this many samples per
/// column are drawn from it.
pub const SUMMARY_BUCKET: u64 = 256;

/// How long a chain is left unchanged (and viewed) before it is frozen: a parameter being dragged
/// makes a new chain per value, and those are not worth processing whole.
#[cfg_attr(target_family = "wasm", allow(dead_code))]
const FREEZE_AFTER_SECS: f64 = 1.5;

/// The preview a zoomed-out view asked for last: a chain is frozen when, after
/// [`FREEZE_AFTER_SECS`], no other chain was asked for since (a view left as it is counts).
static LAST_ASKED: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// When a view last asked for a chain (any chain): a freeze pass waits while views are drawn.
static VIEW_AT: Mutex<Option<Instant>> = Mutex::new(None);

/// One freeze pass at a time (each takes the device and the recording's reads for seconds).
#[cfg(not(target_family = "wasm"))]
static FREEZING: Mutex<()> = Mutex::new(());

/// A freeze pass yields this long after a view asked (views first), in seconds.
#[cfg(not(target_family = "wasm"))]
const YIELD_TO_VIEWS_SECS: f64 = 0.3;

/// A chain's output over the whole recording as `[min, max]` per bucket of [`SUMMARY_BUCKET`]
/// samples, row after row (NaN where a bucket holds no sample): a DAW's frozen track, as a peak
/// file. Built once in the background when the chain stops changing, kept in the RAM cache
/// ([`dspk_core::memory::host_cache`]); a zoomed-out view folds its buckets into columns.
pub struct Summary {
    pub rows: usize,
    pub buckets: usize,
    pub data: Vec<[f32; 2]>,
    /// Which buckets are built (all of them once frozen; a summary being built fills in from the
    /// part on screen outwards).
    pub built: Vec<bool>,
}

impl Summary {
    /// `[min, max]` of each of `width` even columns of `range`, every row: the buckets each column
    /// overlaps folded (a column's edges reach at most one bucket past them).
    pub fn columns(&self, range: &Range<u64>, width: usize) -> Vec<[f32; 2]> {
        self.columns_built(range, width).0
    }

    /// [`Self::columns`] from the buckets built so far (NaN where a column has none yet), and
    /// whether every bucket of `range` is built.
    pub fn columns_built(&self, range: &Range<u64>, width: usize) -> (Vec<[f32; 2]>, bool) {
        let len = range.end - range.start;
        let mut out = vec![[f32::NAN; 2]; self.rows * width];
        let mut complete = true;
        for x in 0..width {
            let c0 = range.start + x as u64 * len / width as u64;
            let c1 = (range.start + (x as u64 + 1) * len / width as u64).max(c0 + 1);
            let b0 = ((c0 / SUMMARY_BUCKET) as usize).min(self.buckets);
            let b1 = (c1.div_ceil(SUMMARY_BUCKET) as usize).min(self.buckets);
            complete &= self.built[b0..b1].iter().all(|&b| b);
            for r in 0..self.rows {
                let mut v = [f32::NAN; 2];
                for (b, built) in self.data[r * self.buckets + b0..r * self.buckets + b1].iter().zip(&self.built[b0..b1]) {
                    if *built {
                        v = [v[0].min(b[0]), v[1].max(b[1])];
                    }
                }
                out[r * width + x] = v;
            }
        }
        (out, complete)
    }
}

/// Largest rate reduction of a chain's tail ([`LowRate`]).
const MAX_RATE_FACTOR: usize = 256;

/// The new rate is at least this many times the low-pass's cutoff.
const RATE_PER_CUTOFF: f64 = 8.0;

/// Taps of the decimating filter per unit of the factor, and its Kaiser shape: about 100 dB of
/// stop-band from 0.7 × the new rate on, flat to 0.3 × it. What matters is what folds into the
/// low-pass's band (0..fc, fc ≤ new rate / 8): energy near multiples of the new rate, from
/// new rate − fc ≥ 0.875 × it, all of it in the stop-band. A Hamming window's 53 dB let a
/// Teager-Kaiser output's broadband power fold in at 7 % of an envelope's scale.
const DECIMATE_TAPS_PER_FACTOR: usize = 16;
const DECIMATE_KAISER_BETA: f64 = 10.0;

/// A chain whose last processing step is a low-pass at `fc` holds nothing above about `fc` from
/// there on, but would be computed and kept at the full rate (about 600× more than Nyquist needs
/// for a 20 Hz envelope at 24.4 kHz). Its low-pass runs at the rate divided by `factor` instead:
/// the steps before it at the full rate, then a polyphase FIR decimation (anti-aliasing low-pass at
/// the new Nyquist, every `factor`-th output only: ~20 multiply-adds per input sample), then the
/// low-pass at the low rate. Blocks, columns and frozen overviews are `factor` times smaller, and
/// the low-pass's cutoff ratio rises from 8e-4 to about 0.1.
///
/// Exactness: the result equals the full-rate chain within the decimating filter's pass-band
/// ripple and stop-band leakage (Kaiser `firwin`, β 10: below 1e-4 and about 100 dB), and zoomed-in samples
/// are the low-rate output linearly interpolated (a band-limited signal sampled at 8× its cutoff:
/// within about 2 % between samples). `DSPK_PREVIEW_FULL_RATE=1` turns it off.
struct LowRate {
    /// The full rate over the low one (a power of two, 4 to [`MAX_RATE_FACTOR`]: it divides
    /// [`READ_GRID`], so every block's decimated samples are the recording's samples k · factor).
    factor: usize,
    /// The low-pass, run at the low rate.
    pipeline: Pipeline,
    /// The decimating filter's taps (at the full rate, centred).
    taps: Vec<f64>,
}

impl LowRate {
    /// The lower-rate tail of `stages` (the chain's processing, its trailing pointwise ops left
    /// out) at `rate`: when its last step is a low-pass with room for a factor of at least 4.
    fn plan(stages: &[PipelineStage], rate: f64) -> Option<Self> {
        if std::env::var_os("DSPK_PREVIEW_FULL_RATE").is_some() {
            return None;
        }
        let PipelineStage::Filter(spec) = stages.last()? else { return None };
        let band = match &spec.design {
            FilterDesign::Butterworth { band, .. } | FilterDesign::Chebyshev1 { band, .. } | FilterDesign::Bessel { band, .. } => *band,
            _ => return None,
        };
        let FilterBand::Lowpass(fc) = band else { return None };
        let most = ((rate / (RATE_PER_CUTOFF * fc)).floor() as usize).min(MAX_RATE_FACTOR);
        if most < 4 {
            return None;
        }
        let factor = 1usize << (usize::BITS - 1 - most.leading_zeros());
        // Cut at the new Nyquist (firwin's cutoff is a fraction of the full rate's Nyquist)
        let taps = firwin(DECIMATE_TAPS_PER_FACTOR * factor + 1, 1.0 / factor as f64, FirWindow::Kaiser { beta: DECIMATE_KAISER_BETA });
        Some(Self { factor, pipeline: Pipeline::with_stages(vec![stages.last()?.clone()]), taps })
    }

    /// Samples of context the decimating filter reaches on each side (its half length).
    fn reach(&self) -> usize {
        self.taps.len().div_ceil(2) + self.factor
    }

    /// `columns` (in the recording's samples) in the low rate's samples.
    fn columns(&self, columns: Columns) -> Columns {
        let f = self.factor as u64;
        match columns {
            Columns::Even { start, len, width } => Columns::Even { start: start / f, len: (start + len).div_ceil(f) - start / f, width },
            Columns::Buckets { origin, size, count } => Columns::Buckets { origin: origin / f, size: size.div_ceil(f).max(1), count },
        }
    }
}

/// Where a preview's input comes from.
enum Input {
    /// The recording: read (its stored values, when the source gives them) and uploaded, or taken
    /// from the cache of uploaded windows.
    Recording { stored: bool, cache: Option<Arc<InputCache>> },
    /// Another preview's output, on the device: a branch, whose chain starts with that one's.
    Prefix(Arc<Preview>),
}

/// The uploaded (and scaled) input windows of a recording's groups, kept on the device: a chain
/// edited, or an op turned on or off, runs on the window already uploaded instead of reading and
/// uploading it again. Shared by every preview of a recording ([`super::ProcessedChains`]).
/// Charged to a [`DeviceBudget`] ([`Self::with_budget`]): released, it uploads again.
pub struct InputCache {
    kept: Mutex<Vec<((usize, Vec<usize>, Range<u64>), Handle, u64)>>,
    id: usize,
    budget: Option<Arc<DeviceBudget>>,
}

impl Default for InputCache {
    fn default() -> Self {
        Self { kept: Mutex::default(), id: next_id(), budget: None }
    }
}

impl InputCache {
    /// A cache whose inputs are charged to `budget` (and released when others need the room).
    pub fn with_budget(budget: Arc<DeviceBudget>) -> Arc<Self> {
        let cache = Arc::new(Self { kept: Mutex::default(), id: next_id(), budget: Some(budget.clone()) });
        let weak: Weak<dyn Holder> = Arc::downgrade(&(cache.clone() as Arc<dyn Holder>));
        budget.register(cache.id, weak);
        cache
    }

    fn get(&self, key: &(usize, Vec<usize>, Range<u64>)) -> Option<Handle> {
        let kept = self.kept.lock().unwrap();
        let found = kept.iter().find(|(k, _, _)| k == key).map(|(_, h, _)| h.clone());
        if found.is_some()
            && let Some(b) = &self.budget
        {
            b.touch(self.id);
        }
        found
    }

    /// Keeps `input` (`bytes` on the device) for `key`, making room in the budget first.
    fn put(&self, key: (usize, Vec<usize>, Range<u64>), input: Handle, bytes: u64, client: &Client) {
        let total = {
            let mut kept = self.kept.lock().unwrap();
            kept.retain(|(k, _, _)| *k != key);
            if kept.len() >= INPUTS_KEPT {
                kept.remove(0);
            }
            kept.iter().map(|(_, _, b)| b).sum::<u64>() + bytes
        };
        if let Some(b) = &self.budget {
            b.make_room(self.id, total, Some(client));
        }
        let mut kept = self.kept.lock().unwrap();
        kept.push((key, input, bytes));
        if let Some(b) = &self.budget {
            b.charge(self.id, kept.iter().map(|(_, _, b)| b).sum(), 0);
        }
    }

    /// Forgets the kept inputs (a batch job takes the device).
    pub fn clear(&self) {
        self.kept.lock().unwrap().clear();
        if let Some(b) = &self.budget {
            b.charge(self.id, 0, 0);
        }
    }
}

impl Holder for InputCache {
    fn release(&self, full: bool) -> Option<u64> {
        if !full {
            return Some(0);
        }
        let mut kept = self.kept.try_lock().ok()?;
        let bytes = kept.iter().map(|(_, _, b)| b).sum();
        kept.clear();
        Some(bytes)
    }
}

/// Where the last view's (or block's) time went, in ms: what a measuring script or the status
/// line reports. Host times always; `steps` (each launched step's device time) only in measuring
/// mode (`DSPK_PREVIEW_TIMING=1`, native), which waits for the device after each step.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PreviewTimings {
    /// Reading the recording on the host (0 when the input was reused or came from a prefix).
    pub read: f64,
    /// Uploading and scaling it (or cropping a prefix's rows).
    pub upload: f64,
    /// Launching the chain (and, in measuring mode, running it).
    pub process: f64,
    /// Each launched step's time, measuring mode only.
    pub steps: Vec<f64>,
    /// Reducing to columns and reading them back (zoomed out), or reading samples back.
    pub readback: f64,
    /// Tiles the window took (1: one block).
    pub tiles: usize,
    /// Whether the uploaded input was reused (an edit, a toggled op).
    pub input_reused: bool,
    /// Bytes every view keeps on the device after this one ([`DeviceBudget::charged`]), and the
    /// budget.
    pub kept_bytes: u64,
    pub budget_bytes: u64,
    /// Bytes the device's memory pools reserve (measuring mode, native only; 0 otherwise).
    pub device_reserved: u64,
}

fn measuring() -> bool {
    !cfg!(target_family = "wasm") && std::env::var_os("DSPK_PREVIEW_TIMING").is_some()
}

fn ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1e3
}

/// One processed block of the view's channels: its rows over `read` (the interior `range` and
/// the chain's settling halo), channel-major, before the chain's trailing pointwise ops
/// ([`Preview::trailing`]), kept on the device (zoomed-out views of it are reduced there, and pans
/// reuse it), and the interior on the host once samples are read (after the trailing ops).
struct Block {
    range: Range<u64>,
    read: Range<u64>,
    device: Handle,
    client: Client,
    /// Samples on the host: a region of the block (the samples read, with some slack for pans),
    /// channel-major.
    data: Option<(Range<u64>, Vec<f32>)>,
}

struct State {
    workspace: Option<(PipelineWorkspace<f32>, usize)>,
    block: Option<Block>,
    /// The last zoomed-out view's columns (its range and width, then the group's rows).
    columns: Option<(Range<u64>, usize, Vec<[f32; 2]>)>,
    /// The trailing pointwise ops on the device (uploaded once).
    trailing: Option<Arc<PointwiseRun>>,
    /// Whether the last columns answered were all known (false: a summary still being built).
    columns_complete: bool,
    /// The low-rate tail's workspace ([`LowRate`]) and its capacity.
    low_ws: Option<(PipelineWorkspace<f32>, usize)>,
    timings: PreviewTimings,
}

/// The preview of a chain over a group of channels of a recording.
pub struct Preview {
    source: Arc<dyn RecordingSource>,
    input: Input,
    /// Channels processed together, in the order they are read (indices of the source).
    group: Vec<usize>,
    /// The source channel each output row stands for: `group` through the chain's reorders.
    out_channels: Vec<usize>,
    /// The chain without its trailing pointwise ops (and without its low-pass when that runs at a
    /// lower rate: [`Self::low`]).
    pipeline: Pipeline,
    /// The chain's low-pass run at a lower rate, when it ends in one ([`LowRate`]): the block's
    /// rows are then at the low rate.
    low: Option<LowRate>,
    /// The chain's last pointwise ops (`(code, a, b)`): folded into the reduction of a zoomed-out
    /// view (their output is never written), one fused kernel on samples read back.
    trailing: Vec<(u32, f64, f64)>,
    halo: (usize, usize),
    /// Values one block (or tile) may hold: [`PREVIEW_MAX_VALUES`] unless set lower
    /// ([`Self::with_max_values`]).
    max_values: usize,
    state: Mutex<State>,
    /// Held while a block is made for a reader that waits ([`Self::read`]).
    making: Mutex<()>,
    /// This preview in the device budget (charged what it keeps; released when others need room).
    id: usize,
    budget: Arc<DeviceBudget>,
    /// How to make this chain from the recording alone (the source, the channels read, every stage
    /// from the recording on): what a frozen summary is built with.
    recipe: (Arc<dyn RecordingSource>, Vec<usize>, Vec<PipelineStage>),
    /// Where the frozen summary is kept in the RAM cache (set by the chains that keep this one;
    /// none: never frozen), whether one is being built, and when a view last asked.
    summary_key: OnceLock<String>,
    #[cfg_attr(target_family = "wasm", allow(dead_code))]
    freezing: AtomicBool,
    /// The summary while it is built (views of what is built are drawn from it), and the middle
    /// of the last window asked for (the build starts there).
    partial: Mutex<Option<Arc<Mutex<Summary>>>>,
    #[cfg_attr(target_family = "wasm", allow(dead_code))]
    last_center: std::sync::atomic::AtomicU64,
    /// Bumped by each zoomed-out request that is not answered from what is kept: a request made
    /// stale by a newer one stops at its next tile.
    generation: std::sync::atomic::AtomicU64,
    last_view: Mutex<Instant>,
    me: OnceLock<Weak<Preview>>,
}

impl Preview {
    /// The preview of `stages` over `group` (channels of `source`).
    pub fn new(source: Arc<dyn RecordingSource>, group: Vec<usize>, stages: Vec<PipelineStage>) -> DspResult<Self> {
        let info = source.info();
        // Stored values go to the device as they are (int16 stays int16, scaled there) when the
        // source can give them: asked once, on one sample. Float64 (wgpu has none) and sources
        // without stored reads are scaled on the host and sent as float32
        let mut probe = vec![0u8; group.len().min(1) * info.format.bytes()];
        let stored = !matches!(info.format, SampleFormat::F16 | SampleFormat::F64) && (info.samples == 0 || source.read_stored(&group[..group.len().min(1)], 0..1, &mut probe).is_ok());
        Self::with_input(source, Input::Recording { stored, cache: None }, group, stages, true)
    }

    /// [`Self::new`] with every step at the recording's rate (no [`LowRate`] tail): what the
    /// lower-rate tail is checked against.
    pub fn new_full_rate(source: Arc<dyn RecordingSource>, group: Vec<usize>, stages: Vec<PipelineStage>) -> DspResult<Self> {
        let info = source.info();
        let mut probe = vec![0u8; group.len().min(1) * info.format.bytes()];
        let stored = !matches!(info.format, SampleFormat::F16 | SampleFormat::F64) && (info.samples == 0 || source.read_stored(&group[..group.len().min(1)], 0..1, &mut probe).is_ok());
        Self::with_input(source, Input::Recording { stored, cache: None }, group, stages, false)
    }

    /// A branch: `stages` run on `prefix`'s output, on the device (no read back and upload in
    /// between). The branch's rows are the prefix's output rows, in its order.
    pub fn after(prefix: Arc<Preview>, stages: Vec<PipelineStage>) -> DspResult<Self> {
        if prefix.low.is_some() {
            // Its rows are at a lower rate: the branch runs from the recording instead
            return Err(DspError::InvalidConfig("the prefix runs its low-pass at a lower rate".into()));
        }
        let (source, group) = (prefix.source.clone(), prefix.out_channels.clone());
        Self::with_input(source, Input::Prefix(prefix), group, stages, false)
    }

    fn with_input(source: Arc<dyn RecordingSource>, input: Input, group: Vec<usize>, stages: Vec<PipelineStage>, lower_rate: bool) -> DspResult<Self> {
        let rate = source.info().sample_rate_hz();
        let full = Pipeline::with_stages(stages.clone());
        let halo = full.settling(rate)?;
        let mut out_channels = group.clone();
        for stage in full.stages() {
            if let PipelineStage::ReorderChannels { order } = stage {
                if order.len() != out_channels.len() || order.iter().any(|&i| i >= out_channels.len()) {
                    return Err(DspError::InvalidConfig(format!("a reorder of {} channels over {}", order.len(), out_channels.len())));
                }
                out_channels = order.iter().map(|&i| out_channels[i]).collect();
            }
        }
        // The chain's trailing pointwise ops run with the reduction or the read back
        let keep = stages.iter().rposition(|s| pointwise_op(s).is_none()).map_or(0, |i| i + 1);
        let trailing = stages[keep..].iter().filter_map(pointwise_op).collect();
        // A low-pass last runs at a lower rate (a branch reads its prefix's full-rate rows: not
        // there)
        let low = if lower_rate && matches!(input, Input::Recording { .. }) { LowRate::plan(&stages[..keep], rate) } else { None };
        let (pipeline, halo) = match &low {
            Some(low) => {
                let head = Pipeline::with_stages(stages[..keep - 1].to_vec());
                let (hl, hr) = head.settling(rate)?;
                let (tl, tr) = low.pipeline.settling(rate / low.factor as f64)?;
                let (reach, f) = (low.reach(), low.factor);
                (head, (hl + reach + tl * f, hr + reach + tr * f))
            }
            None => (Pipeline::with_stages(stages[..keep].to_vec()), halo),
        };
        let state = State { workspace: None, block: None, columns: None, trailing: None, columns_complete: true, low_ws: None, timings: PreviewTimings::default() };
        // A branch shares its prefix's budget; its recipe is the prefix's then its own stages
        let (budget, recipe) = match &input {
            Input::Prefix(p) => (p.budget.clone(), (p.recipe.0.clone(), p.recipe.1.clone(), p.recipe.2.iter().chain(&stages).cloned().collect())),
            Input::Recording { .. } => (DeviceBudget::global(), (source.clone(), group.clone(), stages.clone())),
        };
        let mut preview = Self {
            source,
            input,
            group,
            out_channels,
            pipeline,
            low,
            trailing,
            halo,
            max_values: PREVIEW_MAX_VALUES,
            state: Mutex::new(state),
            making: Mutex::new(()),
            id: next_id(),
            budget,
            recipe,
            summary_key: OnceLock::new(),
            freezing: AtomicBool::new(false),
            partial: Mutex::new(None),
            last_center: std::sync::atomic::AtomicU64::new(0),
            generation: std::sync::atomic::AtomicU64::new(0),
            last_view: Mutex::new(Instant::now()),
            me: OnceLock::new(),
        };
        preview.max_values = preview.max_values_for_budget(PREVIEW_MAX_VALUES);
        Ok(preview)
    }

    /// Full-size buffers one block needs at once (its input, the workspace's ping-pong, upload and
    /// unpack buffers, the block kept, the next tile in flight): a block's values are at most the
    /// budget over this many float32 buffers.
    const BUFFERS_PER_BLOCK: u64 = 8;

    /// `max_values`, lowered so one block fits the budget ([`Self::BUFFERS_PER_BLOCK`]).
    fn max_values_for_budget(&self, max_values: usize) -> usize {
        let fits = (self.budget.limit() / (4 * Self::BUFFERS_PER_BLOCK)) as usize;
        max_values.min(fits.max(1 << 20))
    }

    /// The same preview charged to `budget` instead of the process's ([`DeviceBudget::global`]).
    pub fn with_budget(mut self, budget: Arc<DeviceBudget>) -> Self {
        self.budget = budget;
        self.max_values = self.max_values_for_budget(self.max_values);
        self
    }

    /// Enters `preview` in its budget (where it is wrapped in an `Arc`): from then on other
    /// previews' room may be made by releasing its device memory.
    pub fn register(preview: &Arc<Preview>) {
        let weak: Weak<dyn Holder> = Arc::downgrade(&(preview.clone() as Arc<dyn Holder>));
        preview.budget.register(preview.id, weak);
        let _ = preview.me.set(Arc::downgrade(preview));
    }

    /// Freezes this chain under `key` in the RAM cache once it stops changing ([`Summary`]).
    pub fn set_summary_key(&self, key: String) {
        let _ = self.summary_key.set(key);
    }

    /// The frozen summary, when built and still kept.
    pub fn summary(&self) -> Option<Arc<Summary>> {
        dspk_core::memory::host_cache().get::<Summary>(self.summary_key.get()?)
    }

    /// A zoomed-out view was asked: after [`FREEZE_AFTER_SECS`] with views still asking for this
    /// chain, the whole recording is processed once in the background into its [`Summary`]
    /// (native; a browser keeps processing what it shows).
    fn consider_freezing(&self, soon: bool) {
        #[cfg(target_family = "wasm")]
        let _ = soon;
        if self.summary_key.get().is_none() {
            // Not a chain the views keep (a summary's own builder): nothing to freeze
            return;
        }
        *self.last_view.lock().unwrap() = Instant::now();
        *VIEW_AT.lock().unwrap() = Some(Instant::now());
        LAST_ASKED.store(self.id, std::sync::atomic::Ordering::SeqCst);
        #[cfg(not(target_family = "wasm"))]
        {
            let (Some(key), Some(me)) = (self.summary_key.get().cloned(), self.me.get().cloned()) else { return };
            if self.summary().is_some() || self.freezing.swap(true, std::sync::atomic::Ordering::SeqCst) {
                return;
            }
            std::thread::Builder::new()
                .name("dspk-freeze".into())
                .spawn(move || {
                    // A window too wide to process is waited on for less (nothing else draws it)
                    std::thread::sleep(std::time::Duration::from_secs_f64(if soon { YIELD_TO_VIEWS_SECS } else { FREEZE_AFTER_SECS }));
                    let Some(preview) = me.upgrade() else { return };
                    // Another chain was asked for since and this one not again: an edit replaced it
                    // (a parameter dragged); it may be asked again later
                    let replaced = LAST_ASKED.load(std::sync::atomic::Ordering::SeqCst) != preview.id;
                    if replaced && preview.last_view.lock().unwrap().elapsed().as_secs_f64() > FREEZE_AFTER_SECS.max(if soon { 0.0 } else { FREEZE_AFTER_SECS }) {
                        preview.freezing.store(false, std::sync::atomic::Ordering::SeqCst);
                        return;
                    }
                    drop(preview);
                    let built = Self::freeze(&me);
                    if let Some(preview) = me.upgrade() {
                        match built {
                            Some(summary) => {
                                let bytes = (summary.data.len() * 8 + summary.built.len()) as u64;
                                dspk_core::memory::host_cache().put(key, Arc::new(summary), bytes);
                            }
                            None => preview.freezing.store(false, std::sync::atomic::Ordering::SeqCst),
                        }
                        *preview.partial.lock().unwrap() = None;
                    }
                })
                .ok();
        }
    }

    /// The chain over the whole recording as a [`Summary`]: made from the recording by a preview of
    /// its own (the views' blocks stay), segment by segment, each reduced to its buckets on the
    /// device. `None` when the chain went (dropped) or a segment failed.
    #[cfg(not(target_family = "wasm"))]
    fn freeze(me: &Weak<Preview>) -> Option<Summary> {
        let (source, group, stages, budget) = {
            let p = me.upgrade()?;
            (p.recipe.0.clone(), p.recipe.1.clone(), p.recipe.2.clone(), p.budget.clone())
        };
        // One pass at a time
        let _one = FREEZING.lock().unwrap_or_else(|e| e.into_inner());
        // A quarter of a view's block: the pass leaves the device to what is on screen
        let quarter = (budget.limit() / (4 * Self::BUFFERS_PER_BLOCK) / 4) as usize;
        let builder = Arc::new(Preview::new(source, group, stages).ok()?.with_budget(budget.clone()).with_max_values(quarter.max(1 << 20)));
        Preview::register(&builder);
        let client = Self::client().ok()?;
        let total = builder.info().samples;
        let rows = builder.out_channels.len();
        // Segments of whole buckets that fit one block
        let fit = (builder.max_values / builder.group.len().max(1)).saturating_sub(builder.halo.0 + builder.halo.1 + 2 * READ_GRID as usize) as u64;
        let segment = (fit / SUMMARY_BUCKET * SUMMARY_BUCKET).max(SUMMARY_BUCKET);
        let need = (Self::BUFFERS_PER_BLOCK as usize * builder.group.len() * size_class(segment as usize + 2 * READ_GRID as usize) * 4) as u64;
        let buckets = total.div_ceil(SUMMARY_BUCKET) as usize;
        let partial = Arc::new(Mutex::new(Summary { rows, buckets, data: vec![[f32::NAN; 2]; rows * buckets], built: vec![false; buckets] }));
        let center = {
            let p = me.upgrade()?;
            *p.partial.lock().unwrap() = Some(partial.clone());
            p.last_center.load(std::sync::atomic::Ordering::Relaxed)
        };
        // Segments from the one on screen outwards (as map tiles fill in from the view)
        let count = total.div_ceil(segment);
        let mid = (center / segment).min(count.saturating_sub(1)) as i64;
        let mut order: Vec<u64> = (0..count).collect();
        order.sort_by_key(|&k| ((k as i64 - mid).abs(), k));
        for k in order {
            let start = k * segment;
            let p = me.upgrade()?;
            drop(p);
            // Views first: wait while one is being drawn, and while the device has no room for a
            // segment (given up after a minute of waiting: frozen later, when asked again)
            let waited = Instant::now();
            loop {
                let viewing = VIEW_AT.lock().unwrap().is_some_and(|t| t.elapsed().as_secs_f64() < YIELD_TO_VIEWS_SECS);
                if !viewing && budget.fits(need, Some(&client)) {
                    break;
                }
                if waited.elapsed().as_secs() > 60 {
                    return None;
                }
                std::thread::sleep(std::time::Duration::from_millis(50));
            }
            let end = (start + segment).min(total);
            let width = (end - start).div_ceil(SUMMARY_BUCKET) as usize;
            let part = cubecl::future::block_on(builder.columns_async(start..end, width)).ok()?;
            let b0 = (start / SUMMARY_BUCKET) as usize;
            let mut sum = partial.lock().unwrap();
            for r in 0..rows {
                sum.data[r * buckets + b0..r * buckets + b0 + width].copy_from_slice(&part[r * width..(r + 1) * width]);
            }
            sum.built[b0..b0 + width].iter_mut().for_each(|b| *b = true);
        }
        let done = std::mem::replace(&mut *partial.lock().unwrap(), Summary { rows: 0, buckets: 0, data: Vec::new(), built: Vec::new() });
        Some(done)
    }

    /// A window `range` is asked for that is too wide to process now: its summary is built (from
    /// there outwards), and views of it are drawn from what is built meanwhile.
    pub fn want_overview(&self, range: &Range<u64>) {
        self.last_center.store((range.start + range.end) / 2, std::sync::atomic::Ordering::Relaxed);
        self.consider_freezing(true);
    }

    /// Whether a window of `samples` is processed exactly now: in one block or in a few tiles.
    fn exact_now(&self, samples: u64) -> bool {
        self.fits_one_block(samples) || self.tile_samples().is_some_and(|t| samples.div_ceil(t as u64) as usize <= PREVIEW_MAX_TILES)
    }

    /// The budget this preview is charged to.
    pub fn budget(&self) -> &Arc<DeviceBudget> {
        &self.budget
    }

    /// The device bytes `state` keeps, and how many of them are the workspace's.
    fn kept(&self, state: &State) -> (u64, u64) {
        let ws = state.workspace.as_ref().map_or(0, |(ws, _)| ws.device_bytes());
        let block = state.block.as_ref().map_or(0, |b| self.block_bytes(&b.read));
        (ws + block, ws)
    }

    /// The bytes of a processed block over `read` (allocated in a size class).
    fn block_bytes(&self, read: &Range<u64>) -> u64 {
        (self.group.len() * size_class((read.end - read.start) as usize) * 4) as u64
    }

    /// Charges what this preview keeps now, and records it in the timings.
    fn charge(&self, state: &mut State) {
        let (bytes, ws) = self.kept(state);
        self.budget.charge(self.id, bytes, ws);
        state.timings.kept_bytes = self.budget.charged();
        state.timings.budget_bytes = self.budget.limit();
    }

    /// Makes room in the budget for a block over `read` (with `tiles` blocks in flight) before it
    /// is made: the block, its input and a workspace for it, what this preview keeps now included.
    fn make_room(&self, read: &Range<u64>, tiles: u64, client: Option<&Client>) {
        // Every buffer a block holds while it is made (input, upload and unpack, the workspace's
        // ping-pong and scratch, its own copy, the reduction), per tile in flight
        let rows = self.block_bytes(read);
        self.budget.make_room(self.id, (1 + tiles) * Self::BUFFERS_PER_BLOCK * rows / 2, client);
    }

    /// The same preview reading its recording's input through `cache` (shared by a recording's
    /// previews): a chain edited over a window already uploaded does not read it again.
    pub fn with_input_cache(mut self, cache: Arc<InputCache>) -> Self {
        if let Input::Recording { cache: c, .. } = &mut self.input {
            *c = Some(cache);
        }
        self
    }

    /// The same preview with blocks and tiles of at most `max_values` values (a smaller device;
    /// tests make tiles of a small recording with it).
    pub fn with_max_values(mut self, max_values: usize) -> Self {
        self.max_values = self.max_values_for_budget(max_values);
        self
    }

    /// The source previewed.
    pub fn source(&self) -> Arc<dyn RecordingSource> {
        self.source.clone()
    }

    /// The recording being previewed.
    pub fn info(&self) -> &RecordingInfo {
        self.source.info()
    }

    /// The source channel each output row stands for (the read channels through the reorders).
    pub fn out_channels(&self) -> &[usize] {
        &self.out_channels
    }

    /// The factor the chain's low-pass runs at below the recording's rate (1: every step at the
    /// recording's rate; [`LowRate`]).
    pub fn rate_factor(&self) -> usize {
        self.low.as_ref().map_or(1, |l| l.factor)
    }

    /// Whether the last zoomed-out columns were all known (false while a summary is built: ask
    /// again for more).
    pub fn columns_complete(&self) -> bool {
        self.state.lock().unwrap().columns_complete
    }

    /// Where the last view's time went ([`PreviewTimings`]).
    pub fn last_timings(&self) -> PreviewTimings {
        self.state.lock().unwrap().timings.clone()
    }

    /// Whether a window of `samples` samples is previewed processed: in one block, or zoomed out
    /// in up to [`PREVIEW_MAX_TILES`] tiles (else a viewer draws it raw).
    pub fn covers(&self, samples: u64) -> bool {
        self.summary().is_some() || self.partial.lock().unwrap().is_some() || self.exact_now(samples)
    }

    /// Whether a window of `samples` samples is processed in one block.
    pub fn fits_one_block(&self, samples: u64) -> bool {
        self.block_values(samples) <= self.max_values
    }

    /// Interior samples of one tile (a tile holds them, the chain's halo and the read grid's
    /// rounding within [`PREVIEW_MAX_VALUES`]), or `None` when those alone fill a tile.
    fn tile_samples(&self) -> Option<usize> {
        (self.max_values / self.group.len().max(1)).checked_sub(self.halo.0 + self.halo.1 + 2 * READ_GRID as usize).filter(|&t| t > 0)
    }

    fn block_values(&self, samples: u64) -> usize {
        self.group.len() * (samples as usize + self.halo.0 + self.halo.1 + 2 * READ_GRID as usize)
    }

    /// The samples read for `want`: `want` and the chain's settling halo, rounded out to
    /// [`READ_GRID`] (clipped to the recording).
    fn read_range(&self, want: &Range<u64>) -> Range<u64> {
        let total = self.source.info().samples;
        // A branch reads its prefix's rows on the device: exactly what it needs (the grid is for
        // sharing reads of the recording)
        if matches!(self.input, Input::Prefix(_)) {
            return want.start.saturating_sub(self.halo.0 as u64)..(want.end + self.halo.1 as u64).min(total);
        }
        let start = want.start.saturating_sub(self.halo.0 as u64) / READ_GRID * READ_GRID;
        let end = (want.end + self.halo.1 as u64).div_ceil(READ_GRID).saturating_mul(READ_GRID).min(total);
        start..end
    }

    /// The block to compute for `range`: the samples kept (`want`, with half a window of margin
    /// each side when `slack` and it fits the budget, so small pans reuse it) and the samples
    /// read ([`Self::read_range`] of `want`).
    fn plan(&self, range: Range<u64>, slack: bool) -> (Range<u64>, Range<u64>) {
        let total = self.source.info().samples;
        let len = range.end - range.start;
        let margin = if slack && self.block_values(2 * len) <= self.max_values { len / 2 } else { 0 };
        let mut want = range.start.saturating_sub(margin)..(range.end + margin).min(total);
        // A branch keeps within its prefix's block when that holds what it needs for `range`: its
        // input is then read there, not computed again over a slightly wider window
        if let Input::Prefix(p) = &self.input
            && let Some(kept) = p.state.lock().unwrap().block.as_ref().map(|b| b.range.clone())
        {
            let lo = if kept.start == 0 { 0 } else { kept.start + self.halo.0 as u64 };
            let hi = if kept.end >= total { total } else { kept.end.saturating_sub(self.halo.1 as u64) };
            if lo <= range.start && range.end <= hi {
                want = want.start.max(lo)..want.end.min(hi);
            }
        }
        let read = self.read_range(&want);
        (want, read)
    }

    /// The workspace kept from an earlier block when it holds `n` samples, else a new one.
    fn workspace(&self, state: &mut State, n: usize) -> DspResult<(PipelineWorkspace<f32>, usize)> {
        if let Some((ws, cap)) = state.workspace.take().filter(|(_, cap)| *cap >= n) {
            return Ok((ws, cap));
        }
        // Sized in a class: the next window of a nearby size reuses it (and its buffers' pages)
        let n = size_class(n);
        let client = Self::client()?;
        let rate = self.source.info().sample_rate_hz();
        let mut ws = PipelineWorkspace::<f32>::new(client, self.pipeline.clone(), self.group.len(), n, rate)?;
        // Stored values are scaled on the device; values already scaled on the host pass through
        let info = self.source.info();
        let stored = matches!(self.input, Input::Recording { stored: true, .. });
        let (gains, offsets): (Vec<f32>, Vec<f32>) =
            if stored { self.group.iter().map(|&c| (info.channels[c].gain, info.channels[c].offset)).unzip() } else { (vec![1.0; self.group.len()], vec![0.0; self.group.len()]) };
        ws.set_stored_scaling(&gains, &offsets);
        Ok((ws, n))
    }

    /// The default compute target's client (where previews run).
    fn client() -> DspResult<Client> {
        ComputeTarget::from_env().and_then(ComputeTarget::client).map_err(|e| DspError::ComputeError(e.to_string()))
    }

    /// The trailing pointwise ops on `client` (uploaded on first use), if the chain has any.
    fn trailing_run(&self, client: &Client) -> Option<Arc<PointwiseRun>> {
        if self.trailing.is_empty() {
            return None;
        }
        let mut state = self.state.lock().unwrap();
        Some(state.trailing.get_or_insert_with(|| Arc::new(PointwiseRun::upload::<f32>(client, &self.trailing))).clone())
    }

    /// The group's samples over `read` as the device takes them: stored values as they are, else
    /// float32 scaled on the host (scaling 1, 0 on the device).
    fn host_input(&self, read: &Range<u64>) -> DspResult<(Vec<u8>, SampleFormat)> {
        let (n, nch) = ((read.end - read.start) as usize, self.group.len());
        if matches!(self.input, Input::Recording { stored: true, .. }) {
            let format = self.source.info().format;
            let mut bytes = vec![0u8; nch * n * format.bytes()];
            self.source.read_stored(&self.group, read.clone(), &mut bytes).map(|_| (bytes, format))
        } else {
            let mut values = vec![0.0f32; nch * n];
            self.source.read(&self.group, read.clone(), &mut values).map(|_| (values.iter().flat_map(|v| v.to_le_bytes()).collect(), SampleFormat::F32))
        }
    }

    /// The chain's input rows over `read` on the device (`[group, read]`): a prefix's output
    /// cropped there, the cached upload of this window, or the recording read and uploaded now
    /// (kept in the cache when the preview has one, `keep`).
    async fn device_input(&self, ws: &mut PipelineWorkspace<f32>, read: &Range<u64>, keep: bool, timings: &mut PreviewTimings) -> DspResult<Handle> {
        let n = (read.end - read.start) as usize;
        match &self.input {
            Input::Prefix(prefix) => {
                let t = Instant::now();
                let rows = prefix.rows_on_device(read.clone()).await?;
                timings.upload += ms(t);
                Ok(rows)
            }
            Input::Recording { cache, .. } => {
                let key = (Arc::as_ptr(&self.source) as *const () as usize, self.group.clone(), read.clone());
                if let Some(input) = cache.as_ref().and_then(|c| c.get(&key)) {
                    timings.input_reused = true;
                    return Ok(input);
                }
                let t = Instant::now();
                let (bytes, format) = self.host_input(read)?;
                timings.read += ms(t);
                let t = Instant::now();
                let input = ws.upload_stored_owned(bytes, format, n).map_err(DspError::from)?;
                timings.upload += ms(t);
                if keep && let Some(cache) = cache {
                    // The input is sized to the workspace's capacity (a size class)
                    let held = (self.group.len() * size_class(n) * 4) as u64;
                    cache.put(key, input.clone(), held, ws.client());
                }
                Ok(input)
            }
        }
    }

    /// Runs the chain (without its trailing pointwise ops) over `read`: the processed rows, in a
    /// buffer of their own (the workspace's are reused by the next block).
    async fn process(&self, ws: &mut PipelineWorkspace<f32>, read: &Range<u64>, keep: bool, timings: &mut PreviewTimings) -> DspResult<Handle> {
        let (n, nch) = ((read.end - read.start) as usize, self.group.len());
        let input = self.device_input(ws, read, keep, timings).await?;
        // Fixed launch choices: a preview never stops to benchmark kernels for a window size it
        // has not seen (zooming meets a new size at every step), and its numbers do not depend on
        // what a tuner picked
        let _pinned = dspk_core::compute::pin_tuned_choices();
        let t = Instant::now();
        let out = if measuring() {
            #[cfg(not(target_family = "wasm"))]
            let client = ws.client().clone();
            let mut last = Instant::now();
            let mut steps = Vec::new();
            let out = ws.process_handle_with(&input, n, |_| {
                #[cfg(not(target_family = "wasm"))]
                cubecl::future::block_on(client.sync()).ok();
                steps.push(ms(last));
                last = Instant::now();
            });
            timings.steps = steps;
            out
        } else {
            ws.process_handle(&input, n)
        };
        let own = buffer::empty::<f32>(ws.client(), nch * size_class(n));
        buffer::copy::<f32>(ws.client(), &out, &own, nch * n);
        // The low-pass at the lower rate: the block's rows are the low rate's
        let own = match &self.low {
            Some(low) => self.lower_rate(ws.client(), &own, n, low)?,
            None => own,
        };
        timings.process += ms(t);
        Ok(own)
    }

    /// `rows` (`[group, n]` at the full rate) decimated by `low.factor` and low-passed at the low
    /// rate, into a buffer of their own (`[group, decimate_len(n, factor)]`).
    fn lower_rate(&self, client: &Client, rows: &Handle, n: usize, low: &LowRate) -> DspResult<Handle> {
        let nch = self.group.len();
        let m = decimate_len(n, low.factor);
        let decimated = buffer::empty::<f32>(client, (nch * m).max(1));
        // Polyphase: only every factor-th output is computed
        execute_resample_poly::<f32>(client, rows, &decimated, nch, n, 1, low.factor, &ResampleFilter::Taps(low.taps.clone()), RESAMPLE_POLY_DEFAULT_EDGE);
        let kept = self.state.lock().unwrap().low_ws.take().filter(|(_, cap)| *cap >= m);
        let (mut ws, cap) = match kept {
            Some(k) => k,
            None => {
                let cap = size_class(m.max(1));
                (PipelineWorkspace::<f32>::new(client.clone(), low.pipeline.clone(), nch, cap, self.source.info().sample_rate_hz() / low.factor as f64)?, cap)
            }
        };
        let out = ws.process_handle(&decimated, m);
        let own = buffer::empty::<f32>(client, (nch * size_class(m.max(1))).max(1));
        buffer::copy::<f32>(client, &out, &own, nch * m);
        self.state.lock().unwrap().low_ws = Some((ws, cap));
        Ok(own)
    }

    /// This preview's output rows (after its trailing ops) over `range` exactly, on the device
    /// (`[out rows, range]`): what a branch reads as its input. Boxed: a prefix may itself be a
    /// branch of another prefix (the async functions call each other through it).
    fn rows_on_device(&self, range: Range<u64>) -> std::pin::Pin<Box<dyn std::future::Future<Output = DspResult<Handle>> + Send + '_>> {
        Box::pin(self.rows_on_device_inner(range))
    }

    async fn rows_on_device_inner(&self, range: Range<u64>) -> DspResult<Handle> {
        self.block_async(range.clone(), false).await?;
        let (client, rows, read) = {
            let state = self.state.lock().unwrap();
            let b = state.block.as_ref().filter(|b| b.range.start <= range.start && range.end <= b.range.end).ok_or_else(|| DspError::ComputeError("the prefix's block was replaced while it was read".into()))?;
            (b.client.clone(), b.device.clone(), b.read.clone())
        };
        let (nch, len) = (self.group.len(), (range.end - range.start) as usize);
        let cropped = buffer::empty::<f32>(&client, nch * len);
        dspk_ops::core::layout::crop_rows::<f32>(&client, &rows, &cropped, nch, (read.end - read.start) as usize, (range.start - read.start) as usize, len);
        Ok(match self.trailing_run(&client) {
            Some(run) => {
                let out = buffer::empty::<f32>(&client, nch * len);
                run.apply::<f32>(&client, &cropped, &out, nch * len);
                out
            }
            None => cropped,
        })
    }

    /// The processed block holding `range` (made now when the cached one does not hold it, by
    /// [`Self::warm_async`], waited for here). `slack`: with half a window more on each side, so
    /// small pans reuse it (a view's reads); a reader that asks for exactly what it needs passes
    /// `false`. In a browser nothing may wait for the device: there a block that was not made
    /// ready first is an error.
    fn with_block<R>(&self, range: Range<u64>, slack: bool, f: impl FnOnce(&Block) -> R) -> DspResult<R> {
        let holds = |b: &Block| b.data.as_ref().is_some_and(|(region, _)| region.start <= range.start && range.end <= region.end);
        // One block made at a time per preview (two threads asking for one window make it once)
        let _making = self.making.lock().unwrap();
        if let Some(b) = self.state.lock().unwrap().block.as_ref().filter(|b| holds(b)) {
            self.budget.touch(self.id);
            return Ok(f(b));
        }
        cubecl::future::reader::try_read_sync(self.warm_async(range.clone(), slack))
            .ok_or_else(|| DspError::ComputeError("the processed block was not made ready before it was read (a browser cannot wait for the device)".into()))??;
        let state = self.state.lock().unwrap();
        let b = state.block.as_ref().filter(|b| holds(b)).ok_or_else(|| DspError::ComputeError("the processed block was replaced while it was read".into()))?;
        Ok(f(b))
    }

    /// Makes the block holding `range` ready on the device (processing it when the kept one does
    /// not hold it). Nothing is read back.
    async fn block_async(&self, range: Range<u64>, slack: bool) -> DspResult<()> {
        let (want, read, client) = {
            let mut state = self.state.lock().unwrap();
            if state.block.as_ref().is_some_and(|b| b.range.start <= range.start && range.end <= b.range.end) {
                drop(state);
                self.budget.touch(self.id);
                return Ok(());
            }
            let (want, read) = self.plan(range, slack);
            // The block this replaces goes first (its memory is free for the new one)
            let client = state.block.take().map(|b| b.client);
            self.charge(&mut state);
            (want, read, client)
        };
        // Room in the budget before anything is allocated: other views release what they keep
        let client = match client {
            Some(c) => c,
            None => Self::client()?,
        };
        self.make_room(&read, 0, Some(&client));
        let (mut ws, capacity) = {
            let mut state = self.state.lock().unwrap();
            self.workspace(&mut state, (read.end - read.start) as usize)?
        };
        let mut timings = PreviewTimings { tiles: 1, ..Default::default() };
        let processed = self.process(&mut ws, &read, true, &mut timings).await;
        let client = ws.client().clone();
        if measuring() {
            timings.device_reserved = client.memory_report(cubecl::MemoryScope::Device).usage().bytes_reserved;
        }
        let mut state = self.state.lock().unwrap();
        state.workspace = Some((ws, capacity));
        state.block = Some(Block { range: want, read, device: processed?, client, data: None });
        state.timings = timings;
        self.charge(&mut state);
        Ok(())
    }

    /// Makes the block holding `range` ready with its samples on the host, reading the device's
    /// answer back without blocking (a browser hands it back only asynchronously). The desktop
    /// and a server wait for it ([`Self::read`]), a browser's engine awaits it before reading.
    pub async fn warm_async(&self, range: Range<u64>, slack: bool) -> DspResult<()> {
        self.block_async(range.clone(), slack).await?;
        let (handle, client, read, want) = {
            let state = self.state.lock().unwrap();
            let Some(b) = state.block.as_ref().filter(|b| b.range.start <= range.start && range.end <= b.range.end) else {
                return Err(DspError::ComputeError("the processed block was replaced while it was made".into()));
            };
            if b.data.as_ref().is_some_and(|(region, _)| region.start <= range.start && range.end <= region.end) {
                return Ok(());
            }
            (b.device.clone(), b.client.clone(), b.read.clone(), b.range.clone())
        };
        // Only what is read comes back (and half of it more on each side for pans), not the block
        let len = range.end - range.start;
        let region = if slack { range.start.saturating_sub(len / 2).max(want.start)..(range.end + len / 2).min(want.end) } else { range.clone() };
        let t = Instant::now();
        let data = self.download_region(&client, &handle, &read, &region).await?;
        let mut state = self.state.lock().unwrap();
        state.timings.readback = ms(t);
        // Still the block read back (not replaced meanwhile)
        if let Some(b) = state.block.as_mut().filter(|b| b.range == want && b.read == read) {
            b.data = Some((region, data));
        }
        Ok(())
    }

    /// Samples `region` of every row of a block's device buffer (rows over `read`), after the
    /// chain's trailing ops, channel-major: cropped on the device and read back in one transfer
    /// of `rows × region` values.
    async fn download_region(&self, client: &Client, rows: &Handle, read: &Range<u64>, region: &Range<u64>) -> DspResult<Vec<f32>> {
        let (nch, n) = (self.group.len(), (read.end - read.start) as usize);
        let m = (region.end - region.start) as usize;
        if nch == 0 || m == 0 {
            return Ok(Vec::new());
        }
        if let Some(low) = &self.low {
            return self.download_low(client, rows, read, region, low).await;
        }
        let cropped = buffer::empty::<f32>(client, nch * m);
        dspk_ops::core::layout::crop_rows::<f32>(client, rows, &cropped, nch, n, (region.start - read.start) as usize, m);
        let out = match self.trailing_run(client) {
            Some(run) => {
                let out = buffer::empty::<f32>(client, nch * m);
                run.apply::<f32>(client, &cropped, &out, nch * m);
                out
            }
            None => cropped,
        };
        buffer::download_prefix_async::<f32>(client, out, nch * m).await.map_err(DspError::ComputeError)
    }

    /// [`Self::download_region`] of low-rate rows: the low-rate samples around `region` read back,
    /// interpolated linearly to the recording's samples (the signal is band-limited well below the
    /// low rate's Nyquist).
    async fn download_low(&self, client: &Client, rows: &Handle, read: &Range<u64>, region: &Range<u64>, low: &LowRate) -> DspResult<Vec<f32>> {
        let (nch, f) = (self.group.len(), low.factor as u64);
        let n_low = decimate_len((read.end - read.start) as usize, low.factor) as u64;
        let base = read.start / f;
        let lo = (region.start / f).max(base);
        let hi = (region.end.div_ceil(f) + 1).min(base + n_low).max(lo + 1);
        let k = (hi - lo) as usize;
        let cropped = buffer::empty::<f32>(client, nch * k);
        dspk_ops::core::layout::crop_rows::<f32>(client, rows, &cropped, nch, n_low as usize, (lo - base) as usize, k);
        let out = match self.trailing_run(client) {
            Some(run) => {
                let out = buffer::empty::<f32>(client, nch * k);
                run.apply::<f32>(client, &cropped, &out, nch * k);
                out
            }
            None => cropped,
        };
        let values = buffer::download_prefix_async::<f32>(client, out, nch * k).await.map_err(DspError::ComputeError)?;
        let m = (region.end - region.start) as usize;
        let mut samples = Vec::with_capacity(nch * m);
        for row in values.chunks_exact(k) {
            for t in region.clone() {
                let pos = (t as f64 / f as f64 - lo as f64).max(0.0);
                let i = (pos.floor() as usize).min(k - 1);
                let j = (i + 1).min(k - 1);
                let w = (pos - i as f64).clamp(0.0, 1.0) as f32;
                samples.push(row[i] * (1.0 - w) + row[j] * w);
            }
        }
        Ok(samples)
    }

    /// Reduces rows over `read` (`rows`, on `client`) to `columns` over the interior `valid`, the
    /// chain's trailing ops applied on the way, and reads the columns back (before `finish`).
    async fn reduce(&self, client: Client, rows: Handle, read: Range<u64>, valid: Range<u64>, columns: Columns, width: usize) -> DspResult<Vec<[f32; 2]>> {
        let nch = self.group.len();
        let n = (read.end - read.start) as usize;
        // Low-rate rows: everything in the low rate's samples
        let (n, start, valid, columns) = match &self.low {
            Some(low) => {
                let f = low.factor as u64;
                (decimate_len(n, low.factor), read.start / f, valid.start / f..valid.end.div_ceil(f), low.columns(columns))
            }
            None => (n, read.start, valid, columns),
        };
        let run = self.trailing_run(&client);
        let map = run.as_ref().map(|r| PointwiseMap { codes: r.codes(), params: r.params(), count: r.count() });
        let Some(out) = envelope_launch_mapped::<f32>(&client, &rows, nch, n, start, valid, columns, map)? else { return Ok(vec![EMPTY; nch * width]) };
        let pairs = buffer::download_prefix_async::<f32>(&client, out, nch * width * 2).await.map_err(DspError::ComputeError)?;
        Ok(pairs.as_chunks::<2>().0.to_vec())
    }

    /// `[min, max]` of each of `width` even columns of samples `range`, for every row of the group
    /// (row after row, NaN where a column holds no sample): reduced on the device, so only the
    /// columns are read back. A window that fits one block is processed as a block (kept: pans
    /// and finer views reuse it); a wider one in tiles of [`PREVIEW_MAX_VALUES`], each with the
    /// chain's halo on both sides and reduced over its own interior only, the tiles' partial
    /// columns merged. Tiles overlap: the next tile is read and launched before the last one's
    /// columns are awaited. The last answer is kept for [`Self::columns`].
    pub async fn columns_async(&self, range: Range<u64>, width: usize) -> DspResult<Vec<[f32; 2]>> {
        let kept = self.state.lock().unwrap().columns.as_ref().filter(|(r, w, _)| *r == range && *w == width).map(|(_, _, v)| v.clone());
        let exact = self.exact_now(range.end - range.start);
        self.last_center.store((range.start + range.end) / 2, std::sync::atomic::Ordering::Relaxed);
        self.consider_freezing(!exact);
        if let Some(v) = kept {
            self.budget.touch(self.id);
            self.state.lock().unwrap().columns_complete = true;
            return Ok(v);
        }
        // Frozen: folded from the summary's buckets (nothing processed)
        if (range.end - range.start) >= SUMMARY_BUCKET * width as u64
            && let Some(summary) = self.summary()
        {
            let v = summary.columns(&range, width);
            let mut state = self.state.lock().unwrap();
            state.columns = Some((range, width, v.clone()));
            state.columns_complete = true;
            return Ok(v);
        }
        // Too wide to process now: what the summary being built holds (it fills in from here)
        if !exact {
            let built = self.partial.lock().unwrap().clone();
            let (v, complete) = match built {
                Some(p) => p.lock().unwrap().columns_built(&range, width),
                None => (vec![[f32::NAN; 2]; self.group.len() * width], false),
            };
            self.state.lock().unwrap().columns_complete = complete;
            return Ok(v);
        }
        self.state.lock().unwrap().columns_complete = true;
        let generation = self.generation.fetch_add(1, std::sync::atomic::Ordering::SeqCst) + 1;
        let nch = self.group.len();
        let columns = Columns::Even { start: range.start, len: range.end - range.start, width };
        let mut acc = if self.fits_one_block(range.end - range.start) {
            self.block_async(range.clone(), true).await?;
            let (client, rows, read) = {
                let state = self.state.lock().unwrap();
                let b = state.block.as_ref().filter(|b| b.range.start <= range.start && range.end <= b.range.end).ok_or_else(|| DspError::ComputeError("the processed block was replaced while it was reduced".into()))?;
                (b.client.clone(), b.device.clone(), b.read.clone())
            };
            let t = Instant::now();
            let acc = self.reduce(client, rows, read, range.clone(), columns, width).await?;
            self.state.lock().unwrap().timings.readback = ms(t);
            acc
        } else {
            let tile = self.tile_samples().ok_or_else(|| DspError::InvalidConfig("the chain's settling halo alone fills a preview tile".into()))? as u64;
            // Room for two tiles in flight (one reduced while the next runs) and their workspace
            let first = self.read_range(&(range.start..(range.start + tile).min(range.end)));
            let client = Self::client()?;
            self.make_room(&first, 1, Some(&client));
            let mut timings = PreviewTimings::default();
            let mut acc = vec![EMPTY; nch * width];
            // The tile launched last, whose columns are read back after the next one is launched
            let mut pending: Option<(Client, Handle, Range<u64>, Range<u64>)> = None;
            let mut start = range.start;
            loop {
                let next = (start < range.end).then(|| start..(start + tile).min(range.end));
                // A newer view asked: this one is not drawn, stop here
                if self.generation.load(std::sync::atomic::Ordering::SeqCst) != generation {
                    return Err(DspError::ComputeError("a newer view replaced this one".into()));
                }
                if let Some(want) = &next {
                    let read = self.read_range(want);
                    // Room for this tile before anything of it is allocated; with no room for two
                    // in flight, the last one is read back first
                    self.make_room(&read, 1, Some(&client));
                    let two = 2 * Self::BUFFERS_PER_BLOCK * self.block_bytes(&read);
                    if !self.budget.fits(two, Some(&client))
                        && let Some((c, rows, read, valid)) = pending.take()
                    {
                        let t = Instant::now();
                        let part = self.reduce(c, rows, read, valid, columns, width).await?;
                        timings.readback += ms(t);
                        for (a, p) in acc.iter_mut().zip(part) {
                            *a = merge(*a, p);
                        }
                    }
                    let (mut ws, capacity) = {
                        let mut state = self.state.lock().unwrap();
                        self.workspace(&mut state, (read.end - read.start) as usize)?
                    };
                    let processed = self.process(&mut ws, &read, false, &mut timings).await;
                    let client = ws.client().clone();
                    self.state.lock().unwrap().workspace = Some((ws, capacity));
                    let launched = (client, processed?, read, want.clone());
                    // The previous tile's columns, while this one runs on the device
                    if let Some((client, rows, read, valid)) = pending.replace(launched) {
                        let t = Instant::now();
                        let part = self.reduce(client, rows, read, valid, columns, width).await?;
                        timings.readback += ms(t);
                        for (a, p) in acc.iter_mut().zip(part) {
                            *a = merge(*a, p);
                        }
                    }
                    timings.tiles += 1;
                    start = want.end;
                } else {
                    if let Some((client, rows, read, valid)) = pending.take() {
                        let t = Instant::now();
                        let part = self.reduce(client, rows, read, valid, columns, width).await?;
                        timings.readback += ms(t);
                        for (a, p) in acc.iter_mut().zip(part) {
                            *a = merge(*a, p);
                        }
                    }
                    break;
                }
            }
            if measuring() {
                timings.device_reserved = client.memory_report(cubecl::MemoryScope::Device).usage().bytes_reserved;
            }
            let mut state = self.state.lock().unwrap();
            state.timings = timings;
            self.charge(&mut state);
            drop(state);
            acc
        };
        finish(&mut acc);
        self.state.lock().unwrap().columns = Some((range, width, acc.clone()));
        Ok(acc)
    }

    /// [`Self::columns_async`], answered from the last one when it was for this range and width,
    /// else made now, waited for (the desktop, a server; in a browser it is made first).
    pub fn columns(&self, range: Range<u64>, width: usize) -> DspResult<Vec<[f32; 2]>> {
        let _making = self.making.lock().unwrap();
        cubecl::future::reader::try_read_sync(self.columns_async(range, width))
            .ok_or_else(|| DspError::ComputeError("the processed columns were not made ready before they were read (a browser cannot wait for the device)".into()))?
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
                let Some((region, data)) = b.data.as_ref() else { return };
                let (m, a) = ((region.end - region.start) as usize, (samples.start - region.start) as usize);
                for (k, row) in rows.iter().enumerate() {
                    if let Some(r) = row {
                        out[k * n..(k + 1) * n].copy_from_slice(&data[r * m + a..r * m + a + n]);
                    }
                }
            })?;
        }
        Ok(rows.iter().enumerate().filter(|(_, r)| r.is_none()).map(|(k, _)| k).collect())
    }
}

impl Holder for Preview {
    /// Frees the workspace (`full == false`: a chain off screen keeps only its result) or the block
    /// too (the chain is made again when a view asks for it). Busy while a block is made or read.
    fn release(&self, full: bool) -> Option<u64> {
        let _making = self.making.try_lock().ok()?;
        let mut state = self.state.try_lock().ok()?;
        let ws = state.workspace.take().map_or(0, |(ws, _)| ws.device_bytes());
        let block = if full { state.block.take().map_or(0, |b| self.block_bytes(&b.read)) } else { 0 };
        Some(ws + block)
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
/// starts with), planned as every chain is ([`crate::chain`], for a view): `group`, the channels
/// read (through the remaps that open the chain: they fold into the read and may drop); `steps`,
/// the processing on the device with each step's key (a remap between processing steps reorders
/// there and keeps every channel); `shown`, the channels it ends with (through the remaps after
/// the last processing step: they may drop). Resampling and crossings are not previewed (the view
/// shows the signal before them). `base_dir`: where a remap's channel-map file is read.
pub fn view_plan(ops: &[&crate::Node], group: Vec<usize>, base_dir: &std::path::Path) -> Result<ViewPlan, String> {
    use crate::chain::{plan_chain, refusals_text, Purpose, Step};
    let nodes: Vec<(String, crate::Node)> = ops.iter().enumerate().map(|(i, n)| (format!("op {}", i + 1), (*n).clone())).collect();
    let plan = plan_chain(&nodes, &group, Purpose::View, Some(base_dir)).map_err(|r| refusals_text(&r))?;
    let fits = plan.steps.iter().enumerate().filter_map(|(i, s)| if let Step::Whiten(fit) = &s.step { Some((i, *fit)) } else { None }).collect();
    let steps = plan.stages(|_, channels| PipelineStage::SpatialWhitening(crate::unfitted_whitening(channels)));
    Ok(ViewPlan { group: plan.read.clone(), steps, shown: plan.shown(), fits })
}
