//! The chains made for a recording's views, kept so views share what they have in common.
//!
//! A chain's **key** is its source and channels, then each step's key (the node as JSON). A view
//! whose chain starts with a kept chain's whole key reads that chain's output on the device and
//! runs only its own steps ([`ProcessedSignal::after`]): two views of one recording, one
//! band-passed and one band-passed then rectified, compute the band-pass once.
//!
//! **Editing a step** ([`ProcessedChains::signal_editing`]): the chain up to the step being edited
//! is kept as a chain of its own, so each new value of the step reruns only from there; and every
//! chain of a recording reads its windows through one [`InputCache`], so a new chain over the window
//! already on screen does not read and upload it again.
//!
//! A **whitening** is fitted on the signal of the steps before it, which reads the recording
//! (seconds). Fits are kept by the key of the chain up to them; a chain that needs one not fitted
//! yet is answered with a [`FitJob`] for the caller to run where it may wait (the app in the
//! background, a server at once; a browser cannot fit yet), then [`ProcessedChains::put_fit`].

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use dspk_core::RecordingSource;
use dspk_ops::spatial::SpatialWhitening;
use dspk_ops::PipelineStage;
#[cfg(not(target_family = "wasm"))]
use dspk_view::PyramidIdentity;
use dspk_view::{Pyramid, SignalBackend, View};
use sha2::{Digest, Sha256};

use super::budget::{next_id, DeviceBudget};
use super::cached::CachedSource;
#[cfg(not(target_family = "wasm"))]
use super::preview::SUMMARY_BUCKET;
use super::preview::{view_plan, InputCache, ViewPlan};
use super::processed::ProcessedSignal;
use crate::WhitenFit;

/// Chains kept per recording for anonymous (non-`drv_id`) callers (`dspk serve`, browser engine).
/// Named derived chains (`signal_for_drv`) hold strictly one slot per `drv_id`.
const CHAINS_KEPT: usize = 2;

/// What [`ProcessedChains::signal`] gives.
pub enum Made {
    /// The chain's processed signal.
    Ready(Arc<ProcessedSignal>),
    /// A whitening of the chain is not fitted yet: run the job, give its result to
    /// [`ProcessedChains::put_fit`], and ask again.
    NeedsFit(FitJob),
}

/// A whitening to fit: on `source`'s `group` through the steps `before` it.
pub struct FitJob {
    /// The key of the chain up to the whitening (where the fit is kept).
    pub key: String,
    pub source: Arc<dyn RecordingSource>,
    pub group: Vec<usize>,
    pub before: Vec<PipelineStage>,
    pub fit: WhitenFit,
}

impl FitJob {
    /// Fits the whitening on the default compute target (reads the recording: seconds).
    pub fn run(&self) -> Result<SpatialWhitening, String> {
        let target = dspk_core::ComputeTarget::from_env().map_err(|e| e.to_string())?;
        crate::fit_whitening(target, self.source.as_ref(), &self.group, &self.before, &self.fit).map_err(|e| e.to_string())
    }
}

/// The chains made for one recording, by key, most recent last; and the whitenings fitted.
/// Files the ops name (a remap's channel map) are read from `base_dir`.
pub struct ProcessedChains {
    base_dir: PathBuf,
    /// Recording path on disk when this chain table serves a specific recording (`dspk serve`):
    /// enables disk-backed mmapped pyramids under `<rec>.cache/drv/` instead of in-memory RAM.
    recording_path: Option<PathBuf>,
    /// Named derived chains: strictly one flat `ProcessedSignal` slot per `drv_id`.
    by_drv: Mutex<HashMap<String, (Vec<String>, Arc<ProcessedSignal>)>>,
    made: Mutex<Vec<(Vec<String>, Arc<ProcessedSignal>)>>,
    /// Chains asked by their ops as JSON (a page's requests carry them), by (channels, ops).
    asked: Mutex<Vec<((Vec<usize>, Vec<String>), Arc<ProcessedSignal>)>>,
    fitted: Mutex<HashMap<String, SpatialWhitening>>,
    /// The recording's uploaded windows, shared by its chains.
    inputs: Arc<InputCache>,
    /// What every chain keeps on the device is charged to.
    budget: Arc<DeviceBudget>,
    /// These chains' entries in the RAM cache (decoded recording blocks) are under `chains:{id}:`;
    /// dropped with them.
    id: usize,
    /// Each source read through the RAM cache, made once (by its key and the source itself): every
    /// chain of a source reads the same one, so the uploaded windows ([`InputCache`], keyed by it)
    /// serve them all.
    cached: Mutex<Vec<(String, usize, Arc<dyn RecordingSource>)>>,
}

impl Drop for ProcessedChains {
    fn drop(&mut self) {
        dspk_core::memory::host_cache().forget_prefix(&format!("chains:{}:", self.id));
    }
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

impl ProcessedChains {
    pub fn new(base_dir: impl Into<PathBuf>) -> Self {
        Self::with_budget(base_dir, DeviceBudget::global())
    }

    /// A chain table bound to a recording file on disk (`dspk serve`): derived chains persist
    /// their mmapped pyramids under `<rec>.cache/drv/` instead of holding them in RAM.
    pub fn for_recording(recording: &Path) -> Self {
        let base_dir = recording.parent().map(Path::to_path_buf).unwrap_or_default();
        let mut chains = Self::with_budget(base_dir, DeviceBudget::global());
        chains.recording_path = Some(recording.to_path_buf());
        chains
    }

    /// [`Self::new`] charging the chains' device memory to `budget` (tests; the process's budget is
    /// sized from the device).
    pub fn with_budget(base_dir: impl Into<PathBuf>, budget: Arc<DeviceBudget>) -> Self {
        let inputs = InputCache::with_budget(budget.clone());
        Self {
            base_dir: base_dir.into(),
            recording_path: None,
            by_drv: Mutex::default(),
            made: Default::default(),
            asked: Default::default(),
            fitted: Default::default(),
            inputs,
            budget,
            id: next_id(),
            cached: Mutex::default(),
        }
    }

    /// The device budget the chains are charged to.
    pub fn budget(&self) -> &Arc<DeviceBudget> {
        &self.budget
    }

    /// Where files the ops name are read from.
    pub fn base_dir(&self) -> &Path {
        &self.base_dir
    }

    /// The processed signal of `plan` (a view's chain, [`view_plan`]) over `source` (`raw`
    /// answers what the chain does not process), `source_key` naming the source among a
    /// recording's. Always executes as one flat [`PipelineWorkspace`] (constant GPU memory
    /// regardless of op count); a whitening not fitted yet is a [`Made::NeedsFit`].
    pub fn signal(&self, raw: Arc<dyn SignalBackend>, source: Arc<dyn RecordingSource>, source_key: &str, plan: &ViewPlan) -> Result<Made, String> {
        self.signal_editing(raw, source, source_key, plan, None)
    }

    /// The flat processed signal for a named derived chain (`drv_id`), backed by its resumable
    /// mmapped pyramid at `<dir>/cache/<rec>.cache/drv/<drv_id>_drv_<source_key>.minmax` when
    /// `recording_path` is provided (else in memory).
    ///
    /// Strictly one slot is kept per `drv_id`: changing the chain replaces the slot in place
    /// (immediately freeing the old GPU buffers) and resets the mmapped `.minmax` file in place.
    pub fn signal_for_drv(
        &self,
        drv_id: &str,
        recording_path: Option<&Path>,
        raw: Arc<dyn SignalBackend>,
        source: Arc<dyn RecordingSource>,
        source_key: &str,
        plan: &ViewPlan,
    ) -> Result<Made, String> {
        let (key, stages) = match self.resolve_stages(&source, source_key, plan) {
            Ok(v) => v,
            Err(job) => return Ok(Made::NeedsFit(job)),
        };
        if let Some((existing_key, signal)) = lock(&self.by_drv).get(drv_id).cloned()
            && existing_key == key
        {
            return Ok(Made::Ready(signal));
        }
        let built = self.build_flat(raw, source.clone(), source_key, plan.group.clone(), stages, &key, recording_path.map(|p| (p, drv_id)))?;
        lock(&self.by_drv).insert(drv_id.to_string(), (key, built.clone()));
        Ok(Made::Ready(built))
    }

    /// Removes a derived chain's runtime slot (its mmapped `drv/` file on disk is left intact).
    pub fn remove_drv(&self, drv_id: &str) {
        lock(&self.by_drv).remove(drv_id);
    }

    /// [`Self::signal`] (flat single-workspace execution; `editing` is accepted for API compatibility
    /// — the shared [`InputCache`] reuses the uploaded window without recursive prefix chains).
    pub fn signal_editing(&self, raw: Arc<dyn SignalBackend>, source: Arc<dyn RecordingSource>, source_key: &str, plan: &ViewPlan, _editing: Option<usize>) -> Result<Made, String> {
        let (key, stages) = match self.resolve_stages(&source, source_key, plan) {
            Ok(v) => v,
            Err(job) => return Ok(Made::NeedsFit(job)),
        };
        if let Some((_, p)) = lock(&self.made).iter().find(|(c, _)| *c == key) {
            return Ok(Made::Ready(p.clone()));
        }
        let built = self.build_flat(raw, source, source_key, plan.group.clone(), stages, &key, None)?;
        let mut made = lock(&self.made);
        made.retain(|(c, _)| *c != key);
        if made.len() >= CHAINS_KEPT {
            made.remove(0);
        }
        made.push((key, built.clone()));
        Ok(Made::Ready(built))
    }

    fn resolve_stages(&self, source: &Arc<dyn RecordingSource>, source_key: &str, plan: &ViewPlan) -> Result<(Vec<String>, Vec<PipelineStage>), FitJob> {
        let group = plan.group.clone();
        let mut stages: Vec<PipelineStage> = plan.steps.iter().map(|(_, st)| st.clone()).collect();
        let mut key = vec![format!("{source_key}|{group:?}")];
        key.extend(plan.steps.iter().map(|(k, _)| k.clone()));
        for (at, fit) in &plan.fits {
            let up_to = key[..at + 2].join("\n");
            match lock(&self.fitted).get(&up_to) {
                Some(w) => stages[*at] = PipelineStage::SpatialWhitening(w.clone()),
                None => return Err(FitJob { key: up_to, source: source.clone(), group, before: stages[..*at].to_vec(), fit: *fit }),
            }
        }
        Ok((key, stages))
    }

    fn build_flat(
        &self,
        raw: Arc<dyn SignalBackend>,
        source: Arc<dyn RecordingSource>,
        source_key: &str,
        group: Vec<usize>,
        stages: Vec<PipelineStage>,
        key: &[String],
        drv_file: Option<(&Path, &str)>,
    ) -> Result<Arc<ProcessedSignal>, String> {
        let cached = self.cached_source(source.clone(), source_key);
        let built = Arc::new(
            ProcessedSignal::with_budget(raw, cached, group, stages, self.inputs.clone(), self.budget.clone())
                .map_err(|e| e.to_string())?,
        );
        // Attach a resumable mmapped pyramid (for named derived chains on disk) or an in-memory
        // pyramid (for anonymous / wasm / synthetic chains)
        let info = source.info();
        let rows = built.out_channels().len();
        let sig = chain_signature(key);
        #[cfg(not(target_family = "wasm"))]
        let pyramid = match drv_file {
            Some((rec_path, drv_id)) => {
                let id = PyramidIdentity::of(rec_path, source_key, source.as_ref())
                    .map(|mut id| {
                        id.channels = rows;
                        id.with_signature(sig)
                    });
                match id.and_then(|id| Pyramid::open_or_create_drv(rec_path, source_key, drv_id, &id, SUMMARY_BUCKET)) {
                    Ok(p) => Some(Arc::new(p)),
                    Err(e) => {
                        tracing::warn!("derived pyramid for {drv_id}: {e}; using in-memory pyramid");
                        Pyramid::in_memory_shape(rows, info.samples, info.sample_rate_hz(), SUMMARY_BUCKET).ok().map(Arc::new)
                    }
                }
            }
            None => Pyramid::in_memory_shape(rows, info.samples, info.sample_rate_hz(), SUMMARY_BUCKET).ok().map(Arc::new),
        };
        #[cfg(target_family = "wasm")]
        let pyramid: Option<Arc<Pyramid>> = {
            let _ = (drv_file, sig, rows, info);
            None
        };
        if let Some(p) = pyramid {
            built.attach_pyramid(p);
        }
        Ok(built)
    }

    /// `source` read through the RAM cache: one wrapper per source, kept (the same object for every
    /// chain, so their uploaded windows are shared).
    fn cached_source(&self, source: Arc<dyn RecordingSource>, source_key: &str) -> Arc<dyn RecordingSource> {
        let ptr = Arc::as_ptr(&source) as *const () as usize;
        let mut cached = lock(&self.cached);
        if let Some((_, _, c)) = cached.iter().find(|(k, p, _)| k == source_key && *p == ptr) {
            return c.clone();
        }
        let c: Arc<dyn RecordingSource> = Arc::new(CachedSource::new(source, format!("chains:{}:raw:{source_key}:{ptr}", self.id)));
        cached.retain(|(k, _, _)| k != source_key);
        cached.push((source_key.to_string(), ptr, c.clone()));
        c
    }

    /// Keeps a whitening fitted for the chain up to it ([`FitJob::key`]).
    pub fn put_fit(&self, key: String, whitening: SpatialWhitening) {
        lock(&self.fitted).insert(key, whitening);
    }

    /// The signal of `ops` (pipeline nodes as JSON, in order) over `channels` (none = all) of
    /// `source`: what `dspk serve` and a browser's engine make for a view a page asks for. A
    /// whitening is fitted now (seconds), except in a browser, which cannot wait for its GPU.
    pub fn make(&self, raw: Arc<dyn SignalBackend>, source: Arc<dyn RecordingSource>, channels: &[usize], ops: &[String]) -> Result<Arc<ProcessedSignal>, String> {
        let nodes: Vec<crate::Node> = ops.iter().map(|op| serde_json::from_str(op).map_err(|e| format!("an op that is not a pipeline node: {e}"))).collect::<Result<_, _>>()?;
        let group: Vec<usize> = if channels.is_empty() { (0..source.info().channel_count()).collect() } else { channels.to_vec() };
        let plan = view_plan(&nodes.iter().collect::<Vec<_>>(), group, &self.base_dir)?;
        let source_key = {
            let n = source.info().name.as_str();
            if n.is_empty() { "main" } else { n }
        };
        let drv_id = self.recording_path.as_ref().map(|_| {
            let mut key = vec![format!("{source_key}|{:?}", plan.group)];
            key.extend(plan.steps.iter().map(|(k, _)| k.clone()));
            let sig = chain_signature(&key);
            format!("{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}", sig[0], sig[1], sig[2], sig[3], sig[4], sig[5], sig[6], sig[7])
        });
        let signal = loop {
            let made = match (&self.recording_path, &drv_id) {
                (Some(rec), Some(id)) => self.signal_for_drv(id, Some(rec.as_path()), raw.clone(), source.clone(), source_key, &plan)?,
                _ => self.signal(raw.clone(), source.clone(), source_key, &plan)?,
            };
            match made {
                Made::Ready(signal) => break signal,
                Made::NeedsFit(job) => {
                    if cfg!(target_family = "wasm") {
                        return Err("whitening is fitted on a server or in the desktop app, not in a browser yet".into());
                    }
                    let fitted = job.run()?;
                    self.put_fit(job.key, fitted);
                }
            }
        };
        // Keep at most `CHAINS_KEPT` slots in `by_drv` for anonymous server requests too
        if let Some(id) = &drv_id {
            let mut by_drv = lock(&self.by_drv);
            if by_drv.len() > CHAINS_KEPT {
                let drop_key = by_drv.keys().find(|k| *k != id).cloned();
                if let Some(k) = drop_key {
                    by_drv.remove(&k);
                }
            }
        }
        let mut asked = lock(&self.asked);
        asked.retain(|(k, _)| k.0 != channels || k.1 != ops);
        if asked.len() >= CHAINS_KEPT {
            asked.remove(0);
        }
        asked.push(((channels.to_vec(), ops.to_vec()), signal.clone()));
        Ok(signal)
    }

    /// Forgets the chains kept (a batch job takes the GPU: their device memory goes with them).
    /// Fitted whitenings are kept (they are small and slow to fit again).
    pub fn clear(&self) {
        lock(&self.by_drv).clear();
        lock(&self.made).clear();
        lock(&self.asked).clear();
        self.inputs.clear();
    }

    /// Makes `view` of the chain of `ops` over `channels` ([`Self::make`]) ready without
    /// blocking on the device; nothing to do for a chain not made here.
    pub async fn warm(&self, channels: &[usize], ops: &[String], view: &View) -> dspk_core::DspResult<()> {
        let signal = lock(&self.asked).iter().find(|(k, _)| k.0 == channels && k.1 == ops).map(|(_, s)| s.clone());
        match signal {
            Some(signal) => signal.warm_view_async(view).await,
            None => Ok(()),
        }
    }
}

/// Canonical 32-byte SHA-256 signature of a chain key (source, channels, and ordered op specs).
fn chain_signature(key: &[String]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    for part in key {
        hasher.update((part.len() as u64).to_le_bytes());
        hasher.update(part.as_bytes());
    }
    hasher.finalize().into()
}
