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
use dspk_view::{SignalBackend, View};

use super::preview::{view_plan, ViewPlan};
use super::budget::{next_id, DeviceBudget};
use super::cached::CachedSource;
use super::preview::InputCache;
use super::processed::ProcessedSignal;
use crate::WhitenFit;

/// Chains kept per recording. What they keep on the device is the device budget's to bound
/// ([`DeviceBudget`]): a chain released for room keeps its plan and fit, and is made again when a
/// view asks for it.
const CHAINS_KEPT: usize = 8;

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
    made: Mutex<Vec<(Vec<String>, Arc<ProcessedSignal>)>>,
    /// Chains asked by their ops as JSON (a page's requests carry them), by (channels, ops).
    asked: Mutex<Vec<((Vec<usize>, Vec<String>), Arc<ProcessedSignal>)>>,
    fitted: Mutex<HashMap<String, SpatialWhitening>>,
    /// The recording's uploaded windows, shared by its chains.
    inputs: Arc<InputCache>,
    /// What every chain keeps on the device is charged to.
    budget: Arc<DeviceBudget>,
    /// These chains' entries in the RAM cache (decoded recording blocks, frozen summaries) are
    /// under `chains:{id}:`; dropped with them.
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

    /// [`Self::new`] charging the chains' device memory to `budget` (tests; the process's budget is
    /// sized from the device).
    pub fn with_budget(base_dir: impl Into<PathBuf>, budget: Arc<DeviceBudget>) -> Self {
        let inputs = InputCache::with_budget(budget.clone());
        Self { base_dir: base_dir.into(), made: Default::default(), asked: Default::default(), fitted: Default::default(), inputs, budget, id: next_id(), cached: Mutex::default() }
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
    /// recording's. Kept, sharing the longest kept prefix; a whitening not fitted yet is a
    /// [`Made::NeedsFit`].
    pub fn signal(&self, raw: Arc<dyn SignalBackend>, source: Arc<dyn RecordingSource>, source_key: &str, plan: &ViewPlan) -> Result<Made, String> {
        self.signal_editing(raw, source, source_key, plan, None)
    }

    /// [`Self::signal`] while step `editing` of the plan is being edited (a parameter dragged):
    /// the chain up to it is kept on its own first, so the next values of the step rerun only the
    /// steps from it on.
    pub fn signal_editing(&self, raw: Arc<dyn SignalBackend>, source: Arc<dyn RecordingSource>, source_key: &str, plan: &ViewPlan, editing: Option<usize>) -> Result<Made, String> {
        if let Some(k) = editing.filter(|&k| k > 0 && k < plan.steps.len()) {
            let prefix = ViewPlan {
                group: plan.group.clone(),
                steps: plan.steps[..k].to_vec(),
                shown: plan.group.clone(),
                fits: plan.fits.iter().filter(|(at, _)| *at < k).cloned().collect(),
            };
            if let Made::NeedsFit(job) = self.signal_editing(raw.clone(), source.clone(), source_key, &prefix, None)? {
                return Ok(Made::NeedsFit(job));
            }
        }
        let group = plan.group.clone();
        let mut stages: Vec<PipelineStage> = plan.steps.iter().map(|(_, st)| st.clone()).collect();
        // The chain's key, step by step: [source and channels, step 1, step 2, …]
        let mut key = vec![format!("{source_key}|{group:?}")];
        key.extend(plan.steps.iter().map(|(k, _)| k.clone()));
        for (at, fit) in &plan.fits {
            let up_to = key[..at + 2].join("\n");
            match lock(&self.fitted).get(&up_to) {
                Some(w) => stages[*at] = PipelineStage::SpatialWhitening(w.clone()),
                None => return Ok(Made::NeedsFit(FitJob { key: up_to, source, group, before: stages[..*at].to_vec(), fit: *fit })),
            }
        }
        // The longest kept prefix of this chain
        let found = {
            let made = lock(&self.made);
            (2..=key.len()).rev().find_map(|k| made.iter().find(|(c, _)| c[..] == key[..k]).map(|(_, p)| (k, p.clone())))
        };
        let built = match found {
            Some((k, p)) if k == key.len() => return Ok(Made::Ready(p)),
            // A branch reads the shared prefix's rows on the device, in the order it leaves them
            Some((k, p)) => ProcessedSignal::after(&p, stages[k - 1..].to_vec()),
            None => {
                // The recording read through the RAM cache: windows come back without decoding again
                let cached = self.cached_source(source, source_key);
                ProcessedSignal::with_budget(raw, cached, group, stages, self.inputs.clone(), self.budget.clone())
            }
        }
        .map(Arc::new)
        .map_err(|e| e.to_string())?;
        // Frozen once it stops changing, under its key
        built.set_summary_key(format!("chains:{}:summary:{}", self.id, key.join("\n")));
        let mut made = lock(&self.made);
        made.retain(|(c, _)| *c != key);
        if made.len() >= CHAINS_KEPT {
            made.remove(0);
        }
        made.push((key, built.clone()));
        Ok(Made::Ready(built))
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
        let signal = loop {
            match self.signal(raw.clone(), source.clone(), "", &plan)? {
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
