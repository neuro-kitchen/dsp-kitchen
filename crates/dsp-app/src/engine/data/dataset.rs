//! The open recording: any `dsp-io` format behind [`RecordingSource`], read in chunks.

use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};

use anyhow::Result;
use dsp_core::{DspResult, RecordingInfo, RecordingSource};
use dsp_base::resampler::cache::DEFAULT_BASE;
use dsp_base::resampler::minmax::mean_range;
use dsp_base::resampler::{cache_path, CacheIdentity, MinMaxCache, MinMaxSummary, OnProgress, Summarizer};
use dsp_io::{SyntheticParams, SyntheticRecording};



/// A recording plus the summary fields the UI reads every frame.
pub struct Dataset {
    source: Arc<dyn RecordingSource>,
    pub total_channels: usize,
    pub total_samples: usize,
    pub sample_rate: f64,
    pub name: String,
    /// Session time of sample 0 (sources of one file can start at different times).
    pub start_time_sec: f64,
    /// Unit of the values reads return (`µV` for electrical recordings).
    pub unit: String,
    /// Min/max of the regions views have shown this session (zoomed-out drawing).
    summary: Arc<MinMaxSummary>,
    /// Min/max levels for zoomed-out drawing: an existing cache file, or one the user asked to build.
    lod: Arc<OnceLock<Arc<MinMaxCache>>>,
    /// Set once a build has been started (builds run at most once).
    lod_building: Arc<AtomicBool>,
    /// Where to write a cache once the summary is complete (large recordings; see
    /// [`Self::cache_after_summary`]).
    cache_later: std::sync::Mutex<Option<Option<(PathBuf, String)>>>,
    /// Stops the background build when the dataset is dropped.
    cancel: Arc<AtomicBool>,
    /// Fills `summary` in the background (once, for every view).
    summarizer: Summarizer,
}

impl Dataset {
    pub fn new(source: Arc<dyn RecordingSource>) -> Self {
        let info = source.info();
        Self {
            total_channels: info.channel_count(),
            total_samples: info.samples as usize,
            sample_rate: info.sample_rate_hz(),
            name: info.name.clone(),
            start_time_sec: info.start_time_sec,
            unit: info.metadata.get("unit").map_or_else(|| "µV".into(), |u| u.replace("uV", "µV")),
            summary: Arc::new(MinMaxSummary::new(source.as_ref())),
            source,
            lod: Arc::new(OnceLock::new()),
            lod_building: Arc::new(AtomicBool::new(false)),
            cache_later: std::sync::Mutex::new(None),
            cancel: Arc::new(AtomicBool::new(false)),
            summarizer: Summarizer::default(),
        }
    }

    /// Summarizes the whole recording in the background, nearest sample `focus` first (later
    /// calls only move the focus). Not needed once a complete min/max cache file is open. When the
    /// summary completes, the cache asked for by [`Self::cache_after_summary`] is written.
    pub fn summarize(&self, focus: u64, on_progress: OnProgress) {
        if self.lod().is_some() {
            return;
        }
        let later = self.cache_later.lock().expect("dataset lock").take();
        let build = later.map(|recording| (recording, self.source.clone(), self.lod.clone(), self.cancel.clone(), self.lod_building.clone()));
        let build = std::sync::Mutex::new(build);
        let on_progress: OnProgress = Arc::new(move |p| {
            let complete = p.done >= p.total;
            on_progress(p);
            // The file is read once for the summary, then once more for the cache, never both at once
            if complete && let Some((recording, source, slot, cancel, building)) = build.lock().expect("dataset lock").take() {
                spawn_lod_build(source, slot, cancel, &building, recording);
            }
        });
        self.summarizer.run(self.source.clone(), self.summary.clone(), focus, self.cancel.clone(), on_progress);
    }

    /// Writes the min/max cache (next to `recording`) once the background summary is complete,
    /// so the two never read the file at the same time; the next open then zooms out at once.
    pub fn cache_after_summary(&self, recording: Option<(PathBuf, String)>) {
        *self.cache_later.lock().expect("dataset lock") = Some(recording);
    }

    /// Uses the complete min/max cache already next to `path` for source `id`, if there is one
    /// (nothing is built).
    pub fn open_lod(&self, path: &Path, id: &str) {
        let found = CacheIdentity::of(path, id, self.source.as_ref()).and_then(|identity| MinMaxCache::open(&cache_path(path, id), &identity, DEFAULT_BASE));
        match found {
            Ok(Some(cache)) => {
                let _ = self.lod.set(Arc::new(cache));
            }
            Ok(None) => {}
            Err(e) => tracing::warn!("could not open the min/max cache of {}: {e}", path.display()),
        }
    }

    /// Builds the min/max cache on a background thread (user request): next to the recording file
    /// for `Some((path, source id))` (a temporary file when that folder is not writable), else a
    /// temporary file. A no-op once a build started or a complete cache is open.
    pub fn build_lod(&self, recording: Option<(PathBuf, String)>) {
        if self.lod.get().is_some_and(|c| c.is_complete()) {
            return;
        }
        spawn_lod_build(self.source.clone(), self.lod.clone(), self.cancel.clone(), &self.lod_building, recording);
    }

    /// The min/max cache file once complete (while it builds, views use the session summary).
    pub fn lod(&self) -> Option<Arc<MinMaxCache>> {
        self.lod.get().filter(|c| c.is_complete()).cloned()
    }

    pub fn summary(&self) -> Arc<MinMaxSummary> {
        self.summary.clone()
    }

    /// Physical sensor site of `channel` when the recording has a probe layout.
    pub fn site(&self, channel: usize) -> Option<&dsp_core::SensorSite> {
        self.source.info().layout.as_ref()?.get_site(channel).ok()
    }

    /// Distinct shank ids in ascending order when the recording has a probe layout.
    pub fn shanks(&self) -> Vec<usize> {
        let Some(layout) = &self.source.info().layout else { return Vec::new() };
        let mut shanks: Vec<usize> = layout.contacts.iter().map(|s| s.shank_id).collect();
        shanks.sort_unstable();
        shanks.dedup();
        shanks
    }

    /// Opens any format `dsp-io` detects (SpikeGLX, IBL `.cbin`, raw binary + JSON sidecar, Zarr).
    #[cfg(test)]
    pub fn open(path: &std::path::Path) -> Result<Self> {
        use anyhow::Context;
        let source = dsp_io::open(path).with_context(|| format!("Failed to open {}", path.display()))?;
        Ok(Self::new(Arc::from(source)))
    }

    /// Procedural recording of any length (noise, hum, drifting units), computed on demand.
    pub fn procedural(channels: usize, sample_rate: f64, duration_sec: f64) -> Result<Self> {
        let units = (channels / 4).clamp(1, 64);
        let rec = SyntheticRecording::new(SyntheticParams { channels, sample_rate_hz: sample_rate, duration_sec, units, ..Default::default() })?;
        Ok(Self::new(Arc::new(rec)))
    }

    /// Wraps an in-memory channel-major buffer.
    #[cfg(test)]
    pub fn from_samples(name: &str, data: Vec<f32>, total_channels: usize, sample_rate: f64) -> Self {
        Self::new(Arc::new(dsp_core::MemoryRecording::new(name, data, total_channels, sample_rate).expect("valid shape")))
    }

    #[cfg(test)]
    pub fn total_duration_sec(&self) -> f64 {
        if self.sample_rate > 0.0 {
            self.total_samples as f64 / self.sample_rate
        } else {
            0.0
        }
    }

    /// One sample in µV (0 when out of range or unreadable).
    #[cfg(test)]
    pub fn sample(&self, channel: usize, sample: usize) -> f32 {
        let mut v = [0.0f32];
        let s = sample as u64;
        match self.source.read(&[channel], s..s + 1, &mut v) {
            Ok(()) => v[0],
            Err(_) => 0.0,
        }
    }

    /// Every channel over `samples`, channel-major.
    #[cfg(test)]
    pub fn read_all(&self, samples: Range<usize>) -> DspResult<Vec<f32>> {
        let channels: Vec<usize> = (0..self.total_channels).collect();
        let mut out = vec![0.0f32; channels.len() * samples.len()];
        self.source.read(&channels, samples.start as u64..samples.end as u64, &mut out)?;
        Ok(out)
    }

    /// Synthetic 32-channel-style signal (hum, noise, a spike every ~300 ms per channel) held in
    /// memory; tests rely on its exact content.
    #[cfg(test)]
    pub fn generate_synthetic(total_channels: usize, sample_rate: f64, duration_sec: f64) -> Self {
        let total_samples = (sample_rate * duration_sec) as usize;
        let mut raw_data = vec![0.0f32; total_channels * total_samples];

        let dt = 1.0 / sample_rate;
        let omega_60 = 2.0 * std::f64::consts::PI * 60.0;
        for ch in 0..total_channels {
            let ch_offset = ch * total_samples;
            let phase = (ch as f64 * 0.1).fract() * 2.0 * std::f64::consts::PI;
            for s in 0..total_samples {
                let t = s as f64 * dt;
                let hum = (25.0 * (omega_60 * t + phase).sin()) as f32;
                let noise = (((s * 37 + ch * 101) % 1000) as f32 / 1000.0 - 0.5) * 20.0;
                raw_data[ch_offset + s] = hum + noise;
            }

            // Inject action potentials every ~300ms
            let spike_interval = (sample_rate * 0.3).max(1.0) as usize;
            for sp in 1..(total_samples / spike_interval) {
                let center = sp * spike_interval + (ch * 17) % 500;
                if center + 40 < total_samples {
                    for i in 0..40 {
                        let t_rel = (i as f32 - 15.0) / 6.0;
                        let shape = -90.0 * (-0.5 * t_rel * t_rel).exp();
                        raw_data[ch_offset + center + i] += shape;
                    }
                }
            }
        }

        let rec = dsp_core::MemoryRecording::new(format!("synthetic_{total_channels}ch"), raw_data, total_channels.max(1), sample_rate)
            .expect("valid synthetic shape");
        Self::new(Arc::new(rec))
    }
}

impl Drop for Dataset {
    fn drop(&mut self) {
        self.cancel.store(true, Ordering::Relaxed);
    }
}

/// Starts the cache build on its own thread, once (`building` is set by the first call).
fn spawn_lod_build(source: Arc<dyn RecordingSource>, slot: Arc<OnceLock<Arc<MinMaxCache>>>, cancel: Arc<AtomicBool>, building: &AtomicBool, recording: Option<(PathBuf, String)>) {
    if building.swap(true, Ordering::Relaxed) {
        return;
    }
    let spawned = std::thread::Builder::new().name("minmax-cache".into()).spawn(move || {
        if let Err(e) = fill_lod(source.as_ref(), recording.as_ref().map(|(p, id)| (p.as_path(), id.as_str())), &slot, &cancel) {
            tracing::warn!("min/max cache unavailable, zoomed-out views read raw samples: {e}");
        }
    });
    if let Err(e) = spawned {
        tracing::warn!("could not start the min/max cache build: {e}");
    }
}

fn fill_lod(source: &dyn RecordingSource, recording: Option<(&Path, &str)>, slot: &OnceLock<Arc<MinMaxCache>>, cancel: &AtomicBool) -> DspResult<()> {
    let (cache, complete) = MinMaxCache::open_or_create(source, recording, DEFAULT_BASE)?;
    let cache = Arc::new(cache);
    let _ = slot.set(cache.clone());
    if !complete {
        // One second of data per read
        let chunk = source.info().sample_rate_hz().ceil().max(1.0) as u64;
        cache.build(source, chunk, cancel, |_, _| {})?;
    }
    Ok(())
}

/// Views and workers hold `Arc<Dataset>` as a plain `RecordingSource`.
impl RecordingSource for Dataset {
    fn info(&self) -> &RecordingInfo {
        self.source.info()
    }

    fn chunk_samples(&self) -> Option<u64> {
        self.source.chunk_samples()
    }

    fn read(&self, channels: &[usize], samples: Range<u64>, out: &mut [f32]) -> DspResult<()> {
        self.source.read(channels, samples, out)
    }

    fn read_stored(&self, channels: &[usize], samples: Range<u64>, out: &mut [u8]) -> DspResult<()> {
        self.source.read_stored(channels, samples, out)
    }
}

/// Activity of the whole recording in `width` columns, for the timeline overview: the mean
/// min-to-max range over channels per column (from the cache file, else the summary), scaled so
/// the busiest column is 1. Columns not summarized yet are NaN.
pub fn activity(ds: &Dataset, width: usize) -> Vec<f32> {
    let channels: Vec<usize> = (0..ds.total_channels).collect();
    let total = ds.total_samples as u64;
    if width == 0 || channels.is_empty() || total == 0 {
        return Vec::new();
    }
    let mut env = vec![[f32::NAN, f32::NAN]; channels.len() * width];
    let from_cache = ds.lod().is_some_and(|c| matches!(c.envelope(&channels, 0, total, width, &mut env), Ok(true)));
    if !from_cache {
        ds.summary().envelope(&channels, 0, total, width, &mut env);
    }
    let mut out = mean_range(&env, channels.len(), width);
    let max = out.iter().copied().filter(|v| v.is_finite()).fold(0.0f32, f32::max);
    if max > 0.0 {
        out.iter_mut().for_each(|v| *v /= max);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_opens_raw_file_through_dsp_io() {
        let dir = std::env::temp_dir().join(format!("dsp_app_ds_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("rec.bin");
        let values: Vec<f32> = (0..12).map(|v| v as f32).collect();
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(&path, bytes).unwrap();
        std::fs::write(path.with_extension("meta"), r#"{"channels": 3, "sample_rate_hz": 1000.0}"#).unwrap();

        let ds = Dataset::open(&path).unwrap();
        assert_eq!(ds.total_samples, 4);
        assert_eq!(ds.sample(1, 2), 6.0);
        assert_eq!(ds.read_all(1..3).unwrap(), vec![1.0, 2.0, 5.0, 6.0, 9.0, 10.0]);
        // Out of range reads are zero, not a panic
        assert_eq!(ds.sample(7, 0), 0.0);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_synthetic_dataset_generation() {
        let ds = Dataset::generate_synthetic(8, 10_000.0, 1.0);
        assert_eq!(ds.total_channels, 8);
        assert_eq!(ds.total_samples, 10_000);
        assert!((ds.total_duration_sec() - 1.0).abs() < 1e-6);

        let long = Dataset::procedural(384, 30_000.0, 4.0 * 3600.0).unwrap();
        assert_eq!(long.total_samples, 432_000_000);
    }

    /// Time to summarize a whole recording in the background. Run with:
    /// `DSP_APP_BENCH_FILE=<recording> cargo test -p dsp-app --release bench_summarize -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn bench_summarize() {
        let Some(path) = std::env::var_os("DSP_APP_BENCH_FILE") else { return };
        let sources = crate::engine::data::SourceSet::open(Path::new(&path)).unwrap();
        let ds = sources.default_dataset();
        let done = Arc::new(AtomicBool::new(false));
        let d = done.clone();
        let t0 = std::time::Instant::now();
        ds.summarize(0, Arc::new(move |p| d.store(p.done >= p.total, Ordering::Relaxed)));
        while !done.load(Ordering::Relaxed) {
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
        let mb = ds.total_samples as f64 * ds.total_channels as f64 * 2.0 / 1e6;
        let secs = t0.elapsed().as_secs_f64();
        println!("{} ch × {} samples ({mb:.0} MB int16): {secs:.2} s ({:.0} MB/s)", ds.total_channels, ds.total_samples, mb / secs);
    }
}
