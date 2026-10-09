//! Arguments shared by the commands that read a recording and write a sorting (`detect`, `sort`,
//! `convert`): which recording, source, probe and time window; where and in which format to write.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{bail, Context};
use clap::{Args, ValueEnum};
use dsp_core::recording::SlicedRecording;
use dsp_core::RecordingSource;
use dsp_io::neuro::SortingFormat;
use dsp_io::SensorLayout;
use dsp_synapse::SortingOutput;

use super::probe::{Preset, DEFAULT_HDEMG_PITCH_UM};

/// A recording, the source in it, its probe and the part to process.
#[derive(Args, Debug)]
pub struct RecordingArgs {
    /// Recording (SpikeGLX .bin / .cbin, raw .bin + JSON .meta, NWB, Zarr)
    pub recording: PathBuf,
    /// Source within the file (an id `dsp-cli open` lists); default: the largest electrical one
    #[arg(long)]
    pub source: Option<String>,
    /// Probe preset; default: the geometry the recording carries (SpikeGLX), else an error
    #[arg(long, value_enum)]
    pub probe: Option<Preset>,
    /// Electrode pitch of HD-EMG grid presets (µm)
    #[arg(long, default_value_t = DEFAULT_HDEMG_PITCH_UM)]
    pub pitch_um: f32,
    /// Start of the part to process (s)
    #[arg(long, default_value_t = 0.0)]
    pub start_sec: f64,
    /// Length of the part to process (s); default: to the end
    #[arg(long)]
    pub duration_sec: Option<f64>,
}

/// An opened recording (the part asked for) and its probe.
pub struct Opened {
    pub source: Arc<dyn RecordingSource>,
    pub probe: SensorLayout,
}

impl RecordingArgs {
    /// Opens the source, cuts the time window and finds the probe.
    pub fn open(&self) -> anyhow::Result<Opened> {
        let path = self.recording.as_path();
        let id = match &self.source {
            Some(id) => id.clone(),
            None => {
                let list = dsp_io::sources(path)?;
                dsp_io::default_source(&list).with_context(|| format!("{} has no sources", path.display()))?.id.clone()
            }
        };
        let full: Arc<dyn RecordingSource> = Arc::from(dsp_io::open_source(path, &id)?);
        let info = full.info();
        let (fs, total) = (info.sample_rate_hz(), info.samples);
        let start = seconds_to_samples(self.start_sec, fs).min(total);
        let end = match self.duration_sec {
            Some(d) => start.saturating_add(seconds_to_samples(d, fs)).min(total),
            None => total,
        };
        if start >= end {
            bail!("the window starts at {} s, past the end of the recording ({:.1} s)", self.start_sec, info.duration_sec());
        }
        let channels = info.channel_count();
        let source: Arc<dyn RecordingSource> =
            if start == 0 && end == total { full } else { Arc::new(SlicedRecording::new(full, start..end, None)?) };
        let probe = match self.probe {
            Some(preset) => preset.layout(self.pitch_um),
            None => dsp_io::probe_of(path, &id)?.with_context(|| {
                format!("{} carries no probe geometry: choose one with --probe", path.display())
            })?,
        };
        if probe.total_channels() != channels {
            bail!("the probe has {} sites but the recording {channels} channels", probe.total_channels());
        }
        Ok(Opened { source, probe })
    }
}

fn seconds_to_samples(sec: f64, fs: f64) -> u64 {
    (sec.max(0.0) * fs).round() as u64
}

/// Sorting file formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum Format {
    /// Phy / Kilosort folder (`.npy` arrays, `.tsv` tables)
    Phy,
    /// dsp-kitchen `.sorting.zarr` store
    Zarr,
    /// NWB `/units` table in a `.nwb.zarr` store
    Nwb,
}

impl From<Format> for SortingFormat {
    fn from(f: Format) -> Self {
        match f {
            Format::Phy => SortingFormat::Phy,
            Format::Zarr => SortingFormat::SortingZarr,
            Format::Nwb => SortingFormat::NwbUnits,
        }
    }
}

/// Where a sorting is written.
#[derive(Args, Debug)]
pub struct OutputArgs {
    /// Output folder or store
    #[arg(short, long)]
    pub output: PathBuf,
    /// Output format; default: from the name (`.nwb.zarr` → nwb, other `.zarr` → zarr, else phy)
    #[arg(long, value_enum)]
    pub format: Option<Format>,
}

impl OutputArgs {
    /// Writes `sorting` and prints a one-line summary.
    pub fn save(&self, sorting: &SortingOutput) -> anyhow::Result<()> {
        save(sorting, &self.output, self.format)
    }
}

/// Writes `sorting` at `path` as `format` (default: from the name) and prints a summary.
pub fn save(sorting: &SortingOutput, path: &Path, format: Option<Format>) -> anyhow::Result<()> {
    let format = format.map(SortingFormat::from).unwrap_or_else(|| SortingFormat::from_path_name(path));
    dsp_synapse::storage::save_sorting(sorting, path, Some(format))?;
    let spikes: usize = sorting.units.iter().map(|u| u.spike_samples.len()).sum();
    println!("Wrote {} units, {spikes} spikes to {} ({format:?})", sorting.units.len(), path.display());
    Ok(())
}
