//! Arguments shared by the commands that read a recording and write a sorting (`detect`, `sort`,
//! `convert`): which recording, source, probe and time window; where and in which format to write.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{bail, Context};
use clap::{Args, ValueEnum};
use dspk_core::recording::{RemappedSource, SlicedRecording};
use dspk_core::RecordingSource;
use dspk_synapse::io::SortingFormat;
use dspk_synapse::io::SensorLayout;
use dspk_synapse::SortingOutput;

use super::probe::{Preset, DEFAULT_HDEMG_PITCH_UM};

/// A recording, the source in it, its probe and the part to process.
#[derive(Args, Debug)]
pub struct RecordingArgs {
    /// Recording (SpikeGLX .bin / .cbin, raw .bin + JSON .meta, NWB, Zarr)
    pub recording: PathBuf,
    /// Source within the file (an id `dspk-cli open` lists); default: the largest electrical one
    #[arg(long)]
    pub source: Option<String>,
    /// Probe preset; default: the geometry the recording carries (SpikeGLX), else an error
    #[arg(long, value_enum, conflicts_with = "probe_file")]
    pub probe: Option<Preset>,
    /// Probe file (.json: probeinterface or Kilosort 4) saying which channel each contact is
    /// wired to: the data is read in contact order (unconnected contacts left out), with its
    /// geometry
    #[arg(long)]
    pub probe_file: Option<PathBuf>,
    /// Channel map (.toml / .json, pairs under `remap`) applied after the probe: channels
    /// reordered, dropped, renamed; the geometry follows them
    #[arg(long)]
    pub channel_map: Option<PathBuf>,
    /// Electrode pitch of HD-EMG grid presets (µm)
    #[arg(long, default_value_t = DEFAULT_HDEMG_PITCH_UM)]
    pub pitch_um: f32,
    /// Start of the part to process (s)
    #[arg(long, default_value_t = 0.0)]
    pub start_sec: f64,
    /// Length of the part to process (s); default: to the end
    #[arg(long)]
    pub duration_sec: Option<f64>,
    /// Channels to leave out (as Kilosort4's `bad_channels`), comma-separated, numbered after the
    /// probe and the channel map (0-based); their sites leave the probe too
    #[arg(long, value_delimiter = ',')]
    pub bad_channels: Vec<usize>,
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
                let list = dspk_synapse::io::sources(path)?;
                dspk_io::default_source(&list).with_context(|| format!("{} has no sources", path.display()))?.id.clone()
            }
        };
        let full: Arc<dyn RecordingSource> = Arc::from(dspk_synapse::io::open_source(path, &id)?);
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
        if let Some(file) = &self.probe_file {
            // The wiring picks and orders the channels itself
            let wiring = dspk_synapse::io::probe::read_probe_wiring(file)?;
            let timed: Arc<dyn RecordingSource> = if start == 0 && end == total { full } else { Arc::new(SlicedRecording::new(full, start..end, None)?) };
            let source: Arc<dyn RecordingSource> = Arc::new(RemappedSource::new(timed, &wiring.map).with_context(|| format!("{} on {}", file.display(), path.display()))?);
            return self.with_channel_map(Opened { source, probe: wiring.layout });
        }
        let probe = match self.probe {
            Some(preset) => preset.layout(self.pitch_um),
            None => dspk_synapse::io::probe_of(path, &id)?.with_context(|| {
                format!("{} carries no probe geometry: choose one with --probe", path.display())
            })?,
        };
        // Extra channels after the probe's sites (SpikeGLX's sync channel, the 385th): read only the
        // probe's, the first `sites` channels in order
        let sites = probe.total_channels();
        let in_order = probe.sites().iter().enumerate().all(|(i, s)| s.channel_id == i);
        let pick = match (sites.cmp(&channels), in_order) {
            (std::cmp::Ordering::Equal, _) => None,
            (std::cmp::Ordering::Less, true) => {
                eprintln!("note: the probe has {sites} sites, the recording {channels} channels: channels {sites}.. are left out");
                Some((0..sites).collect())
            }
            _ => bail!("the probe has {sites} sites but the recording {channels} channels"),
        };
        let source: Arc<dyn RecordingSource> = if start == 0 && end == total && pick.is_none() {
            full
        } else {
            Arc::new(SlicedRecording::new(full, start..end, pick)?)
        };
        self.with_channel_map(Opened { source, probe })
    }

    /// `opened` through `--channel-map`, when given: the source reads the mapped channels and the
    /// probe's sites follow them.
    fn with_channel_map(&self, opened: Opened) -> anyhow::Result<Opened> {
        let opened = match &self.channel_map {
            None => opened,
            Some(file) => {
                let map = dspk_io::read_channel_map(file)?;
                let probe = opened.probe.remap(&map).with_context(|| format!("{}: the probe", file.display()))?;
                let source: Arc<dyn RecordingSource> = Arc::new(RemappedSource::new(opened.source, &map).with_context(|| format!("{}: the recording", file.display()))?);
                Opened { source, probe }
            }
        };
        self.without_bad_channels(opened)
    }

    /// `opened` without `--bad-channels`: the source reads the others, the probe keeps their sites.
    fn without_bad_channels(&self, opened: Opened) -> anyhow::Result<Opened> {
        if self.bad_channels.is_empty() {
            return Ok(opened);
        }
        let channels = opened.source.info().channel_count();
        if let Some(&c) = self.bad_channels.iter().find(|&&c| c >= channels) {
            bail!("--bad-channels {c}: the recording has {channels} channels (0-based)");
        }
        let kept: Vec<usize> = (0..channels).filter(|c| !self.bad_channels.contains(c)).collect();
        if kept.is_empty() {
            bail!("--bad-channels leaves no channel");
        }
        let map = dspk_core::recording::ChannelMap::from_order(&kept);
        let probe = opened.probe.remap(&map).context("--bad-channels: the probe")?;
        let source: Arc<dyn RecordingSource> = Arc::new(RemappedSource::new(opened.source, &map).context("--bad-channels: the recording")?);
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
    /// dspk `.sorting.zarr` store
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
    /// Write only the units labelled good (as SpikeInterface's `keep_good_only`)
    #[arg(long)]
    pub keep_good_only: bool,
}

impl OutputArgs {
    /// Writes `sorting` and prints a one-line summary.
    pub fn save(&self, sorting: &SortingOutput) -> anyhow::Result<()> {
        if self.keep_good_only {
            let mut good = sorting.clone();
            good.units.retain(|u| u.quality_label == dspk_synapse::core::UnitQualityLabel::SingleUnit);
            return save(&good, &self.output, self.format);
        }
        save(sorting, &self.output, self.format)
    }
}

/// Writes `sorting` at `path` as `format` (default: from the name) and prints a summary.
pub fn save(sorting: &SortingOutput, path: &Path, format: Option<Format>) -> anyhow::Result<()> {
    let format = format.map(SortingFormat::from).unwrap_or_else(|| SortingFormat::from_path_name(path));
    dspk_synapse::storage::save_sorting(sorting, path, Some(format))?;
    let spikes: usize = sorting.units.iter().map(|u| u.spike_samples.len()).sum();
    println!("Wrote {} units, {spikes} spikes to {} ({format:?})", sorting.units.len(), path.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use dspk_core::{MemoryOrder, MemoryRecording, SampleFormat};

    use super::*;

    /// 4 channels × 100 samples, value = channel · 1000 + sample, as a raw file with its .meta.
    fn recording(dir: &Path) -> PathBuf {
        let data: Vec<f32> = (0..4).flat_map(|c| (0..100).map(move |s| (c * 1000 + s) as f32)).collect();
        let rec = MemoryRecording::new("r", data, 4, 1000.0).unwrap();
        let path = dir.join("r.bin");
        dspk_io::write_raw(&rec, &path, SampleFormat::F32, MemoryOrder::ChannelMajor, 1.0, 100, |_, _| {}).unwrap();
        path
    }

    fn args(recording: PathBuf, probe_file: Option<PathBuf>, channel_map: Option<PathBuf>, probe: Option<Preset>) -> RecordingArgs {
        RecordingArgs { recording, source: None, probe, probe_file, channel_map, pitch_um: DEFAULT_HDEMG_PITCH_UM, start_sec: 0.0, duration_sec: None, bad_channels: Vec::new() }
    }

    fn first_values(opened: &Opened) -> Vec<f32> {
        let n = opened.source.info().channel_count();
        let mut out = vec![0.0; n];
        opened.source.read(&(0..n).collect::<Vec<_>>(), 0..1, &mut out).unwrap();
        out
    }

    /// `--bad-channels` leaves those channels out of the data and their sites out of the probe;
    /// numbers past the recording's channels are refused.
    #[test]
    fn bad_channels_leave_the_data_and_the_probe() {
        let dir = std::env::temp_dir().join(format!("dspk-cli-bad-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let rec = recording(&dir);
        let mut a = args(rec.clone(), None, None, Some(Preset::Tetrode));
        a.bad_channels = vec![1, 3];
        let opened = a.open().unwrap();
        assert_eq!(first_values(&opened), vec![0.0, 2000.0]);
        let tetrode = Preset::Tetrode.layout(0.0);
        let kept: Vec<_> = opened.probe.sites().iter().map(|s| s.position).collect();
        assert_eq!(kept, vec![tetrode.get_site(0).unwrap().position, tetrode.get_site(2).unwrap().position]);
        a.bad_channels = vec![4];
        assert!(a.open().is_err(), "channel 4 of 4");
        a.bad_channels = vec![0, 1, 2, 3];
        assert!(a.open().is_err(), "no channel left");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_probe_file_and_a_channel_map_reorder_the_data_and_the_geometry_follows() {
        let dir = std::env::temp_dir().join(format!("dspk-cli-remap-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let rec = recording(&dir);
        // Contacts at y = 0, 20, 40, 60 wired to channels 3, 1, (none), 0
        let probe = dir.join("probe.json");
        std::fs::write(
            &probe,
            r#"{"specification": "probeinterface", "probes": [{"contact_positions": [[0, 0], [0, 20], [0, 40], [0, 60]], "device_channel_indices": [3, 1, -1, 0]}]}"#,
        )
        .unwrap();
        let opened = args(rec.clone(), Some(probe.clone()), None, None).open().unwrap();
        assert_eq!(first_values(&opened), vec![3000.0, 1000.0, 0.0]);
        let ys: Vec<f32> = opened.probe.sites().iter().map(|s| s.position.y_um).collect();
        assert_eq!(ys, vec![0.0, 20.0, 60.0]);
        // Then a map (1-based) keeping the last two, swapped: channels 0, 1 at y 60, 20
        let map = dir.join("map.toml");
        std::fs::write(&map, "[remap]\nbase = 1\npairs = [{ from = 3, to = 1 }, { from = 2, to = 2 }]\n").unwrap();
        let opened = args(rec.clone(), Some(probe), Some(map.clone()), None).open().unwrap();
        assert_eq!(first_values(&opened), vec![0.0, 1000.0]);
        let ys: Vec<f32> = opened.probe.sites().iter().map(|s| s.position.y_um).collect();
        assert_eq!(ys, vec![60.0, 20.0]);
        // A map on a preset's geometry
        let opened = args(rec.clone(), None, Some(map), Some(Preset::Tetrode)).open().unwrap();
        assert_eq!(first_values(&opened), vec![2000.0, 1000.0]);
        assert_eq!(opened.probe.sites()[0].position, crate::commands::probe::Preset::Tetrode.layout(0.0).get_site(2).unwrap().position);
        // .mat probe files are refused
        let mat = dir.join("chanMap.mat");
        std::fs::write(&mat, b"MATLAB").unwrap();
        assert!(args(rec, Some(mat), None, None).open().is_err());
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
