//! `probe`: a probe layout preset (dsp-io `neuro::probe`).

use clap::{Args, ValueEnum};
use dsp_io::neuro::probe::{self, SensorLayout};

/// Default electrode pitch of the HD-EMG grid presets (µm).
pub const DEFAULT_HDEMG_PITCH_UM: f32 = 8_000.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum Preset {
    Neuropixels1,
    Neuropixels2,
    Tetrode,
    Utah,
    Hdemg4x8,
    Hdemg8x8,
}

#[derive(Args, Debug)]
pub struct ProbeArgs {
    /// Layout preset
    #[arg(value_enum)]
    preset: Preset,
    /// Electrode pitch of HD-EMG grids (µm)
    #[arg(long, default_value_t = DEFAULT_HDEMG_PITCH_UM)]
    pitch_um: f32,
}

impl Preset {
    /// The preset's layout; `pitch_um` sets the HD-EMG grids' electrode pitch.
    pub fn layout(self, pitch_um: f32) -> SensorLayout {
        match self {
            Preset::Neuropixels1 => probe::neuropixels_1_0(),
            Preset::Neuropixels2 => probe::neuropixels_2_0(),
            Preset::Tetrode => probe::tetrode(),
            Preset::Utah => probe::utah_array(),
            Preset::Hdemg4x8 => probe::hdemg_4x8(pitch_um),
            Preset::Hdemg8x8 => probe::hdemg_8x8(pitch_um),
        }
    }
}

pub fn run(args: &ProbeArgs) -> anyhow::Result<()> {
    let layout = args.preset.layout(args.pitch_um);
    let sites = layout.sites();
    let span = |coord: fn(&probe::SensorSite) -> f32| {
        let (lo, hi) = sites.iter().map(coord).fold((f32::INFINITY, f32::NEG_INFINITY), |(a, b), v| (a.min(v), b.max(v)));
        if lo <= hi { hi - lo } else { 0.0 }
    };
    let shanks = sites.iter().map(|s| s.shank_id).max().map_or(0, |m| m + 1);
    println!("{}", layout.name);
    println!("Sites:    {} ({} active), {shanks} shank(s)", layout.total_channels(), layout.active_channels());
    println!("Extent:   {:.0} µm × {:.0} µm", span(|s| s.position.x_um), span(|s| s.position.y_um));
    for site in sites.iter().take(1).chain(sites.last()) {
        let p = &site.position;
        println!("  channel {:>4}: ({:.1}, {:.1}, {:.1}) µm, shank {}", site.channel_id, p.x_um, p.y_um, p.z_um, site.shank_id);
    }
    Ok(())
}
