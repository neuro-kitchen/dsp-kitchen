//! `sort`: Kilosort4 or EMUsort over a recording (dsp-synapse-ml), written as a sorting.
//!
//! The settings people tune are flags whose defaults come from the sorter's config (`--help` shows
//! them); `--show-config` prints every setting of the run, including those without a flag.

use std::path::PathBuf;
use std::sync::LazyLock;
use std::time::Instant;

use clap::{Args, Subcommand};
use dsp_core::ComputeTarget;
use dsp_synapse_ml::sorters::kilosort4::{run_plan, RunPlan, UniversalTemplates};
use dsp_synapse_ml::{EmusortConfig, Kilosort4Config};

use super::recording::{OutputArgs, RecordingArgs};
use crate::progress::TerminalProgress;

static KILOSORT4: LazyLock<Kilosort4Config> = LazyLock::new(Kilosort4Config::default);
static EMUSORT: LazyLock<EmusortConfig> = LazyLock::new(EmusortConfig::default);

/// The shared sorter settings exposed as flags, with `$defaults` (a `Kilosort4Config` or an
/// `EmusortConfig`: same field names) as defaults, applied to a `$target` of either type.
macro_rules! kilosort4_settings {
    ($name:ident, $defaults:expr, $target:ty) => {
        #[derive(Args, Debug)]
        pub struct $name {
            /// Samples per waveform (odd); 61 is 2 ms at 30 kHz
            #[arg(long, default_value_t = $defaults.nt)]
            nt: usize,
            /// Waveform length in ms (overrides --nt: rounded to an odd sample count for the recording)
            #[arg(long)]
            nt_ms: Option<f64>,
            /// Sample a waveform's trough is aligned to; default: int(20 · nt / 61)
            #[arg(long)]
            nt0min: Option<usize>,
            /// Universal-template detection threshold (whitened σ)
            #[arg(long, default_value_t = $defaults.th_universal)]
            th_universal: f32,
            /// Learned-template matching threshold (whitened σ)
            #[arg(long, default_value_t = $defaults.th_learned)]
            th_learned: f32,
            /// Single-channel thresholds of the clips templates are learned from (whitened σ), comma-separated
            #[arg(long, value_delimiter = ',', default_values_t = $defaults.th_single_ch.clone())]
            th_single_ch: Vec<f32>,
            /// Learn the universal templates from the recording (false: Kilosort4's predefined ones,
            /// from --templates or the model hub)
            #[arg(long, default_value_t = $defaults.templates_from_data, action = clap::ArgAction::Set)]
            templates_from_data: bool,
            /// Universal templates learned
            #[arg(long, default_value_t = $defaults.n_templates)]
            n_templates: usize,
            /// Principal components per channel
            #[arg(long, default_value_t = $defaults.n_pcs)]
            n_pcs: usize,
            /// Every nskip-th batch is used for template learning
            #[arg(long, default_value_t = $defaults.nskip)]
            nskip: usize,
            /// Common average reference before filtering
            #[arg(long, default_value_t = $defaults.do_car, action = clap::ArgAction::Set)]
            do_car: bool,
            /// Butterworth band-pass (order 3 per edge, zero phase); false: no Butterworth
            #[arg(long, default_value_t = $defaults.do_bandpass, action = clap::ArgAction::Set)]
            do_bandpass: bool,
            /// Lower band edge (Hz)
            #[arg(long, default_value_t = $defaults.bandpass_low_hz)]
            bandpass_low_hz: f64,
            /// Upper band edge (Hz), below Nyquist; 0: no upper edge (a high-pass)
            #[arg(long, default_value_t = $defaults.bandpass_high_hz.unwrap_or(0.0))]
            bandpass_high_hz: f64,
            /// Line-noise notch after the band
            #[arg(long, default_value_t = $defaults.do_notch, action = clap::ArgAction::Set)]
            do_notch: bool,
            /// Notch frequency (Hz; 50 outside the Americas)
            #[arg(long, default_value_t = $defaults.notch_hz)]
            notch_hz: f64,
            /// Quality factor of the notch (bandwidth = notch / Q)
            #[arg(long, default_value_t = $defaults.notch_q)]
            notch_q: f64,
            /// Channels in each local whitening neighbourhood
            #[arg(long, default_value_t = $defaults.whitening_range)]
            whitening_range: usize,
            /// Samples per batch
            #[arg(long, default_value_t = $defaults.batch_size)]
            batch_size: usize,
            /// Matching-pursuit rounds per window
            #[arg(long, default_value_t = $defaults.max_peels)]
            max_peels: usize,
            /// Merge units whose waveforms are alike and whose spikes are mutually refractory
            #[arg(long, default_value_t = $defaults.global_merge.enabled, action = clap::ArgAction::Set)]
            global_merges: bool,
            /// Waveform similarity (correlation over lags) a pair needs to be tested for a merge
            #[arg(long, default_value_t = $defaults.global_merge.min_similarity)]
            merge_similarity: f64,
            /// A unit's spikes within this many ms of its previous spike are removed (0 keeps them)
            #[arg(long, default_value_t = $defaults.duplicate_spike_ms)]
            duplicate_spike_ms: f64,
            /// Same input on the same device gives the same spikes
            #[arg(long, default_value_t = $defaults.reproducible, action = clap::ArgAction::Set)]
            reproducible: bool,
        }

        impl $name {
            fn apply(&self, c: &mut $target) {
                c.nt = self.nt;
                c.nt_ms = self.nt_ms;
                c.nt0min = self.nt0min;
                c.th_universal = self.th_universal;
                c.th_learned = self.th_learned;
                c.th_single_ch = self.th_single_ch.clone();
                c.templates_from_data = self.templates_from_data;
                c.n_templates = self.n_templates;
                c.n_pcs = self.n_pcs;
                c.nskip = self.nskip;
                c.do_car = self.do_car;
                c.do_bandpass = self.do_bandpass;
                c.bandpass_low_hz = self.bandpass_low_hz;
                c.bandpass_high_hz = (self.bandpass_high_hz > 0.0).then_some(self.bandpass_high_hz);
                c.do_notch = self.do_notch;
                c.notch_hz = self.notch_hz;
                c.notch_q = self.notch_q;
                c.whitening_range = self.whitening_range;
                c.batch_size = self.batch_size;
                c.max_peels = self.max_peels;
                c.global_merge.enabled = self.global_merges;
                c.global_merge.min_similarity = self.merge_similarity;
                c.duplicate_spike_ms = self.duplicate_spike_ms;
                c.reproducible = self.reproducible;
            }
        }
    };
}

kilosort4_settings!(Kilosort4Settings, KILOSORT4, Kilosort4Config);
kilosort4_settings!(EmusortKilosort4Settings, EMUSORT, EmusortConfig);

/// What every sorter run takes besides its settings.
#[derive(Args, Debug)]
struct RunArgs {
    #[command(flatten)]
    recording: RecordingArgs,
    #[command(flatten)]
    output: OutputArgs,
    /// Predefined universal templates (`.npz` with `wPCA`, `wTEMP`) when --templates-from-data false
    #[arg(long)]
    templates: Option<PathBuf>,
    /// Print every setting of the run and exit
    #[arg(long)]
    show_config: bool,
}

#[derive(Args, Debug)]
pub struct Kilosort4Args {
    #[command(flatten)]
    run: RunArgs,
    #[command(flatten)]
    settings: Kilosort4Settings,
}

#[derive(Args, Debug)]
pub struct EmusortArgs {
    #[command(flatten)]
    run: RunArgs,
    #[command(flatten)]
    settings: EmusortKilosort4Settings,
    /// Estimate and remove per-channel delays (up to 2 ms)
    #[arg(long, default_value_t = EMUSORT.remove_channel_delays, action = clap::ArgAction::Set)]
    remove_channel_delays: bool,
    /// Remove outlier clips (HDBSCAN) before learning the universal templates
    #[arg(long, default_value_t = EMUSORT.remove_spike_outliers, action = clap::ArgAction::Set)]
    remove_spike_outliers: bool,
    /// HDBSCAN min_cluster_size of the outlier removal
    #[arg(long, default_value_t = EMUSORT.hdbscan_min_cluster_size)]
    hdbscan_min_cluster_size: usize,
}

#[derive(Subcommand, Debug)]
pub enum SortCommand {
    /// Kilosort4 (Pachitariu et al. 2024): high-density extracellular probes
    Kilosort4(Kilosort4Args),
    /// EMUsort (O'Connell et al. 2026): high-density intramuscular arrays, motor units
    Emusort(EmusortArgs),
}

#[derive(Args, Debug)]
pub struct SortArgs {
    #[command(subcommand)]
    sorter: SortCommand,
}

pub fn run(target: ComputeTarget, args: &SortArgs) -> anyhow::Result<()> {
    match &args.sorter {
        SortCommand::Kilosort4(a) => {
            let mut config = KILOSORT4.clone();
            a.settings.apply(&mut config);
            if a.run.show_config {
                println!("{config:#?}");
                return Ok(());
            }
            execute(target, &a.run, |_| RunPlan::kilosort4(&config))
        }
        SortCommand::Emusort(a) => {
            let mut config = EMUSORT.clone();
            a.settings.apply(&mut config);
            config.remove_channel_delays = a.remove_channel_delays;
            config.remove_spike_outliers = a.remove_spike_outliers;
            config.hdbscan_min_cluster_size = a.hdbscan_min_cluster_size;
            if a.run.show_config {
                println!("{config:#?}");
                return Ok(());
            }
            execute(target, &a.run, |fs| RunPlan::emusort(&config, fs))
        }
    }
}

/// Opens the recording, runs the plan (`plan(sample_rate_hz)`) with a progress bar, writes the sorting.
fn execute(target: ComputeTarget, args: &RunArgs, plan: impl Fn(f64) -> RunPlan) -> anyhow::Result<()> {
    let opened = args.recording.open()?;
    let info = opened.source.info();
    let mut plan = plan(info.sample_rate_hz());
    if let Some(path) = &args.templates {
        plan.templates = Some(UniversalTemplates::from_npz(path)?);
    }
    println!("{} on {} channels × {:.1} s, {}", plan.sorter, info.channel_count(), info.duration_sec(), target.name());
    let client = target.client()?;
    let t = Instant::now();
    let result = run_plan(&client, opened.source.as_ref(), &opened.probe, &plan, &TerminalProgress::default())?;
    println!(
        "{} units, {} spikes ({} merges, {} duplicates removed) in {:.1} s",
        result.clusters.n_units,
        result.spikes.len(),
        result.merges,
        result.duplicates,
        t.elapsed().as_secs_f64()
    );
    args.output.save(&result.to_sorting_output(Some(opened.probe)))
}
