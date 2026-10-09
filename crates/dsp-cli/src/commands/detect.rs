//! `detect`: streaming spike detection over a whole recording (dsp-synapse `StreamingDetector`),
//! written as a sorting with one unit per primary channel.

use std::sync::LazyLock;
use std::time::Instant;

use clap::{Args, ValueEnum};
use dsp_base::{Pipeline, PipelineStage};
use dsp_core::ComputeTarget;
use dsp_synapse::streaming::{StreamingDetectionConfig, StreamingDetector};
use dsp_synapse::{DistanceRule, SpikePolarity};

use super::recording::{OutputArgs, RecordingArgs};

/// Sorter name recorded in the output.
const SORTER_NAME: &str = "dsp-synapse streaming detection";
/// Default band-pass of the preprocessing (Hz).
const DEFAULT_BAND_LOW_HZ: f64 = 300.0;
const DEFAULT_BAND_HIGH_HZ: f64 = 5_000.0;

/// Library defaults, shown in `--help`.
static DEFAULTS: LazyLock<StreamingDetectionConfig> = LazyLock::new(StreamingDetectionConfig::default);

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum Polarity {
    Negative,
    Positive,
    Both,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum Spacing {
    /// A crossing stays unless a larger one is nearer (streaming equals whole-recording detection)
    LocallyExclusive,
    /// As `scipy.signal.find_peaks(distance=…)` (may differ at batch edges)
    Scipy,
}

#[derive(Args, Debug)]
pub struct DetectArgs {
    #[command(flatten)]
    recording: RecordingArgs,
    #[command(flatten)]
    output: OutputArgs,

    /// Common average reference before filtering
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    car: bool,
    /// Band-pass low edge (Hz)
    #[arg(long, default_value_t = DEFAULT_BAND_LOW_HZ)]
    band_low_hz: f64,
    /// Band-pass high edge (Hz); must be below Nyquist
    #[arg(long, default_value_t = DEFAULT_BAND_HIGH_HZ)]
    band_high_hz: f64,

    /// Threshold in multiples of each channel's noise σ (MAD)
    #[arg(long, default_value_t = DEFAULTS.threshold_factor)]
    threshold: f32,
    /// Smallest interval between two spikes on a channel (ms)
    #[arg(long, default_value_t = DEFAULTS.refractory_ms)]
    refractory_ms: f64,
    /// Which extrema are spikes
    #[arg(long, value_enum, default_value_t = Polarity::Negative)]
    polarity: Polarity,
    /// Which of two nearer crossings stays
    #[arg(long, value_enum, default_value_t = Spacing::LocallyExclusive)]
    spacing: Spacing,
    /// Crossings within this radius (and the refractory window) are one spike (µm)
    #[arg(long, default_value_t = DEFAULTS.spatial_radius_um)]
    radius_um: f32,
    /// Channels per waveform snippet (the primary and its nearest)
    #[arg(long, default_value_t = DEFAULTS.k_neighbors)]
    k_neighbors: usize,
    /// Snippet window before the trough (ms)
    #[arg(long, default_value_t = DEFAULTS.pre_ms)]
    pre_ms: f64,
    /// Snippet window after the trough (ms)
    #[arg(long, default_value_t = DEFAULTS.post_ms)]
    post_ms: f64,
    /// Realign snippets to the sub-sample trough
    #[arg(long, default_value_t = DEFAULTS.apply_sinc_shift, action = clap::ArgAction::Set)]
    sinc_shift: bool,
    /// Length of each processing window (s); memory only, not results
    #[arg(long, default_value_t = DEFAULTS.batch_duration_sec)]
    batch_sec: f64,
    /// Total length of the noise calibration (s)
    #[arg(long, default_value_t = DEFAULTS.calibration_duration_sec)]
    calibration_sec: f64,
    /// Chunks the noise calibration is spread over
    #[arg(long, default_value_t = DEFAULTS.calibration_chunks)]
    calibration_chunks: usize,
}

impl DetectArgs {
    fn config(&self) -> StreamingDetectionConfig {
        StreamingDetectionConfig {
            batch_duration_sec: self.batch_sec,
            calibration_duration_sec: self.calibration_sec,
            calibration_chunks: self.calibration_chunks,
            threshold_factor: self.threshold,
            refractory_ms: self.refractory_ms,
            polarity: match self.polarity {
                Polarity::Negative => SpikePolarity::Negative,
                Polarity::Positive => SpikePolarity::Positive,
                Polarity::Both => SpikePolarity::Both,
            },
            distance_rule: match self.spacing {
                Spacing::LocallyExclusive => DistanceRule::LocallyExclusive,
                Spacing::Scipy => DistanceRule::Scipy,
            },
            spatial_radius_um: self.radius_um,
            k_neighbors: self.k_neighbors,
            pre_ms: self.pre_ms,
            post_ms: self.post_ms,
            apply_sinc_shift: self.sinc_shift,
        }
    }

    fn pipeline(&self) -> Pipeline {
        let mut pipeline = Pipeline::new();
        if self.car {
            pipeline.add(PipelineStage::CommonAverageReference);
        }
        pipeline.add(PipelineStage::bandpass(self.band_low_hz, self.band_high_hz));
        pipeline
    }
}

pub fn run(target: ComputeTarget, args: &DetectArgs) -> anyhow::Result<()> {
    let opened = args.recording.open()?;
    let info = opened.source.info();
    let pipeline = args.pipeline();
    pipeline.validate(info.sample_rate_hz())?;
    println!(
        "Detecting in {} channels × {:.1} s on {}",
        info.channel_count(),
        info.duration_sec(),
        target.name()
    );
    let t = Instant::now();
    let result = StreamingDetector::new(args.config()).run_with(target, opened.source.as_ref(), &pipeline, &opened.probe)?;
    println!("{} spikes in {:.1} s", result.total_dedup_spikes, t.elapsed().as_secs_f64());
    args.output.save(&result.to_sorting_output(SORTER_NAME, Some(opened.probe)))
}
