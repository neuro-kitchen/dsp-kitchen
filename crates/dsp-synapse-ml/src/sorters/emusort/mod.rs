//! EMUsort (O'Connell et al., eLife 2026): Kilosort4 adapted to motor-unit action potentials
//! from high-density intramuscular arrays (Myomatrix). Written from the paper and the published
//! defaults — the upstream GPL-3.0 code is not ported.
//!
//! EMUsort is a Kilosort4 fork, so it reuses every stage of [`crate::sorters::kilosort4`] with
//! its own settings ([`EmusortConfig`]) and adds:
//! - **channel-delay removal** on the device ([`delays::ChannelDelayEstimator`],
//!   [`delays::ChannelAligner`]): the lag (within ±2 ms) aligning each channel with the reference
//!   channel that correlates best with all others, estimated on the high-passed data and removed
//!   after whitening, before template learning and detection;
//! - **HDBSCAN outlier removal** before the universal templates are clustered
//!   ([`EmusortConfig::learn_options`]).

use cubecl::prelude::Client;
use dsp_core::{DspResult, RecordingSource};
use dsp_io::neuro::probe::SensorLayout;

use dsp_synapse::NeuralBand;

use crate::provenance::{Attributed, Paper, Provenance, ProvenanceKind, UpstreamCode};
use crate::sorters::kilosort4::runner::{run_plan, Kilosort4Result, RunPlan};
use crate::sorters::kilosort4::{
    CentreOptions, ClipOptions, ClipScaling, ClusteringOptions, GlobalMergeOptions, Kilosort4Config, LearnOptions, TemplateMergeOptions,
};

/// Sorter name of an EMUsort run in [`dsp_synapse::core::SortingOutput`].
pub const EMUSORT_SORTER: &str = "emusort";

/// Largest channel delay searched: `fs / MAX_DELAY_DIVISOR` samples (2 ms).
pub const MAX_DELAY_DIVISOR: f64 = 500.0;
/// HDBSCAN `min_cluster_size` of outlier removal (`hdbscan_min_cluster_size`).
pub const HDBSCAN_MIN_CLUSTER_SIZE: usize = 20;

/// EMUsort settings: every setting of the run, with EMUsort's defaults (the paper's and
/// upstream's). The settings it shares with Kilosort4 mean the same as in [`Kilosort4Config`]
/// (see its fields); EMUsort's own come last. Nothing here is a nested Kilosort4 config, so
/// EMUsort can never silently run with Kilosort4's defaults: [`EmusortConfig::kilosort4`] builds the
/// runner's settings from these fields.
#[derive(Debug, Clone, PartialEq)]
pub struct EmusortConfig {
    /// Waveform window (samples, odd); see [`Kilosort4Config::nt`].
    pub nt: usize,
    /// Waveform window in ms, converted per recording; see [`Kilosort4Config::nt_ms`].
    pub nt_ms: Option<f64>,
    /// Trough sample of the window; see [`Kilosort4Config::nt0min`].
    pub nt0min: Option<usize>,
    /// Universal-template detection threshold (whitened σ). EMUsort: upstream's 9.
    pub th_universal: f32,
    /// Learned-template matching threshold (whitened σ). EMUsort: upstream's 8.
    pub th_learned: f32,
    /// Single-channel clip thresholds, pooled (whitened σ). EMUsort: `[6, 9, 12, 15]`.
    pub th_single_ch: Vec<f32>,
    /// Learn the universal templates from the recording (EMUsort always does).
    pub templates_from_data: bool,
    /// Universal templates learned. EMUsort: 9 (Kilosort4: 6).
    pub n_templates: usize,
    /// Temporal PCs learned. EMUsort: 9 (Kilosort4: 6).
    pub n_pcs: usize,
    /// Batch stride of fitting and learning. EMUsort: 2 (Kilosort4: 25).
    pub nskip: usize,
    /// Template positions; see [`Kilosort4Config::centres`].
    pub centres: CentreOptions,
    /// Common average reference. EMUsort: off (signals span most channels of a muscle array).
    pub do_car: bool,
    /// Butterworth band-pass; see [`Kilosort4Config::do_bandpass`]. EMUsort: on.
    pub do_bandpass: bool,
    /// Band edges (Hz). EMUsort: the EMG band [`NeuralBand::Emg`], 300–5000 Hz.
    pub bandpass_low_hz: f64,
    pub bandpass_high_hz: Option<f64>,
    /// Line-noise notch. EMUsort: off (the paper uses 60 Hz; the 300 Hz edge already attenuates it
    /// by ~84 dB, and the notch makes every run ~45% slower).
    pub do_notch: bool,
    pub notch_hz: f64,
    pub notch_q: f64,
    /// Channels per whitening neighbourhood.
    pub whitening_range: usize,
    /// Samples per batch.
    pub batch_size: usize,
    /// Clustering; see [`Kilosort4Config::clustering`].
    pub clustering: ClusteringOptions,
    /// Learned-template merging; see [`Kilosort4Config::template_merge`].
    pub template_merge: TemplateMergeOptions,
    /// Matching pursuit rounds per window.
    pub max_peels: usize,
    /// Global merges; see [`Kilosort4Config::global_merge`].
    pub global_merge: GlobalMergeOptions,
    /// Duplicate-spike window (ms); see [`Kilosort4Config::duplicate_spike_ms`].
    pub duplicate_spike_ms: f64,
    /// Pin the result-changing tuned choices; see [`Kilosort4Config::reproducible`].
    pub reproducible: bool,
    /// Estimate and remove per-channel delays (`remove_chan_delays`).
    pub remove_channel_delays: bool,
    /// HDBSCAN outlier removal before k-means of the universal templates (`remove_spike_outliers`).
    pub remove_spike_outliers: bool,
    pub hdbscan_min_cluster_size: usize,
}

impl Default for EmusortConfig {
    fn default() -> Self {
        let ks = Kilosort4Config::default();
        Self {
            nt: ks.nt,
            nt_ms: ks.nt_ms,
            nt0min: ks.nt0min,
            // Measured on the HD-EMG data with the EMG band: upstream's thresholds
            th_universal: crate::sorters::kilosort4::UPSTREAM_TH_UNIVERSAL,
            th_learned: crate::sorters::kilosort4::UPSTREAM_TH_LEARNED,
            th_single_ch: vec![6.0, 9.0, 12.0, 15.0],
            templates_from_data: true,
            n_templates: 9,
            n_pcs: 9,
            nskip: 2,
            centres: ks.centres,
            do_car: false,
            // One band-pass, the EMG band (not upstream's 250–5000 Hz band-pass followed by a
            // second 300 Hz high-pass); no notch (task E1)
            do_bandpass: true,
            bandpass_low_hz: NeuralBand::Emg.low_hz(),
            bandpass_high_hz: Some(NeuralBand::Emg.high_hz()),
            do_notch: false,
            notch_hz: ks.notch_hz,
            notch_q: ks.notch_q,
            whitening_range: ks.whitening_range,
            batch_size: ks.batch_size,
            clustering: ks.clustering,
            template_merge: ks.template_merge,
            max_peels: ks.max_peels,
            global_merge: ks.global_merge,
            duplicate_spike_ms: ks.duplicate_spike_ms,
            reproducible: ks.reproducible,
            remove_channel_delays: true,
            remove_spike_outliers: true,
            hdbscan_min_cluster_size: HDBSCAN_MIN_CLUSTER_SIZE,
        }
    }
}

impl EmusortConfig {
    /// The runner's settings: every shared field, as given here (written out field by field, so a
    /// setting added to [`Kilosort4Config`] fails to compile until EMUsort decides its value).
    pub fn kilosort4(&self) -> Kilosort4Config {
        Kilosort4Config {
            nt: self.nt,
            nt_ms: self.nt_ms,
            nt0min: self.nt0min,
            th_universal: self.th_universal,
            th_learned: self.th_learned,
            th_single_ch: self.th_single_ch.clone(),
            templates_from_data: self.templates_from_data,
            n_templates: self.n_templates,
            n_pcs: self.n_pcs,
            nskip: self.nskip,
            centres: self.centres,
            do_car: self.do_car,
            do_bandpass: self.do_bandpass,
            bandpass_low_hz: self.bandpass_low_hz,
            bandpass_high_hz: self.bandpass_high_hz,
            do_notch: self.do_notch,
            notch_hz: self.notch_hz,
            notch_q: self.notch_q,
            whitening_range: self.whitening_range,
            batch_size: self.batch_size,
            clustering: self.clustering,
            template_merge: self.template_merge,
            max_peels: self.max_peels,
            global_merge: self.global_merge,
            duplicate_spike_ms: self.duplicate_spike_ms,
            reproducible: self.reproducible,
        }
    }

    /// EMUsort settings from the shared settings `ks` and EMUsort's own, all given explicitly
    /// (for bindings that hold the shared settings in one place).
    pub fn from_kilosort4(ks: Kilosort4Config, remove_channel_delays: bool, remove_spike_outliers: bool, hdbscan_min_cluster_size: usize) -> Self {
        let Kilosort4Config {
            nt,
            nt_ms,
            nt0min,
            th_universal,
            th_learned,
            th_single_ch,
            templates_from_data,
            n_templates,
            n_pcs,
            nskip,
            centres,
            do_car,
            do_bandpass,
            bandpass_low_hz,
            bandpass_high_hz,
            do_notch,
            notch_hz,
            notch_q,
            whitening_range,
            batch_size,
            clustering,
            template_merge,
            max_peels,
            global_merge,
            duplicate_spike_ms,
            reproducible,
        } = ks;
        Self {
            nt,
            nt_ms,
            nt0min,
            th_universal,
            th_learned,
            th_single_ch,
            templates_from_data,
            n_templates,
            n_pcs,
            nskip,
            centres,
            do_car,
            do_bandpass,
            bandpass_low_hz,
            bandpass_high_hz,
            do_notch,
            notch_hz,
            notch_q,
            whitening_range,
            batch_size,
            clustering,
            template_merge,
            max_peels,
            global_merge,
            duplicate_spike_ms,
            reproducible,
            remove_channel_delays,
            remove_spike_outliers,
            hdbscan_min_cluster_size,
        }
    }

    pub fn clip_options(&self) -> ClipOptions {
        self.kilosort4().clip_options()
    }

    pub fn learn_options(&self) -> LearnOptions {
        LearnOptions {
            // EMUsort's fork scales clips by one common factor
            clip_scaling: ClipScaling::Common,
            outlier_min_cluster_size: self.remove_spike_outliers.then_some(self.hdbscan_min_cluster_size),
            ..self.kilosort4().learn_options()
        }
    }

    /// Largest delay (samples) at `sample_rate_hz`.
    pub fn max_delay_samples(&self, sample_rate_hz: f64) -> usize {
        (sample_rate_hz / MAX_DELAY_DIVISOR).floor() as usize
    }
}

/// Paper and code of EMUsort.
pub fn emusort_provenance() -> Provenance {
    Provenance {
        name: "EMUsort".into(),
        kind: ProvenanceKind::ReimplementedFromPaper,
        paper: Some(Paper {
            title: "High performance sorting of motor unit action potentials with EMUsort".into(),
            authors: ["O’Connell", "Michaels", "Wang", "Mamidipaka", "Venkatesh", "Aresh", "Pachitariu", "Pruszynski", "Sober", "Pandarinath"]
                .map(String::from)
                .to_vec(),
            venue: "eLife 15:RP110417 (reviewed preprint)".into(),
            year: 2026,
            doi: "10.7554/eLife.110417.1".into(),
            license: Some("CC-BY-4.0".into()),
        }),
        code: UpstreamCode { url: "https://github.com/snel-repo/EMUsort".into(), license: Some("GPL-3.0".into()), version: "a06bb60 (Kilosort4 fork snel-repo/Kilosort4, base v4.0.18)".into() },
        artifacts: Vec::new(),
        notes: "Preprint: bioRxiv, doi:10.64898/2026.01.06.697952. Written from the paper and the published defaults; GPL code not ported. Universal templates are always learned from the recording (no EMUsort artifacts exist). Implemented: channel-delay removal, HDBSCAN outlier removal, and the Kilosort4 stages of crate::sorters::kilosort4.".into(),
    }
}

pub mod delays;
pub mod kernels;

pub use delays::{delays_from_cross_correlation, ChannelAligner, ChannelDelayEstimator, DELAY_TILE_SAMPLES};

impl RunPlan {
    /// EMUsort's run at `sample_rate_hz`: Kilosort4's stages with `config`'s settings, channel
    /// delays and HDBSCAN outlier removal.
    pub fn emusort(config: &EmusortConfig, sample_rate_hz: f64) -> Self {
        Self {
            sorter: EMUSORT_SORTER,
            config: config.kilosort4(),
            learn: config.learn_options(),
            max_channel_delay: config.remove_channel_delays.then(|| config.max_delay_samples(sample_rate_hz)),
            templates: None,
            fitted: None,
        }
    }
}

/// EMUsort over a whole recording ([`run_plan`] with [`RunPlan::emusort`]).
#[derive(Debug, Clone, Default)]
pub struct Emusort {
    pub config: EmusortConfig,
}

impl Emusort {
    pub fn new(config: EmusortConfig) -> Self {
        Self { config }
    }

    /// Runs EMUsort over `source` on `client`'s device runtime.
    pub fn run(&self, client: &Client, source: &dyn RecordingSource, probe: &SensorLayout) -> DspResult<Kilosort4Result> {
        self.run_with_progress(client, source, probe, &dsp_core::NoProgress)
    }

    /// [`Self::run`], reporting each stage to `progress` (Kilosort4's runner stages).
    pub fn run_with_progress(
        &self,
        client: &Client,
        source: &dyn RecordingSource,
        probe: &SensorLayout,
        progress: &dyn dsp_core::ProgressSink,
    ) -> DspResult<Kilosort4Result> {
        run_plan(client, source, probe, &RunPlan::emusort(&self.config, source.info().sample_rate_hz()), progress)
    }
}

impl Attributed for Emusort {
    fn provenance(&self) -> Provenance {
        emusort_provenance()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_follow_the_paper() {
        let c = EmusortConfig::default();
        assert_eq!((c.n_pcs, c.n_templates, c.nskip), (9, 9, 2));
        assert_eq!(c.th_single_ch, vec![6.0, 9.0, 12.0, 15.0]);
        assert!(!c.do_car && !c.do_notch && c.remove_channel_delays && c.remove_spike_outliers);
        // The runner's settings carry EMUsort's values, and round-trip
        let ks = c.kilosort4();
        assert_eq!((ks.n_pcs, ks.do_car, ks.bandpass_high_hz), (9, false, Some(5000.0)));
        assert_eq!(EmusortConfig::from_kilosort4(ks, c.remove_channel_delays, c.remove_spike_outliers, c.hdbscan_min_cluster_size), c);
        assert_eq!(c.learn_options().outlier_min_cluster_size, Some(20));
    }

    #[test]
    fn emusort_runs_on_synthetic_recording() {
        use dsp_core::compute::{ComputeTarget, ComputeTask};
        use dsp_io::neuro::synthetic::{SyntheticParams, SyntheticRecording};

        let rec = SyntheticRecording::new(SyntheticParams { channels: 4, duration_sec: 1.0, sample_rate_hz: 30_000.0, ..Default::default() })
            .expect("synthetic recording");
        let probe = SensorLayout::from_channel_arrays("4ch", &[0, 1, 2, 3], &[[0.0, 0.0], [0.0, 25.0], [0.0, 50.0], [0.0, 75.0]], &[0, 0, 0, 0])
            .expect("probe layout");
        let mut config = EmusortConfig::default();
        config.batch_size = 1000;
        config.nskip = 1;
        config.whitening_range = 4;
        config.th_single_ch = vec![4.0];

        struct Task<'a>(&'a dyn RecordingSource, &'a SensorLayout, &'a EmusortConfig);
        impl ComputeTask for Task<'_> {
            type Output = DspResult<Kilosort4Result>;
            fn run(self, client: Client) -> Self::Output {
                Emusort::new(self.2.clone()).run(&client, self.0, self.1)
            }
        }

        if let Ok(target) = ComputeTarget::from_env() {
            let res = target.run(Task(&rec, &probe, &config)).expect("compute target should run").expect("EMUsort should succeed");
            assert!(res.fitted.schedule.len() > 0);
            assert_eq!(res.fitted.channel_delays.as_ref().map(|d| d.delays.len()), Some(4));
            assert_eq!(res.to_sorting_output(Some(probe.clone())).sorter_name, EMUSORT_SORTER);
        }
    }
}
