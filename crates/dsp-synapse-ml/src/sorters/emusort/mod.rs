//! EMUsort (O'Connell et al., openRxiv 2026): Kilosort4 adapted to motor-unit action potentials
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

use crate::provenance::{Attributed, Paper, Provenance, ProvenanceKind, UpstreamCode};
use crate::sorters::kilosort4::runner::{run_plan, Kilosort4Result, RunPlan};
use crate::sorters::kilosort4::{ClipOptions, Kilosort4Config, LearnOptions};

/// Sorter name of an EMUsort run in [`dsp_synapse::core::SortingOutput`].
pub const EMUSORT_SORTER: &str = "emusort";

/// Largest channel delay searched: `fs / MAX_DELAY_DIVISOR` samples (2 ms).
pub const MAX_DELAY_DIVISOR: f64 = 500.0;
/// HDBSCAN `min_cluster_size` of outlier removal (`hdbscan_min_cluster_size`).
pub const HDBSCAN_MIN_CLUSTER_SIZE: usize = 20;

/// EMUsort settings: Kilosort4's, with the paper's changes.
#[derive(Debug, Clone, PartialEq)]
pub struct EmusortConfig {
    pub kilosort4: Kilosort4Config,
    /// Estimate and remove per-channel delays (`remove_chan_delays`).
    pub remove_channel_delays: bool,
    /// HDBSCAN outlier removal before k-means of the universal templates (`remove_spike_outliers`).
    pub remove_spike_outliers: bool,
    pub hdbscan_min_cluster_size: usize,
}

impl Default for EmusortConfig {
    fn default() -> Self {
        Self {
            kilosort4: Kilosort4Config {
                th_single_ch: vec![6.0, 9.0, 12.0, 15.0],
                n_pcs: 9,
                n_templates: 9,
                nskip: 2,
                do_car: false,
                ..Kilosort4Config::default()
            },
            remove_channel_delays: true,
            remove_spike_outliers: true,
            hdbscan_min_cluster_size: HDBSCAN_MIN_CLUSTER_SIZE,
        }
    }
}

impl EmusortConfig {
    pub fn clip_options(&self) -> ClipOptions {
        self.kilosort4.clip_options()
    }

    pub fn learn_options(&self) -> LearnOptions {
        LearnOptions {
            outlier_min_cluster_size: self.remove_spike_outliers.then_some(self.hdbscan_min_cluster_size),
            ..self.kilosort4.learn_options()
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
            venue: "openRxiv (bioRxiv)".into(),
            year: 2026,
            doi: "10.64898/2026.01.06.697952".into(),
            license: Some("CC-BY-4.0".into()),
        }),
        code: UpstreamCode { url: "https://github.com/snel-repo/EMUsort".into(), license: Some("GPL-3.0".into()), version: "a06bb60 (Kilosort4 fork snel-repo/Kilosort4, base v4.0.18)".into() },
        artifacts: Vec::new(),
        notes: "Written from the paper and the published defaults; GPL code not ported. Universal templates are always learned from the recording (no EMUsort artifacts exist). Implemented: channel-delay removal, HDBSCAN outlier removal, and the Kilosort4 stages of crate::sorters::kilosort4.".into(),
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
            config: config.kilosort4.clone(),
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
        assert_eq!((c.kilosort4.n_pcs, c.kilosort4.n_templates, c.kilosort4.nskip), (9, 9, 2));
        assert_eq!(c.kilosort4.th_single_ch, vec![6.0, 9.0, 12.0, 15.0]);
        assert!(!c.kilosort4.do_car && c.remove_channel_delays && c.remove_spike_outliers);
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
        config.kilosort4.batch_size = 1000;
        config.kilosort4.nskip = 1;
        config.kilosort4.whitening_range = 4;
        config.kilosort4.th_single_ch = vec![4.0];

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
