use cubecl::prelude::*;
use super::session::PipelineWorkspace;
use crate::core::DspFloat;
use super::stage::PipelineStage;
use crate::filter::design::FilterError;

/// Stages (filters, spatial operators, pointwise math) run one after the other on a
/// `[channels, samples]` device buffer. Intermediate results stay on the device, in two buffers
/// used in turn; nothing is read back between stages.
///
/// [`execute`](Self::execute) processes one buffer as an independent chunk. For a recording read in
/// windows, build a [`PipelineWorkspace`] once: it keeps the filter designs, weights and buffers
/// on the device, and with state carried across calls the windows join without seams.
///
/// # Examples
///
/// ```
/// use dsp_base::core::buffer;
/// use dsp_base::{Pipeline, PipelineStage};
/// use dsp_core::compute::ComputeTarget;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let client = ComputeTarget::from_env()?.client()?;
/// let (channels, samples, fs) = (4, 3_000, 30_000.0);
/// // A 1 kHz sine on every channel plus a 50 µV offset shared by all of them
/// let x: Vec<f32> = (0..channels * samples)
///     .map(|i| 50.0 + (2.0 * std::f32::consts::PI * 1_000.0 * (i % samples) as f32 / fs as f32).sin())
///     .collect();
///
/// let mut pipeline = Pipeline::new();
/// pipeline
///     .add(PipelineStage::CommonAverageReference) // removes what all channels share
///     .add(PipelineStage::bandpass(300.0, 5_000.0));
/// pipeline.validate(fs)?;
///
/// let input = buffer::upload(&client, &x);
/// let output = pipeline.execute::<f32>(&client, &input, channels, samples, fs)?;
/// let y: Vec<f32> = buffer::download(&client, output);
/// assert_eq!(y.len(), channels * samples);
/// // Every channel carried the same signal, so the reference removed it all
/// assert!(y.iter().all(|v| v.abs() < 1e-3));
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone, Default)]
pub struct Pipeline {
    stages: Vec<PipelineStage>,
}

impl Pipeline {
    /// An empty pipeline (it returns its input unchanged).
    pub fn new() -> Self {
        Self { stages: Vec::new() }
    }

    /// A pipeline of `stages`, run in order.
    pub fn with_stages(stages: Vec<PipelineStage>) -> Self {
        Self { stages }
    }

    /// Appends `stage`; returns the pipeline so calls chain.
    pub fn add(&mut self, stage: PipelineStage) -> &mut Self {
        self.stages.push(stage);
        self
    }

    /// The stages, in the order they run.
    pub fn stages(&self) -> &[PipelineStage] {
        &self.stages
    }

    /// Number of stages.
    pub fn len(&self) -> usize {
        self.stages.len()
    }

    /// Whether there are no stages.
    pub fn is_empty(&self) -> bool {
        self.stages.is_empty()
    }

    /// `(left, right)` context in samples a chunk needs at `sample_rate` Hz so that its interior
    /// equals whole-recording processing. Transients of cascaded stages add, so both sides are
    /// summed over stages.
    ///
    /// # Errors
    ///
    /// [`FilterError`] when a stage's design is invalid at `sample_rate` (see [`validate`](Self::validate)).
    pub fn settling(&self, sample_rate: f64) -> Result<(usize, usize), FilterError> {
        self.stages.iter().try_fold((0, 0), |(l, r), stage| {
            let (sl, sr) = stage.settling(sample_rate)?;
            Ok((l + sl, r + sr))
        })
    }

    /// Checks every stage's parameters at `sample_rate` Hz (filter designs, cutoffs, orders).
    ///
    /// # Errors
    ///
    /// [`FilterError`] for the first invalid stage, e.g. a cutoff at or above Nyquist.
    ///
    /// ```
    /// use dsp_base::{Pipeline, PipelineStage};
    ///
    /// let pipeline = Pipeline::with_stages(vec![PipelineStage::bandpass(300.0, 5_000.0)]);
    /// assert!(pipeline.validate(30_000.0).is_ok());
    /// assert!(pipeline.validate(8_000.0).is_err()); // 5 kHz is above the 4 kHz Nyquist
    /// ```
    pub fn validate(&self, sample_rate: f64) -> Result<(), FilterError> {
        self.settling(sample_rate).map(|_| ())
    }

    /// Executes all stages on a `[channels, samples]` device buffer of `F` as one independent chunk
    /// and returns a handle to the result.
    ///
    /// One-off: every call designs and uploads the filters and allocates the buffers again. For more
    /// than one chunk, create a [`PipelineWorkspace`] once and call
    /// [`PipelineWorkspace::process_handle`]; it keeps designs, weights and buffers on its device.
    ///
    /// # Arguments
    ///
    /// * `client` – device the input lives on.
    /// * `input_handle` – `channels × samples` values of `F`, channel-major.
    /// * `sample_rate` – Hz; the filters are designed for it.
    ///
    /// # Errors
    ///
    /// [`FilterError`] when a stage's design is invalid at `sample_rate`.
    pub fn execute<F: DspFloat>(
        &self,
        client: &Client,
        input_handle: &cubecl::server::Handle,
        channels: usize,
        samples: usize,
        sample_rate: f64,
    ) -> Result<cubecl::server::Handle, FilterError> {
        if self.stages.is_empty() {
            return Ok(input_handle.clone());
        }
        let mut workspace = PipelineWorkspace::<F>::new(
            client.clone(),
            self.clone(),
            channels,
            samples,
            sample_rate,
        )?;
        Ok(workspace.process_handle(input_handle, samples))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::buffer;

    fn chained(client: &Client) {
        let channels = 32;
        let samples = 500;
        let total = channels * samples;

        let in_handle = buffer::upload(client, &vec![100.0f32; total]);

        // Build a 3-stage pipeline: Scale -> CAR -> Notch
        let mut pipeline = Pipeline::new();
        pipeline
            .add(PipelineStage::Scale { alpha: 0.195, beta: 0.0 })
            .add(PipelineStage::CommonAverageReference)
            .add(PipelineStage::notch(60.0, 30.0));

        assert_eq!(pipeline.len(), 3);

        let out_handle = pipeline.execute::<f32>(client, &in_handle, channels, samples, 30000.0).unwrap();
        let out = buffer::download::<f32>(client, out_handle);

        assert_eq!(out.len(), total);
        // A constant input is removed by CAR (every channel equals the average)
        assert!(out.iter().all(|v| v.abs() < 1e-3), "{}", client.name());
    }
    runtime_test!(test_pipeline_chained, chained);

    #[test]
    fn test_settling_sums_both_sides() {
        let fs = 30_000.0;
        let mut pipeline = Pipeline::new();
        pipeline.add(PipelineStage::bandpass(300.0, 6000.0)).add(PipelineStage::median9());
        let (l, r) = pipeline.settling(fs).unwrap();
        let (bl, br) = PipelineStage::bandpass(300.0, 6000.0).settling(fs).unwrap();
        assert_eq!((l, r), (bl + 4, br + 4));
        assert_eq!(bl, br);
    }

    #[test]
    fn test_invalid_cutoff_is_an_error() {
        let pipeline = Pipeline::with_stages(vec![PipelineStage::bandpass(6000.0, 300.0)]);
        assert!(pipeline.validate(30_000.0).is_err());
        let pipeline = Pipeline::with_stages(vec![PipelineStage::lowpass(20_000.0)]);
        assert!(pipeline.validate(30_000.0).is_err());
    }
}
