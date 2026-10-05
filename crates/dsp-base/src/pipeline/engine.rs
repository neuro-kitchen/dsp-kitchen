use cubecl::prelude::*;
use super::session::PipelineWorkspace;
use crate::core::DspFloat;
use super::stage::PipelineStage;
use crate::filter::design::FilterError;

/// In-VRAM DSP Pipeline Engine.
/// Chains an arbitrary sequence of filter and math stages directly in device memory
/// using double-buffered (ping-pong) VRAM allocation, guaranteeing zero host PCIe round-trips.
#[derive(Debug, Clone, Default)]
pub struct Pipeline {
    stages: Vec<PipelineStage>,
}

impl Pipeline {
    pub fn new() -> Self {
        Self { stages: Vec::new() }
    }

    pub fn with_stages(stages: Vec<PipelineStage>) -> Self {
        Self { stages }
    }

    pub fn add(&mut self, stage: PipelineStage) -> &mut Self {
        self.stages.push(stage);
        self
    }

    pub fn stages(&self) -> &[PipelineStage] {
        &self.stages
    }

    pub fn len(&self) -> usize {
        self.stages.len()
    }

    pub fn is_empty(&self) -> bool {
        self.stages.is_empty()
    }

    /// `(left, right)` context in samples a chunk needs at `sample_rate` Hz so that its interior
    /// equals whole-recording processing. Transients of cascaded stages add, so both sides are
    /// summed over stages.
    pub fn settling(&self, sample_rate: f64) -> Result<(usize, usize), FilterError> {
        self.stages.iter().try_fold((0, 0), |(l, r), stage| {
            let (sl, sr) = stage.settling(sample_rate)?;
            Ok((l + sl, r + sr))
        })
    }

    /// Checks every stage's parameters at `sample_rate` Hz (filter designs, cutoffs, orders).
    pub fn validate(&self, sample_rate: f64) -> Result<(), FilterError> {
        self.settling(sample_rate).map(|_| ())
    }

    /// Executes all stages on a `[channels, samples]` device buffer of `F` as one independent chunk
    /// and returns a handle to the result.
    ///
    /// One-off: every call designs and uploads the filters and allocates the buffers again. For more
    /// than one chunk, create a [`PipelineWorkspace`] once and call
    /// [`PipelineWorkspace::process_handle`]; it keeps designs, weights and buffers on its device.
    pub fn execute<R: Runtime, F: DspFloat>(
        &self,
        client: &ComputeClient<R>,
        input_handle: &cubecl::server::Handle,
        channels: usize,
        samples: usize,
        sample_rate: f64,
    ) -> Result<cubecl::server::Handle, FilterError> {
        if self.stages.is_empty() {
            return Ok(input_handle.clone());
        }
        let mut workspace = PipelineWorkspace::<R, F>::new(
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

    fn chained<R: Runtime>(client: &ComputeClient<R>) {
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

        let out_handle = pipeline.execute::<R, f32>(client, &in_handle, channels, samples, 30000.0).unwrap();
        let out = buffer::download::<R, f32>(client, out_handle);

        assert_eq!(out.len(), total);
        // A constant input is removed by CAR (every channel equals the average)
        assert!(out.iter().all(|v| v.abs() < 1e-3), "{}", R::name(client));
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
