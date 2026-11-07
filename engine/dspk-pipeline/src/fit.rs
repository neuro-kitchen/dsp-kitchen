//! Stages fitted to the data before a run: spatial whitening learns the covariance of the signal
//! the stages before it produce, over chunks spread across the recording. The pipeline executor
//! and the app's live preview fit with this one function, so a preview equals the written output.

use dspk_core::{ComputeTarget, RecordingSource};
use dspk_ops::pipeline::{PipelineStage, PipelineWorkspace};
use dspk_ops::spatial::SpatialWhitening;
use dspk_ops::Pipeline;

use crate::PipelineError;

/// How a `whiten` node is fitted (its spec fields).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WhitenFit {
    /// Regularisation added to the covariance's eigenvalues (in the signal's unit squared).
    pub epsilon: f32,
    /// Seconds of signal the covariance is estimated from, in total.
    pub fit_s: f64,
    /// Chunks those seconds are split into, spread evenly over the recording.
    pub chunks: usize,
}

/// The identity on `channels` channels: a whitening's place in a pipeline before it is fitted
/// (same channels, no settling), so a run can be planned without the device.
pub fn unfitted_whitening(channels: usize) -> SpatialWhitening {
    let mut m = vec![0.0f32; channels * channels];
    for c in 0..channels {
        m[c * channels + c] = 1.0;
    }
    SpatialWhitening::from_matrix(channels, m)
}

/// Fits a global ZCA whitening of `channels` of `source` as processed by `before` (the stages
/// ahead of it): `fit.chunks` chunks of `fit.fit_s / fit.chunks` seconds spread evenly over the
/// recording, each processed with the stages' settling context, their covariance whitened.
///
/// # Errors
///
/// When reading `source` fails, the stages are invalid, or the device is not available.
pub fn fit_whitening(target: ComputeTarget, source: &dyn RecordingSource, channels: &[usize], before: &[PipelineStage], fit: &WhitenFit) -> Result<SpatialWhitening, PipelineError> {
    let info = source.info();
    let (total, fs, nch) = (info.samples, info.sample_rate_hz(), channels.len());
    if nch == 0 || total == 0 {
        return Err(PipelineError::Invalid("whiten: no channels or samples to fit on".into()));
    }
    let chunks = fit.chunks.max(1) as u64;
    let len = (((fit.fit_s * fs) as u64) / chunks).clamp(1, total / chunks.max(1)).max(1);
    let pipeline = Pipeline::with_stages(before.to_vec());
    let (left, right) = pipeline.settling(fs).map_err(|e| PipelineError::Invalid(e.to_string()))?;
    let client = target.client().map_err(|e| PipelineError::Compute(e.to_string()))?;
    let mut workspace = if before.is_empty() {
        None
    } else {
        let capacity = (len as usize + left + right).min(total as usize);
        Some(PipelineWorkspace::<f32>::new(client.clone(), pipeline, nch, capacity, fs).map_err(|e| PipelineError::Invalid(e.to_string()))?)
    };
    let mut rows: Vec<Vec<f32>> = vec![Vec::with_capacity((len * chunks) as usize); nch];
    for k in 0..chunks {
        // Chunk k centred at (k + ½) / chunks of the recording
        let centre = ((2 * k + 1) * total) / (2 * chunks);
        let start = centre.saturating_sub(len / 2).min(total - len);
        let read = start.saturating_sub(left as u64)..(start + len + right as u64).min(total);
        let n = (read.end - read.start) as usize;
        let mut values = vec![0.0f32; nch * n];
        source.read(channels, read.clone(), &mut values).map_err(|e| PipelineError::Io(e.to_string()))?;
        let processed = match workspace.as_mut() {
            Some(ws) => {
                let mut out = vec![0.0f32; nch * n];
                ws.process_chunk(&values, n, &mut out);
                out
            }
            None => values,
        };
        let a = (start - read.start) as usize;
        for (c, row) in rows.iter_mut().enumerate() {
            row.extend_from_slice(&processed[c * n + a..c * n + a + len as usize]);
        }
    }
    let samples = rows[0].len();
    let data: Vec<f32> = rows.into_iter().flatten().collect();
    Ok(SpatialWhitening::fit_zca::<f32>(&client, &data, nch, samples, fit.epsilon))
}
