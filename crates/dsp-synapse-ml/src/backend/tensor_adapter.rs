//! Zero-copy and reference tensor adapters between `dsp_synapse::SnippetBatch` and `Tensor3D`.

use dsp_synapse::SnippetBatch;
use super::device::{SynapseMlDevice, Tensor3D};

/// Metadata stripped from a `SnippetBatch` when moving its flat `Vec<f32>` into a `Tensor3D` without copying.
#[derive(Debug, Clone, PartialEq)]
pub struct SnippetBatchMetadata {
    pub num_spikes: usize,
    pub num_channels: usize,
    pub num_samples: usize,
    pub primary_channels: Vec<usize>,
    pub center_samples: Vec<u64>,
    pub subsample_offsets: Vec<f32>,
    pub channel_ids: Vec<usize>,
}

/// Moves a `SnippetBatch`'s flat `Vec<f32>` directly into a `Tensor3D` with **zero memory allocation or copying**.
pub fn snippet_batch_into_tensor(
    batch: SnippetBatch,
    device: SynapseMlDevice,
) -> (Tensor3D, SnippetBatchMetadata) {
    let shape = batch.shape();
    let meta = SnippetBatchMetadata {
        num_spikes: batch.num_spikes,
        num_channels: batch.num_channels,
        num_samples: batch.num_samples,
        primary_channels: batch.primary_channels,
        center_samples: batch.center_samples,
        subsample_offsets: batch.subsample_offsets,
        channel_ids: batch.channel_ids,
    };
    let tensor = Tensor3D::from_floats(batch.data, shape, device);
    (tensor, meta)
}

/// Reconstructs a `SnippetBatch` from a `Tensor3D` and its `SnippetBatchMetadata` with **zero memory allocation or copying**.
pub fn tensor_into_snippet_batch(tensor: Tensor3D, meta: SnippetBatchMetadata) -> SnippetBatch {
    let [n, k, t] = tensor.shape;
    assert_eq!(n, meta.num_spikes);
    assert_eq!(k, meta.num_channels);
    assert_eq!(t, meta.num_samples);
    SnippetBatch::from_raw_parts(
        tensor.into_vec(),
        meta.num_spikes,
        meta.num_channels,
        meta.num_samples,
        meta.primary_channels,
        meta.center_samples,
        meta.subsample_offsets,
        meta.channel_ids,
    )
}

/// Borrows a `SnippetBatch` and constructs a `Tensor3D` of shape `[N, K, T]`.
pub fn snippet_batch_to_tensor(batch: &SnippetBatch, device: SynapseMlDevice) -> Tensor3D {
    Tensor3D::from_floats(batch.data.clone(), batch.shape(), device)
}

/// Replaces the waveform data of a template `SnippetBatch` with the values in `tensor`.
pub fn tensor_to_snippet_batch(tensor: Tensor3D, reference: &SnippetBatch) -> SnippetBatch {
    let [n, k, t] = tensor.shape;
    assert_eq!(n, reference.num_spikes);
    assert_eq!(k, reference.num_channels);
    assert_eq!(t, reference.num_samples);
    SnippetBatch::from_raw_parts(
        tensor.into_vec(),
        reference.num_spikes,
        reference.num_channels,
        reference.num_samples,
        reference.primary_channels.clone(),
        reference.center_samples.clone(),
        reference.subsample_offsets.clone(),
        reference.channel_ids.clone(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero_copy_snippet_batch_tensor_roundtrip() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let ptr_before = data.as_ptr();
        let batch = SnippetBatch::from_raw_parts(
            data,
            1,
            2,
            3,
            vec![0],
            vec![100],
            vec![0.1],
            vec![0, 1],
        );

        let (tensor, meta) = snippet_batch_into_tensor(batch, SynapseMlDevice::Cpu);
        assert_eq!(tensor.shape, [1, 2, 3]);
        // Verify exact same heap allocation pointer (true zero-copy move!)
        assert_eq!(tensor.as_slice().as_ptr(), ptr_before);

        let batch_after = tensor_into_snippet_batch(tensor, meta);
        assert_eq!(batch_after.as_flat_slice().as_ptr(), ptr_before);
    }
}
