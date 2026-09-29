pub mod device;
pub mod tensor_adapter;

pub use device::{SynapseMlDevice, Tensor, Tensor1D, Tensor2D, Tensor3D};
pub use tensor_adapter::{
    SnippetBatchMetadata, snippet_batch_into_tensor, snippet_batch_to_tensor,
    tensor_into_snippet_batch, tensor_to_snippet_batch,
};
