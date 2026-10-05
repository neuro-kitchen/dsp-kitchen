pub mod alignment;
pub mod extractor;
pub mod kernels;

pub use crate::core::{SnippetBatch, WaveformSnippet};
pub use alignment::SINC_KERNEL_RADIUS;
pub use extractor::{
    read_snippets,
    extract_snippet_batch_multichannel, extract_snippets_multichannel,
    extract_snippets_single_channel, extraction_margin, snippet_fits,
};
pub use kernels::{VramSnippets, execute_extract_sinc_in_vram, extract_snippets_kernel, trough_shift_kernel};
