pub mod alignment;
pub mod extractor;
pub mod kernels;

pub use crate::core::{SnippetBatch, WaveformSnippet};
pub use alignment::{
    SINC_KERNEL_RADIUS, blackman_window, interpolate_window, parabolic_subsample_offset,
    resample_sinc_1d, resample_sinc_multichannel, sinc, sinc_tap,
};
pub use extractor::{
    extract_snippet_batch_multichannel, extract_snippets_multichannel,
    extract_snippets_single_channel, extraction_margin, snippet_fits,
};
pub use kernels::{VramSnippets, execute_extract_sinc_in_vram, extract_sinc_snippets_kernel};
