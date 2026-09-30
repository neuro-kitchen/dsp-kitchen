#[cfg(feature = "nwb")]
pub mod convert;
pub mod formats;
pub mod inspect;

/// Input options shared by commands that open a recording.
#[derive(clap::Args, Debug, Default)]
pub struct OpenArgs {
    /// Container formats (TDT tank): the block to open
    #[arg(long)]
    pub block: Option<String>,
    /// Offline spike sort to apply to snippets (TDT: folder name under sort/)
    #[arg(long)]
    pub sort: Option<String>,
}

impl OpenArgs {
    pub fn options(&self) -> neuro_convert::OpenOptions {
        neuro_convert::OpenOptions { block: self.block.clone(), sort: self.sort.clone(), ..Default::default() }
    }
}
