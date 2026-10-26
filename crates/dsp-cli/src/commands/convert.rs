//! `convert`: a sorting from one format to another (Phy folder, `.sorting.zarr`, NWB units).

use std::path::PathBuf;

use clap::Args;

use super::recording::{save, Format};

#[derive(Args, Debug)]
pub struct ConvertArgs {
    /// Sorting to read: Phy / Kilosort folder, `.sorting.zarr` or `.nwb.zarr` (detected)
    input: PathBuf,
    /// Where to write it
    output: PathBuf,
    /// Output format; default: from the output name (`.nwb.zarr` → nwb, other `.zarr` → zarr, else phy)
    #[arg(long, value_enum)]
    to: Option<Format>,
}

pub fn run(args: &ConvertArgs) -> anyhow::Result<()> {
    let sorting = dsp_synapse::storage::load_sorting(&args.input)?;
    println!("Read {} ({}, {:.0} Hz)", args.input.display(), sorting.sorter_name, sorting.sample_rate_hz);
    save(&sorting, &args.output, args.to)
}
