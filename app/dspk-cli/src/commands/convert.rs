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
    /// Apply the curation saved in the input `.sorting.zarr` (labels, merges, splits) and write a
    /// Phy folder as phy's own save does: the output is a Phy folder of the same sorting (only
    /// its cluster files are replaced, originals kept as `.bak`) or a new folder
    #[arg(long)]
    curated: bool,
}

pub fn run(args: &ConvertArgs) -> anyhow::Result<()> {
    if args.curated {
        use dspk_synapse::curation::{apply, phy::export_phy, Curation, SortingData};
        let base = SortingData::load(&args.input)?;
        let log = Curation::load(&args.input)?;
        let clusters = export_phy(&args.input, &apply(&base, &log.ops), &args.output)?;
        println!("Wrote {} ({clusters} clusters, {} curation edits applied)", args.output.display(), log.ops.len());
        return Ok(());
    }
    let sorting = dspk_synapse::storage::load_sorting(&args.input)?;
    println!("Read {} ({}, {:.0} Hz)", args.input.display(), sorting.sorter_name, sorting.sample_rate_hz);
    save(&sorting, &args.output, args.to)
}
