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
    /// The recording the sorting is of: an NWB output gets its session and subject (as the app
    /// keeps them in the recording's .dspk.zarr, and as an NWB recording says them)
    #[arg(long, value_name = "RECORDING")]
    session: Option<PathBuf>,
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
    save(&sorting, &args.output, args.to)?;
    if let Some(recording) = &args.session {
        let nwb = matches!(args.to, Some(Format::Nwb)) || (args.to.is_none() && args.output.to_string_lossy().trim_end_matches('/').ends_with(".nwb.zarr"));
        anyhow::ensure!(nwb, "--session needs an NWB output (.nwb.zarr or --to nwb)");
        use dspk_session::derived::session_meta::{from_nwb_zarr, read, write_nwb_zarr};
        let meta = from_nwb_zarr(recording).over(&read(recording));
        let n = write_nwb_zarr(&args.output, &meta)?;
        let missing: Vec<&str> = meta.missing_required().iter().map(|f| f.label).collect();
        println!("Session: {n} fields written{}", if missing.is_empty() { String::new() } else { format!(" (still missing, NWB requires them: {})", missing.join(", ")) });
    }
    Ok(())
}
