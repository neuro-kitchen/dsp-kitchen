//! `hub`: published artifacts of the dsp-synapse-ml catalog (sorter arrays, model weights),
//! downloaded and verified through dsp-synapse-hub.

use clap::{Args, Subcommand};
use dsp_synapse_hub::ArtifactStatus;
use dsp_synapse_ml::ModelHub;

const BYTES_PER_KIB: f64 = 1024.0;
const SIZE_UNITS: [&str; 4] = ["B", "KiB", "MiB", "GiB"];

#[derive(Subcommand, Debug)]
enum Action {
    /// Catalog entries and whether they are downloaded
    List {
        /// Only this family (e.g. kilosort4)
        #[arg(long)]
        family: Option<String>,
        /// Only downloaded entries
        #[arg(long)]
        installed: bool,
    },
    /// One entry: source, hash, contents, provenance and how to cite it
    Info { id: String },
    /// Download and verify (size and SHA-256)
    Pull {
        id: String,
        /// Download again even if installed
        #[arg(long)]
        force: bool,
    },
    /// Re-hash downloaded files (all, or one)
    Verify {
        id: Option<String>,
        /// Also check that the download link answers
        #[arg(long)]
        check_links: bool,
    },
    /// Delete one downloaded entry
    Remove { id: String },
    /// Delete every downloaded entry
    Clean,
}

#[derive(Args, Debug)]
pub struct HubArgs {
    #[command(subcommand)]
    action: Action,
}

fn size(bytes: u64) -> String {
    let mut value = bytes as f64;
    let mut unit = 0;
    while value >= BYTES_PER_KIB && unit + 1 < SIZE_UNITS.len() {
        value /= BYTES_PER_KIB;
        unit += 1;
    }
    if unit == 0 { format!("{bytes} {}", SIZE_UNITS[0]) } else { format!("{value:.1} {}", SIZE_UNITS[unit]) }
}

pub fn run(args: &HubArgs) -> anyhow::Result<()> {
    let hub = ModelHub::new()?;
    match &args.action {
        Action::List { family, installed } => {
            let entries: Vec<_> = hub
                .list()
                .into_iter()
                .filter(|e| family.as_ref().is_none_or(|f| e.manifest.family.eq_ignore_ascii_case(f)))
                .filter(|e| !installed || e.status == ArtifactStatus::Installed)
                .collect();
            println!("Cache: {}", hub.hub().cache().root_dir().display());
            println!("{:<28} {:<12} {:<8} {:>10}  STATUS", "ID", "FAMILY", "FORMAT", "SIZE");
            for e in &entries {
                let m = &e.manifest;
                println!("{:<28} {:<12} {:<8} {:>10}  {}", m.id, m.family, m.format.to_string(), size(m.size_bytes), e.status);
            }
            if entries.is_empty() {
                println!("(no entries)");
            }
        }
        Action::Info { id } => {
            let e = hub.info(id)?;
            let m = &e.manifest;
            println!("{} — {}", m.id, m.name);
            println!("  Family / task:  {} / {}", m.family, m.task);
            println!("  Format:         {}", m.format);
            println!("  Status:         {}", e.status);
            if !m.description.is_empty() {
                println!("  Description:    {}", m.description);
            }
            println!("  Source:         {}", m.weights_url);
            println!("  Download:       {}", e.resolved_download_url);
            println!("  Size, SHA-256:  {} ({} bytes), {}", size(m.size_bytes), m.size_bytes, m.sha256);
            println!("  Local path:     {}", e.local_path.display());
            for a in &m.arrays {
                println!("  Array:          {} {} {:?}", a.name, a.dtype, a.shape);
            }
            println!("  Cite:           {}", m.provenance.citation());
        }
        Action::Pull { id, force } => {
            let before = hub.info(id)?;
            if !force && before.status == ArtifactStatus::Installed {
                println!("{id} is installed at {}", before.local_path.display());
                return Ok(());
            }
            println!("Downloading {id} from {}", before.resolved_download_url);
            let (manifest, path) = hub.pull(id, *force)?;
            println!("Verified ({}, SHA-256 {}) → {}", size(manifest.size_bytes), manifest.sha256, path.display());
        }
        Action::Verify { id, check_links } => {
            let reports = match id {
                Some(id) => vec![hub.verify(id, *check_links)?],
                None => hub.verify_all(*check_links)?,
            };
            for r in &reports {
                let link = r.remote_check.as_ref().map_or(String::new(), |c| format!(", link {}", if c.reachable { "reachable" } else { c.detail.as_str() }));
                println!("{:<28} {}{link}", r.id, r.status);
            }
        }
        Action::Remove { id } => println!("{}", if hub.remove(id)? { format!("Removed {id}") } else { format!("{id} was not downloaded") }),
        Action::Clean => println!("Freed {}", size(hub.clean()?)),
    }
    Ok(())
}
