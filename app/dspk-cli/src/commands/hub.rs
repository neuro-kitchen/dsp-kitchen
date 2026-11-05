//! `hub`: the artifact catalog of dspk-hub (sorter data files, model weights), downloaded and
//! verified into the local cache.

use anyhow::anyhow;
use clap::{Args, Subcommand};
use dspk_hub::{ArtifactStatus, Catalog, CatalogEntry, Hub};

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
    /// One entry: source, hash, local path and how to cite it
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

fn entry<'a>(catalog: &'a Catalog, id: &str) -> anyhow::Result<&'a CatalogEntry> {
    catalog.get(id).ok_or_else(|| anyhow!("unknown artifact id '{id}' (`dspk hub list` shows the catalog)"))
}

pub fn run(args: &HubArgs) -> anyhow::Result<()> {
    let catalog = Catalog::load_default()?;
    let hub = Hub::from_env_or_default();
    match &args.action {
        Action::List { family, installed } => {
            let entries: Vec<_> = catalog
                .entries()
                .iter()
                .map(|e| (e, hub.status(&e.artifact())))
                .filter(|(e, _)| family.as_ref().is_none_or(|f| e.family.eq_ignore_ascii_case(f)))
                .filter(|(_, status)| !installed || *status == ArtifactStatus::Installed)
                .collect();
            println!("Cache: {}", hub.cache().root_dir().display());
            println!("{:<28} {:<12} {:<10} {:>10}  STATUS", "ID", "FAMILY", "KIND", "SIZE");
            for (e, status) in &entries {
                println!("{:<28} {:<12} {:<10} {:>10}  {status}", e.id, e.family, e.kind, size(e.size_bytes));
            }
            if entries.is_empty() {
                println!("(no entries)");
            }
        }
        Action::Info { id } => {
            let e = entry(&catalog, id)?;
            let artifact = e.artifact();
            println!("{} — {}", e.id, e.name);
            println!("  Family / kind:  {} / {}", e.family, e.kind);
            println!("  Status:         {}", hub.status(&artifact));
            if !e.description.is_empty() {
                println!("  Description:    {}", e.description);
            }
            println!("  Source:         {}", e.url);
            let download = dspk_hub::resolve_weights_uri(&e.url).map(|(_, u)| u).unwrap_or_else(|_| e.url.clone());
            println!("  Download:       {download}");
            println!("  Size, SHA-256:  {} ({} bytes), {}", size(e.size_bytes), e.size_bytes, e.sha256);
            println!("  Local path:     {}", hub.path(&artifact).display());
            if let Some(licence) = &e.credit.file_license {
                println!("  File licence:   {licence}");
            }
            println!("  Cite:           {}", e.credit.citation());
        }
        Action::Pull { id, force } => {
            let e = entry(&catalog, id)?;
            let artifact = e.artifact();
            if !force && hub.status(&artifact) == ArtifactStatus::Installed {
                println!("{id} is installed at {}", hub.path(&artifact).display());
                return Ok(());
            }
            println!("Downloading {id} from {}", e.url);
            let path = hub.pull(&artifact, *force)?;
            println!("Verified ({}, SHA-256 {}) → {}", size(e.size_bytes), e.sha256, path.display());
        }
        Action::Verify { id, check_links } => {
            let reports = match id {
                Some(id) => vec![hub.verify(&entry(&catalog, id)?.artifact(), *check_links)?],
                None => catalog.entries().iter().map(|e| hub.verify(&e.artifact(), *check_links)).collect::<anyhow::Result<Vec<_>>>()?,
            };
            for r in &reports {
                let link = r.remote_check.as_ref().map_or(String::new(), |c| format!(", link {}", if c.reachable { "reachable" } else { c.detail.as_str() }));
                println!("{:<28} {}{link}", r.id, r.status);
            }
        }
        Action::Remove { id } => {
            let removed = hub.remove(&entry(&catalog, id)?.artifact())?;
            println!("{}", if removed { format!("Removed {id}") } else { format!("{id} was not downloaded") });
        }
        Action::Clean => println!("Freed {}", size(hub.clean()?)),
    }
    Ok(())
}
