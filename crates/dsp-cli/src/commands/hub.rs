//! CLI client for the `dsp-synapse-ml` Pretrained Electrophysiology Model Hub (`dsp-cli hub ...`).

use anyhow::Result;
use clap::{Args, Subcommand};
use dsp_synapse_ml::hub::{ModelHub, ModelStatus};

#[derive(Args, Debug)]
pub struct HubArgs {
    #[command(subcommand)]
    pub action: HubCommand,
}

#[derive(Subcommand, Debug)]
pub enum HubCommand {
    /// List pretrained electrophysiology models in the catalog and their local cache status
    List {
        /// Filter models by family (e.g. spikenet2, unitrefine, dartsort, kilosort4, cebra, cascade, deepinterpolation)
        #[arg(short, long)]
        family: Option<String>,

        /// Show only models that are currently installed in the local cache
        #[arg(long, default_value_t = false)]
        installed: bool,
    },

    /// Display full inspection card for a model (links, SHA-256, cache path, and tensor I/O spec)
    Info {
        /// Model ID (e.g. `unitrefine/curation-v1`, `dartsort/singlechan-denoiser-v1`)
        model_id: String,
    },

    /// Download a model's pretrained weights into the local Hub cache and verify its SHA-256
    Pull {
        /// Model ID to download
        model_id: String,

        /// Re-download and overwrite even if already installed and verified
        #[arg(short, long, default_value_t = false)]
        force: bool,
    },

    /// Verify local cached weight SHA-256 checksums and optionally inspect remote URLs
    Verify {
        /// Optional specific model ID to verify (verifies all catalog models if omitted)
        model_id: Option<String>,

        /// Perform an HTTP HEAD / link reachability check against the remote `weights_url`
        #[arg(long, default_value_t = false)]
        check_links: bool,
    },

    /// Remove a specific model's cached weights from disk
    #[command(alias = "remove")]
    Rm {
        /// Model ID to remove from the local cache
        model_id: String,
    },

    /// Remove all cached model weights from the local Hub cache directory
    Clean,
}

pub fn run_hub(args: &HubArgs) -> Result<()> {
    let hub = ModelHub::new()?;

    match &args.action {
        HubCommand::List { family, installed } => {
            let mut entries = hub.list();
            if let Some(fam) = family {
                entries.retain(|e| e.manifest.family.eq_ignore_ascii_case(fam));
            }
            if *installed {
                entries.retain(|e| e.status == ModelStatus::Installed);
            }

            println!("==========================================================================================================");
            println!("                              dsp-synapse-ml Pretrained Model Hub Catalog                                 ");
            println!("==========================================================================================================");
            println!(
                "Cache Directory: {}",
                hub.cache().root_dir().display()
            );
            println!("----------------------------------------------------------------------------------------------------------");
            println!(
                "{:<38} {:<18} {:<12} {:<12} {:>10}  {:<12}",
                "MODEL ID", "FAMILY", "TASK", "FORMAT", "SIZE", "STATUS"
            );
            println!("----------------------------------------------------------------------------------------------------------");

            if entries.is_empty() {
                println!("  (no models matched the filter criteria)");
            } else {
                for entry in &entries {
                    let m = &entry.manifest;
                    println!(
                        "{:<38} {:<18} {:<12} {:<12} {:>10}  {:<12}",
                        m.id,
                        m.family,
                        m.task,
                        m.format.extension(),
                        format_size(m.size_bytes),
                        entry.status.badge()
                    );
                }
            }
            println!("==========================================================================================================");
        }

        HubCommand::Info { model_id } => {
            let entry = hub.info(model_id)?;
            let m = &entry.manifest;

            println!("================================================================================");
            println!(" Model Inspection Card: {}", m.id);
            println!("================================================================================");
            println!("  Name:             {}", m.name);
            println!("  Family:           {}", m.family);
            println!("  Task:             {}", m.task);
            println!("  Version:          {}", m.version);
            println!("  Format:           {}", m.format);
            println!("  Status:           {}", entry.status.badge());
            if !m.description.is_empty() {
                println!("  Description:      {}", m.description);
            }
            println!("\n[Provenance & Links]");
            println!("  Author Repo:      {}", m.upstream_repo);
            println!("  Paper / DOI:      {}", m.paper_url);
            println!("  License:          {}", m.license);
            println!("  Manifest URI:     {}", m.weights_url);
            println!("  Direct Download:  {}", entry.resolved_download_url);
            println!("\n[Artifact & Cache]");
            println!("  Expected Size:    {} ({} bytes)", format_size(m.size_bytes), m.size_bytes);
            println!("  SHA-256 Digest:   {}", m.sha256);
            println!("  Local Cache Path: {}", entry.local_weights_path.display());
            println!("\n[Tensor I/O Specification]");
            println!("  Sample Rate:      {:.1} Hz", m.io_spec.sample_rate_hz);
            println!("  Normalization:    {}", m.io_spec.normalization);
            println!("  Inputs:");
            for inp in &m.io_spec.inputs {
                let desc = inp.description.as_deref().unwrap_or("");
                println!(
                    "    - {:<24} {:<10} {:<16} {}",
                    inp.name,
                    inp.dtype,
                    inp.format_shape(),
                    desc
                );
            }
            println!("  Outputs:");
            for out in &m.io_spec.outputs {
                let desc = out.description.as_deref().unwrap_or("");
                println!(
                    "    - {:<24} {:<10} {:<16} {}",
                    out.name,
                    out.dtype,
                    out.format_shape(),
                    desc
                );
            }
            println!("================================================================================");
        }

        HubCommand::Pull { model_id, force } => {
            let before = hub.info(model_id)?;
            if !force && before.status == ModelStatus::Installed {
                println!(
                    "[INSTALLED] '{}' is already cached and verified at {}",
                    before.manifest.id,
                    before.local_weights_path.display()
                );
                return Ok(());
            }

            println!(
                "Pulling '{}' from {} ...",
                before.manifest.id, before.resolved_download_url
            );
            let after = hub.pull(model_id, *force)?;
            println!(
                "[INSTALLED] Verified SHA-256 ({}) -> {}",
                after.manifest.sha256,
                after.local_weights_path.display()
            );
        }

        HubCommand::Verify {
            model_id,
            check_links,
        } => {
            let reports = match model_id {
                Some(id) => vec![hub.verify_with_options(id, *check_links)?],
                None => hub.verify_all(*check_links)?,
            };

            println!("==========================================================================================");
            println!("                               Model Hub Verification Report                              ");
            println!("==========================================================================================");
            for rep in &reports {
                let sha_note = match &rep.actual_sha256 {
                    Some(actual) => format!("sha256 OK ({}..)", &actual[..12.min(actual.len())]),
                    None => "not cached locally".to_string(),
                };
                print!(
                    "  {:<13} {:<38} {}",
                    rep.status.badge(),
                    rep.id,
                    sha_note
                );
                if let Some(remote) = &rep.remote_check {
                    let link_badge = if remote.reachable {
                        "LINK_OK"
                    } else {
                        "LINK_UNREACHABLE"
                    };
                    print!(" | {} ({})", link_badge, remote.detail);
                }
                println!();
            }
            println!("==========================================================================================");
        }

        HubCommand::Rm { model_id } => {
            let removed = hub.remove(model_id)?;
            if removed {
                println!("Removed cached weights for '{model_id}'.");
            } else {
                println!("Model '{model_id}' was not cached locally.");
            }
        }

        HubCommand::Clean => {
            let freed = hub.clean()?;
            println!(
                "Cleaned local Model Hub cache at {} (freed {}).",
                hub.cache().root_dir().display(),
                format_size(freed)
            );
        }
    }

    Ok(())
}

fn format_size(bytes: u64) -> String {
    const KB: f64 = 1024.0;
    const MB: f64 = 1024.0 * 1024.0;
    const GB: f64 = 1024.0 * 1024.0 * 1024.0;
    let b = bytes as f64;
    if b >= GB {
        format!("{:.2} GB", b / GB)
    } else if b >= MB {
        format!("{:.2} MB", b / MB)
    } else if b >= KB {
        format!("{:.1} KB", b / KB)
    } else {
        format!("{bytes} B")
    }
}
