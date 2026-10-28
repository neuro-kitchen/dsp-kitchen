//! `spec`: pipeline specs without running them: check one (and print its canonical JSON, every
//! default written out), or print the JSON Schema for editors.

use std::path::PathBuf;

use clap::{Args, Subcommand};
use dspk_pipeline::PipelineSpec;

use crate::style::{paint, DIM, OK, VALUE};

#[derive(Args, Debug)]
pub struct SpecArgs {
    #[command(subcommand)]
    command: SpecCommand,
}

#[derive(Subcommand, Debug)]
enum SpecCommand {
    /// Check a spec (.json or .toml) without opening its files; print its canonical JSON
    Check {
        spec: PathBuf,
        /// Only check; print nothing on success
        #[arg(short, long)]
        quiet: bool,
    },
    /// Print the JSON Schema of pipeline specs
    Schema,
}

pub fn run(args: &SpecArgs) -> anyhow::Result<()> {
    match &args.command {
        SpecCommand::Check { spec: path, quiet } => {
            let spec = PipelineSpec::load(path)?;
            let chain = spec.chain()?;
            for (id, node) in &chain.stages {
                node.stage().map_err(|e| anyhow::anyhow!("invalid pipeline: node {id:?}: {e}"))?;
            }
            if !quiet {
                println!("{}", spec.to_json());
                let names: Vec<&str> = std::iter::once(chain.read.0).chain(chain.stages.iter().map(|(id, _)| *id)).chain([chain.write.0]).collect();
                anstream::eprintln!("{} {}  {}", paint(OK, "✓"), paint(VALUE, path.display()), paint(DIM, names.join(" → ")));
            }
        }
        SpecCommand::Schema => println!("{}", PipelineSpec::json_schema()),
    }
    Ok(())
}
