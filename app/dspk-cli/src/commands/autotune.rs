//! `autotune`: benchmark the tuned kernels on this machine now, and save the choices.

use clap::Args;
use dspk_core::compute::ComputeTarget;
use dspk_ops::autotune::{autotune, AutotuneOptions, DEFAULT_CHANNELS, DEFAULT_SAMPLES};

use crate::style::{paint, OK, WARN};

#[derive(Args, Debug)]
pub struct AutotuneArgs {
    /// Channel counts to tune
    #[arg(long, value_delimiter = ',', default_values_t = DEFAULT_CHANNELS)]
    channels: Vec<usize>,
    /// Samples per channel to tune
    #[arg(long, value_delimiter = ',', default_values_t = DEFAULT_SAMPLES)]
    samples: Vec<usize>,
}

/// Tunes every runtime compiled in that can open a device here, or `--runtime`'s (`DSPK_RUNTIME`).
pub fn run(args: &AutotuneArgs) -> anyhow::Result<()> {
    let options = AutotuneOptions { channels: args.channels.clone(), samples: args.samples.clone(), ..Default::default() };
    let targets: Vec<ComputeTarget> = match std::env::var(dspk_core::compute::RUNTIME_ENV) {
        Ok(_) => vec![ComputeTarget::from_env()?],
        Err(_) => ComputeTarget::ALL.into_iter().filter(|t| t.is_available()).collect(),
    };
    for target in targets {
        let name = target.name();
        let opened = std::panic::catch_unwind(|| target.client());
        let Ok(Ok(client)) = opened else {
            println!("{} {name}: no device here", paint(WARN, "skip"));
            continue;
        };
        let summary = autotune(&client, &options, |done, total| eprint!("\r{name}: {done}/{total}"))?;
        eprintln!();
        println!("{} {name}: {} sizes in {:.0} s", paint(OK, "ok"), summary.sizes, summary.seconds);
    }
    Ok(())
}
