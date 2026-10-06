//! `doctor`: what this build can do on this machine, as facts (no fixed text).

use cubecl::prelude::*;
use dsp_core::compute::{ComputeTarget, ComputeTask};

/// The device a runtime opens, described from its reported properties.
struct Describe;

impl ComputeTask for Describe {
    type Output = String;
    fn run<R: Runtime>(self, client: ComputeClient<R>) -> String {
        let hw = &client.properties().hardware;
        let mut parts = vec![
            format!("plane {}", hw.plane_size_max),
            format!("≤ {} units per cube", hw.max_units_per_cube),
        ];
        parts.extend(hw.num_streaming_multiprocessors.map(|n| format!("{n} multiprocessors")));
        parts.extend(hw.num_cpu_cores.map(|n| format!("{n} cores")));
        format!("{} ({})", R::name(&client), parts.join(", "))
    }
}

/// Cargo features of this build that change what it can do.
const FEATURES: [(&str, bool); 5] = [
    ("wgpu", cfg!(feature = "wgpu")),
    ("cpu", cfg!(feature = "cpu")),
    ("cuda", cfg!(feature = "cuda")),
    ("hip", cfg!(feature = "hip")),
    ("hub", cfg!(feature = "hub")),
];

pub fn run() -> anyhow::Result<()> {
    println!("dsp-cli {} ({} / {})", env!("CARGO_PKG_VERSION"), std::env::consts::OS, std::env::consts::ARCH);
    let on: Vec<&str> = FEATURES.iter().filter(|(_, on)| *on).map(|(name, _)| *name).collect();
    println!("Features:   {}", on.join(", "));

    println!("Runtimes (compiled in, in preference order):");
    for target in ComputeTarget::available() {
        // A runtime without a device (no GPU, no driver) panics when opening its client
        let opened = std::panic::catch_unwind(|| target.run(Describe));
        match opened {
            Ok(Ok(device)) => println!("  {:<5} usable: {device}", target.name()),
            Ok(Err(e)) => println!("  {:<5} not usable: {e}", target.name()),
            Err(_) => println!("  {:<5} not usable: no device", target.name()),
        }
    }
    if let Ok(selected) = ComputeTarget::from_env() {
        println!("Selected:   {selected} ({} or the first compiled in)", dsp_core::compute::RUNTIME_ENV);
    }

    #[cfg(feature = "hub")]
    println!("Hub cache:  {}", dsp_synapse_hub::Hub::from_env_or_default().cache().root_dir().display());
    Ok(())
}
