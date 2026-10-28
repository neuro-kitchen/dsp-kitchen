//! `doctor`: what this build can do on this machine, as facts (no fixed text).

use cubecl::prelude::*;
use dspk_core::compute::{ComputeTarget, ComputeTask};

use crate::style::{paint, DIM, HEADING, OK, VALUE, WARN};

/// The device a runtime opens, described from its reported properties.
struct Describe;

impl ComputeTask for Describe {
    type Output = String;
    fn run(self, client: Client) -> String {
        let hw = &client.properties().hardware;
        let mut parts = vec![
            format!("plane {}", hw.plane_size_max),
            format!("≤ {} units per cube", hw.max_units_per_cube),
        ];
        parts.extend(hw.num_streaming_multiprocessors.map(|n| format!("{n} multiprocessors")));
        parts.extend(hw.num_cpu_cores.map(|n| format!("{n} cores")));
        format!("{} ({})", client.name(), parts.join(", "))
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
    anstream::println!(
        "{} {}",
        paint(HEADING, format!("dspk {}", env!("CARGO_PKG_VERSION"))),
        paint(DIM, format!("({} / {})", std::env::consts::OS, std::env::consts::ARCH))
    );
    let on: Vec<&str> = FEATURES.iter().filter(|(_, on)| *on).map(|(name, _)| *name).collect();
    anstream::println!("{}   {}", paint(DIM, "Features:"), paint(VALUE, on.join(", ")));

    anstream::println!("{} {}", paint(DIM, "Runtimes"), paint(DIM, "(compiled in, in preference order):"));
    for target in ComputeTarget::available() {
        // A runtime without a device (no GPU, no driver) panics when opening its client
        let opened = std::panic::catch_unwind(|| target.run(Describe));
        match opened {
            Ok(Ok(device)) => anstream::println!("  {} {} {device}", paint(VALUE, format!("{:<5}", target.name())), paint(OK, "usable")),
            Ok(Err(e)) => anstream::println!("  {} {} {e}", paint(VALUE, format!("{:<5}", target.name())), paint(WARN, "not usable:")),
            Err(_) => anstream::println!("  {} {} no device", paint(VALUE, format!("{:<5}", target.name())), paint(WARN, "not usable:")),
        }
    }
    if let Ok(selected) = ComputeTarget::from_env() {
        anstream::println!(
            "{}   {} {}",
            paint(DIM, "Selected:"),
            paint(VALUE, selected),
            paint(DIM, format!("({} or the first compiled in)", dspk_core::compute::RUNTIME_ENV))
        );
    }

    #[cfg(feature = "hub")]
    anstream::println!("{}  {}", paint(DIM, "Hub cache:"), paint(VALUE, dspk_synapse_hub::Hub::from_env_or_default().cache().root_dir().display()));
    Ok(())
}
