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
    let mut usable = Vec::new();
    for target in ComputeTarget::available() {
        match probe(target) {
            Ok(device) => {
                usable.push(target);
                anstream::println!("  {} {} {device}", paint(VALUE, format!("{:<5}", target.name())), paint(OK, "usable"));
            }
            Err(why) => anstream::println!("  {} {} {why}", paint(VALUE, format!("{:<5}", target.name())), paint(WARN, "not usable:")),
        }
    }
    if let Ok(selected) = ComputeTarget::from_env() {
        anstream::println!(
            "{}   {} {}",
            paint(DIM, "Selected:"),
            paint(VALUE, selected),
            paint(DIM, format!("({} or the first compiled in)", dspk_core::compute::RUNTIME_ENV))
        );
        if !usable.contains(&selected) {
            match usable.first() {
                Some(other) => anstream::println!(
                    "  {} {selected} has no device here: set {}={other}, or pass --runtime {other}",
                    paint(WARN, "warning:"),
                    dspk_core::compute::RUNTIME_ENV
                ),
                None => anstream::println!("  {} no runtime has a device here", paint(WARN, "warning:")),
            }
        }
    }

    #[cfg(feature = "hub")]
    anstream::println!("{}  {}", paint(DIM, "Hub cache:"), paint(VALUE, dspk_hub::Hub::from_env_or_default().cache().root_dir().display()));
    Ok(())
}

/// Opens `target`'s device and describes it; a runtime without a device (no GPU, no driver)
/// panics while opening its client, which is caught and kept short (the default panic message,
/// printed from the runtime's own threads too, is silenced meanwhile).
fn probe(target: ComputeTarget) -> Result<String, String> {
    let reason = std::sync::Arc::new(std::sync::Mutex::new(None::<String>));
    let hook = std::panic::take_hook();
    let seen = reason.clone();
    std::panic::set_hook(Box::new(move |info| {
        let text = info.payload().downcast_ref::<String>().cloned().or_else(|| info.payload().downcast_ref::<&str>().map(|s| s.to_string()));
        let mut slot = seen.lock().unwrap_or_else(|e| e.into_inner());
        // The first panic is the cause (the device thread's); later ones only report it
        if slot.is_none() {
            *slot = text;
        }
    }));
    let opened = std::panic::catch_unwind(|| target.run(Describe));
    std::panic::set_hook(hook);
    match opened {
        Ok(Ok(device)) => Ok(device),
        Ok(Err(e)) => Err(e.to_string()),
        Err(_) => Err(short_reason(reason.lock().unwrap_or_else(|e| e.into_inner()).as_deref())),
    }
}

/// The gist of a runtime's panic: no driver library, no adapter, or its first line.
fn short_reason(panic: Option<&str>) -> String {
    let Some(text) = panic else { return "no device".into() };
    if text.contains("dynamically load") {
        let lib = text.split('"').nth(1).unwrap_or("driver");
        return format!("no {lib} library (driver not installed, or not passed into this container)");
    }
    if text.contains("no adapter found") || text.contains("AdapterUnavailable") {
        return "no adapter (no GPU, or no Vulkan driver)".into();
    }
    let first = text.lines().next().unwrap_or(text);
    first.chars().take(160).collect()
}
