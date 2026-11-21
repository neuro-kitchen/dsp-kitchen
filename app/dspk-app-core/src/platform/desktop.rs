//! The desktop (Linux, macOS, Windows).

/// Whether work can run on threads of its own (else it runs where it is asked for).
pub const THREADS: bool = true;

/// Runs `work` on a new thread called `name`.
pub fn spawn(name: &str, work: impl FnOnce() + Send + 'static) -> std::io::Result<()> {
    std::thread::Builder::new().name(name.into()).spawn(work).map(|_| ())
}

/// Trace frame timings for a measuring script: a page keeps them (`platform::web`); the desktop
/// shows them in the status bar only.
pub fn record_frame_timing(_render_ms: f64, _samples_per_px: f64, _complete: bool) {}

/// Where recordings come from, for the title bar.
pub fn host_label() -> String {
    "This computer".into()
}

/// Whether recordings are opened from files on this computer (a file dialog, recent files).
pub const LOCAL_FILES: bool = true;


/// Every adapter wgpu finds, the best first.
#[cfg(feature = "wgpu")]
pub fn wgpu_device() -> crate::data::device::DeviceInfo {
    use crate::data::device::DeviceInfo;
    use wgpu::DeviceType as T;
    let found = tokio::runtime::Builder::new_current_thread().build().map(|rt| rt.block_on(wgpu::Instance::default().enumerate_adapters(wgpu::Backends::all())));
    let infos: Vec<wgpu::AdapterInfo> = found.map(|list| list.iter().map(|a| a.get_info()).collect()).unwrap_or_default();
    let rank = |t: T| match t {
        T::DiscreteGpu => 0,
        T::IntegratedGpu => 1,
        T::VirtualGpu => 2,
        T::Other => 3,
        T::Cpu => 4,
    };
    let kind = |t: T| match t {
        T::DiscreteGpu => "discrete",
        T::IntegratedGpu => "integrated",
        T::VirtualGpu => "virtual",
        T::Other => "other",
        T::Cpu => "software",
    };
    let adapters = infos.iter().map(|i| format!("{} ({}, {:?})", i.name, kind(i.device_type), i.backend)).collect();
    let label = match infos.iter().min_by_key(|i| rank(i.device_type)) {
        Some(i) if i.device_type == T::Cpu => format!("GPU · {:?} · {} (software)", i.backend, i.name),
        Some(i) => format!("GPU · {:?} · {}", i.backend, i.name),
        None => "GPU · Vulkan / Metal (no adapter found)".to_string(),
    };
    DeviceInfo { label, adapters }
}

