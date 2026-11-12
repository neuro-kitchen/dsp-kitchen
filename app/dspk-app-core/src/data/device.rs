//! The compute device processing runs on, named for the status bar and Settings › Compute (as
//! lumen's): the runtime chosen in Settings (`DSPK_RUNTIME` at start), else the first compiled in
//! (wgpu); with wgpu, the adapter it opens: the discrete GPU first, then the integrated one, a
//! virtual one, and a software one (llvmpipe) last.

use dspk_core::ComputeTarget;

/// What the status bar and Settings say about the device.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceInfo {
    /// One short line: "GPU · Vulkan · Intel(R) UHD Graphics 630", "CPU".
    pub label: String,
    /// Every adapter wgpu found (name, kind, backend), for Settings.
    pub adapters: Vec<String>,
}

impl DeviceInfo {
    /// The device of the runtime chosen now (takes a moment with wgpu: call it off the UI thread).
    pub fn discover() -> DeviceInfo {
        match ComputeTarget::from_env() {
            Ok(ComputeTarget::Wgpu) => wgpu_device(),
            Ok(ComputeTarget::Cuda) => DeviceInfo { label: "GPU · CUDA".into(), adapters: Vec::new() },
            Ok(ComputeTarget::Hip) => DeviceInfo { label: "GPU · ROCm".into(), adapters: Vec::new() },
            Ok(ComputeTarget::Cpu) => DeviceInfo { label: "CPU".into(), adapters: Vec::new() },
            Err(e) => DeviceInfo { label: format!("No compute device: {e}"), adapters: Vec::new() },
        }
    }
}

#[cfg(feature = "wgpu")]
fn wgpu_device() -> DeviceInfo {
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

#[cfg(not(feature = "wgpu"))]
fn wgpu_device() -> DeviceInfo {
    DeviceInfo { label: "GPU · wgpu (not compiled in)".into(), adapters: Vec::new() }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The device is named, and with wgpu (the default) every adapter it found is listed.
    #[test]
    fn the_device_is_named() {
        let d = DeviceInfo::discover();
        eprintln!("compute device: {} · adapters: {:?}", d.label, d.adapters);
        assert!(!d.label.is_empty());
        if ComputeTarget::from_env().ok() == Some(ComputeTarget::Wgpu) {
            assert!(d.label.starts_with("GPU · "), "{}", d.label);
        }
    }
}
