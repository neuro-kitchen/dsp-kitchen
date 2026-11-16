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

/// wgpu's device on this target (`crate::platform`): a computer lists its adapters; a browser
/// hands out its GPU only when work first asks for it.
#[cfg(feature = "wgpu")]
fn wgpu_device() -> DeviceInfo {
    crate::platform::wgpu_device()
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
