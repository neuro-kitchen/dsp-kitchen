//! Running work on a [`ComputeTarget`].
//!
//! Library code takes a [`Client`] (cubecl 0.11 resolves the runtime when the client is made, not
//! through a generic); the CLI, Python bindings and the app pick a target once and get its client,
//! or run a [`ComputeTask`] on it.
//!
//! **Devices are shut down at exit.** CubeCL runs each device on a runner thread that nothing joins,
//! so the device is never destroyed and the GPU driver's own worker threads (NVIDIA's shader
//! compiler) were still running while the process exited: an intermittent crash or hang after all
//! the work was done (SIGSEGV in `libnvidia-glvkspirv`, "corrupted double-linked list" in
//! `libnvidia-gpucomp`, with the main thread inside `exit()`). Every device a client is made for is
//! recorded, and an exit handler, registered after each device is opened (so it runs before the
//! handlers of the libraries and layers loaded until then), shuts them down: [`shutdown_devices`].
//!
//! **Autotune decisions and compiled kernels persist across processes** ([`configure_runtime`]):
//! without it every process re-tuned (about a minute on a GPU, several with the CPU runtime) and
//! recompiled every kernel.

use std::sync::Mutex;

use cubecl::device::{AmdDevice, CpuDevice, CudaDevice, WgpuDevice};
use cubecl::device_handle::DeviceHandle;
use cubecl::prelude::*;
use cubecl::Device;

use crate::device::{ComputeError, ComputeTarget};

/// Environment variable: `0` turns off the compiled-kernel cache (autotune decisions still persist).
pub const KERNEL_CACHE_ENV: &str = "DSPK_KERNEL_CACHE";

/// Environment variable: directory for CubeCL's caches (autotune decisions, compiled kernels);
/// default: the user cache directory.
pub const CACHE_DIR_ENV: &str = "DSPK_CACHE_DIR";

/// Sets CubeCL's runtime configuration once, before the first device opens: compiled kernels are
/// cached (CubeCL's default is off), and the caches live in the user cache directory (or
/// `DSPK_CACHE_DIR`) instead of the `target/` of whichever Cargo project the process runs in.
/// A `cubecl.toml` or CubeCL's own environment variables still apply. Called by
/// [`ComputeTarget::device`] and [`open_device`]; harmless to call again.
pub fn configure_runtime() {
    use cubecl::config::cache::CacheConfig;
    use cubecl::config::{CubeClRuntimeConfig, RuntimeConfig};
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        let mut config = CubeClRuntimeConfig::from_current_dir().override_from_env();
        config.compilation.cache = std::env::var(KERNEL_CACHE_ENV).map_or(true, |v| v.trim() != "0");
        match std::env::var_os(CACHE_DIR_ENV) {
            Some(dir) if !dir.is_empty() => config.environment.path = CacheConfig::Directory(dir.into()),
            _ if matches!(config.environment.path, CacheConfig::Target) => config.environment.path = CacheConfig::Global,
            _ => {}
        }
        // `false` when a configuration was already set (e.g. by the embedding application): keep it
        let _ = CubeClRuntimeConfig::try_set(config);
    });
}

/// Work to run on whichever runtime a [`ComputeTarget`] selects.
pub trait ComputeTask {
    type Output;
    fn run(self, client: Client) -> Self::Output;
}

impl ComputeTarget {
    /// The default device of this runtime, if it is compiled in.
    pub fn device(self) -> Result<Device, ComputeError> {
        configure_runtime();
        Ok(match self.checked()? {
            Self::Wgpu => Device::Wgpu(WgpuDevice::default()),
            Self::Cpu => Device::Cpu(CpuDevice),
            Self::Cuda => Device::Cuda(CudaDevice::default()),
            Self::Hip => Device::Hip(AmdDevice::default()),
        })
    }

    /// A client for this runtime's default device (recorded for [`shutdown_devices`]).
    pub fn client(self) -> Result<Client, ComputeError> {
        Ok(open_device(self.device()?))
    }

    /// Runs `task` with a client for this runtime's default device.
    pub fn run<T: ComputeTask>(self, task: T) -> Result<T::Output, ComputeError> {
        Ok(task.run(self.client()?))
    }
}

/// Devices a client was made for, shut down at exit.
static OPENED: Mutex<Vec<Device>> = Mutex::new(Vec::new());

unsafe extern "C" {
    /// The C library's exit-handler registration (handlers run in reverse order of registration).
    fn atexit(handler: extern "C" fn()) -> std::ffi::c_int;
}

extern "C" fn shutdown_at_exit() {
    shutdown_devices();
}

/// Records `device`, newly opened, and registers [`shutdown_devices`] to run at exit. Called after
/// the client exists: exit handlers run in reverse order of registration, and opening a device
/// loads Vulkan layers that register their own (the validation layer destroys its device records
/// in one), so ours must come later to run first. Registered again for each newly opened device,
/// since a device reopened after a shutdown may load those layers again; running it more than once
/// is harmless (the list is empty after the first run).
fn opened(device: &Device) {
    let mut devices = OPENED.lock().unwrap_or_else(|e| e.into_inner());
    if devices.contains(device) {
        return;
    }
    devices.push(device.clone());
    drop(devices);
    // SAFETY: registers a plain `extern "C"` function; failure (no slot left) only means the
    // device is not shut down at exit, as before
    unsafe {
        atexit(shutdown_at_exit);
    }
}

/// The client of `device`, recorded so that [`shutdown_devices`] shuts it down at exit. Use this
/// instead of `Device::client` wherever a specific device is opened (e.g. one GPU of several).
pub fn open_device(device: Device) -> Client {
    configure_runtime();
    let client = device.client();
    opened(&device);
    client
}

/// Shuts down every device a client was made for: their runner threads stop (queued work runs
/// first) and the devices are destroyed, so the GPU driver's threads end before the process does.
/// Runs at exit by itself; call it earlier to release the GPUs. Clients still alive keep their
/// runner going (CubeCL waits up to 30 s for it, then leaves it), so drop them first. A later
/// [`ComputeTarget::client`] opens the device again.
pub fn shutdown_devices() {
    let devices = std::mem::take(&mut *OPENED.lock().unwrap_or_else(|e| e.into_inner()));
    for device in devices {
        // The runner is registered under the runtime's own id, not `Device`'s runtime-stamped one
        DeviceHandle::<()>::shutdown(cubecl::RuntimeId::strip(device.to_id()));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct RuntimeName;
    impl ComputeTask for RuntimeName {
        type Output = String;
        fn run(self, client: Client) -> String {
            client.name().to_string()
        }
    }

    /// Threads of this process (Linux).
    fn threads() -> usize {
        std::fs::read_dir("/proc/self/task").map(|d| d.count()).unwrap_or(0)
    }

    /// Shutting the devices down stops their runner (and driver) threads; a later client opens the
    /// device again.
    #[test]
    #[cfg(target_os = "linux")]
    fn shutdown_stops_the_device_threads() {
        let Some(target) = ComputeTarget::for_tests().into_iter().find(|t| *t != ComputeTarget::Cpu) else { return };
        let client = target.client().unwrap();
        let data: Vec<u8> = (0..64u8).collect();
        let handle = client.create(cubecl::bytes::Bytes::from_bytes_vec(data.clone()));
        assert_eq!(client.read_one_unchecked(handle).to_vec(), data);
        drop(client);
        let open = threads();
        shutdown_devices();
        let closed = threads();
        assert!(closed < open, "{open} threads with the device open, {closed} after shutdown");
        let again = target.client().unwrap();
        assert!(!again.name().is_empty());
    }

    #[test]
    fn runs_on_available_targets_and_refuses_missing_ones() {
        for target in ComputeTarget::for_tests() {
            assert!(!target.run(RuntimeName).unwrap().is_empty());
        }
        for target in ComputeTarget::ALL.into_iter().filter(|t| !t.is_available()) {
            assert!(matches!(target.run(RuntimeName), Err(ComputeError::Unavailable(..))));
        }
    }
}
