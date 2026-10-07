//! Running work on a [`ComputeTarget`].
//!
//! Library code takes a [`Client`] (cubecl 0.11 resolves the runtime when the client is made, not
//! through a generic); the CLI, Python bindings and the app pick a target once and get its client,
//! or run a [`ComputeTask`] on it.

use cubecl::device::{AmdDevice, CpuDevice, CudaDevice, WgpuDevice};
use cubecl::prelude::*;
use cubecl::Device;

use crate::device::{ComputeError, ComputeTarget};

/// Work to run on whichever runtime a [`ComputeTarget`] selects.
pub trait ComputeTask {
    type Output;
    fn run(self, client: Client) -> Self::Output;
}

impl ComputeTarget {
    /// The default device of this runtime, if it is compiled in.
    pub fn device(self) -> Result<Device, ComputeError> {
        Ok(match self.checked()? {
            Self::Wgpu => Device::Wgpu(WgpuDevice::default()),
            Self::Cpu => Device::Cpu(CpuDevice),
            Self::Cuda => Device::Cuda(CudaDevice::default()),
            Self::Hip => Device::Hip(AmdDevice::default()),
        })
    }

    /// A client for this runtime's default device.
    pub fn client(self) -> Result<Client, ComputeError> {
        Ok(self.device()?.client())
    }

    /// Runs `task` with a client for this runtime's default device.
    pub fn run<T: ComputeTask>(self, task: T) -> Result<T::Output, ComputeError> {
        Ok(task.run(self.client()?))
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

    #[test]
    fn runs_on_available_targets_and_refuses_missing_ones() {
        for target in ComputeTarget::available() {
            assert!(!target.run(RuntimeName).unwrap().is_empty());
        }
        for target in ComputeTarget::ALL.into_iter().filter(|t| !t.is_available()) {
            assert!(matches!(target.run(RuntimeName), Err(ComputeError::Unavailable(..))));
        }
    }
}
