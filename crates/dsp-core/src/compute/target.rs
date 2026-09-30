//! Running work on a [`ComputeTarget`].
//!
//! Library code is generic over `R: cubecl::Runtime`; the CLI, Python bindings and the app pick a
//! target once and run a [`ComputeTask`] on it.

use cubecl::prelude::*;

use crate::device::{ComputeError, ComputeTarget};

/// Work to run on whichever runtime a [`ComputeTarget`] selects.
pub trait ComputeTask {
    type Output;
    fn run<R: Runtime>(self, client: ComputeClient<R>) -> Self::Output;
}

impl ComputeTarget {
    /// Runs `task` with a client for this runtime's default device.
    pub fn run<T: ComputeTask>(self, task: T) -> Result<T::Output, ComputeError> {
        match self.checked()? {
            #[cfg(feature = "wgpu")]
            Self::Wgpu => {
                use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
                Ok(task.run(WgpuRuntime::client(&WgpuDevice::default())))
            }
            #[cfg(feature = "cpu")]
            Self::Cpu => {
                use cubecl::cpu::{CpuDevice, CpuRuntime};
                Ok(task.run(CpuRuntime::client(&CpuDevice)))
            }
            #[cfg(feature = "cuda")]
            Self::Cuda => {
                use cubecl::cuda::{CudaDevice, CudaRuntime};
                Ok(task.run(CudaRuntime::client(&CudaDevice::default())))
            }
            #[cfg(feature = "hip")]
            Self::Hip => {
                use cubecl::hip::{AmdDevice, HipRuntime};
                Ok(task.run(HipRuntime::client(&AmdDevice::default())))
            }
            #[allow(unreachable_patterns)]
            other => Err(ComputeError::Unavailable(other, other.name())),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct RuntimeName;
    impl ComputeTask for RuntimeName {
        type Output = String;
        fn run<R: Runtime>(self, client: ComputeClient<R>) -> String {
            R::name(&client).to_string()
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
