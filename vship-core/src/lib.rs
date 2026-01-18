// Vship Core - Vulkan compute abstraction layer
// Provides GPU device management, memory allocation, and compute pipeline execution

pub mod device;
pub mod memory;
pub mod pipeline;
pub mod buffer;
pub mod shader;
pub mod shader_manager;
pub mod compute;
pub mod error;
pub mod color;

pub use device::{VulkanDevice, DeviceSelector, VulkanInstance};
pub use memory::{BufferAllocator, BufferUsage, AllocatedBuffer};
pub use pipeline::{ComputePipeline, PipelineBuilder};
pub use buffer::{Buffer, BufferView};
pub use shader::ShaderModule;
pub use shader_manager::ShaderManager;
pub use compute::{ComputeContext, compute_dispatch_size};
pub use error::{VshipError, Result};

use std::sync::Arc;

/// Vship context managing Vulkan instance and devices
pub struct VshipContext {
    instance: Arc<VulkanInstance>,
    devices: Vec<Arc<VulkanDevice>>,
}

impl VshipContext {
    /// Create a new Vship context with automatic device detection
    pub fn new() -> Result<Self> {
        // Create Vulkan instance (reference-counted)
        let instance = Arc::new(VulkanInstance::new()?);

        // Enumerate and create devices
        let physical_devices = instance.enumerate_physical_devices()?;
        let mut devices = Vec::new();

        for physical_device in physical_devices {
            match VulkanDevice::new(Arc::clone(&instance), physical_device) {
                Ok(device) => devices.push(Arc::new(device)),
                Err(e) => log::warn!("Failed to create device: {}", e),
            }
        }

        if devices.is_empty() {
            return Err(VshipError::NoDeviceFound);
        }

        log::info!("Initialized Vship with {} device(s)", devices.len());

        Ok(Self {
            instance,
            devices,
        })
    }

    /// Get the Vulkan instance
    pub fn instance(&self) -> &ash::Instance {
        self.instance.instance()
    }

    /// Get all available devices
    pub fn devices(&self) -> &[Arc<VulkanDevice>] {
        &self.devices
    }

    /// Get the default (first) device
    pub fn default_device(&self) -> Arc<VulkanDevice> {
        Arc::clone(&self.devices[0])
    }

    /// Select a device by index
    pub fn device(&self, index: usize) -> Option<Arc<VulkanDevice>> {
        self.devices.get(index).map(Arc::clone)
    }
}

// Note: VshipContext no longer needs a Drop impl because the instance
// is reference-counted via Arc<VulkanInstance>. The instance will only
// be destroyed when all devices (which also hold Arc<VulkanInstance>)
// have been dropped first, ensuring correct Vulkan destruction order.

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_context_creation() {
        let ctx = VshipContext::new().expect("Failed to create context");
        assert!(!ctx.devices().is_empty());
    }
}
