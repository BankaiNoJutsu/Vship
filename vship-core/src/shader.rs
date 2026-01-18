use crate::device::VulkanDevice;
use crate::error::{Result, VshipError};
use ash::vk;
use std::sync::Arc;

/// Shader module wrapper
pub struct ShaderModule {
    device: Arc<VulkanDevice>,
    module: vk::ShaderModule,
}

impl ShaderModule {
    /// Create shader module from SPIR-V bytecode
    pub fn from_spirv(device: Arc<VulkanDevice>, spirv: &[u32]) -> Result<Self> {
        let create_info = vk::ShaderModuleCreateInfo::default().code(spirv);

        let module = unsafe {
            device
                .device()
                .create_shader_module(&create_info, None)?
        };

        Ok(Self { device, module })
    }

    /// Create shader module from SPIR-V file
    pub fn from_file(device: Arc<VulkanDevice>, path: &str) -> Result<Self> {
        let bytes = std::fs::read(path)?;

        // Ensure proper alignment
        if bytes.len() % 4 != 0 {
            return Err(VshipError::ShaderCompilationError(
                "SPIR-V file size is not a multiple of 4".to_string(),
            ));
        }

        let spirv: Vec<u32> = bytes
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();

        Self::from_spirv(device, &spirv)
    }

    /// Get the Vulkan shader module
    pub fn module(&self) -> vk::ShaderModule {
        self.module
    }
}

impl Drop for ShaderModule {
    fn drop(&mut self) {
        unsafe {
            self.device.device().destroy_shader_module(self.module, None);
        }
    }
}

/// Macro for embedding SPIR-V shaders at compile time
#[macro_export]
macro_rules! include_spirv {
    ($path:expr) => {{
        let bytes = include_bytes!($path);
        let spirv: &[u32] = unsafe {
            std::slice::from_raw_parts(
                bytes.as_ptr() as *const u32,
                bytes.len() / std::mem::size_of::<u32>(),
            )
        };
        spirv
    }};
}
