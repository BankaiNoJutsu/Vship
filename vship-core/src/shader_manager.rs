use crate::device::VulkanDevice;
use crate::shader::ShaderModule;
use crate::error::{Result, VshipError};
use std::collections::HashMap;
use std::sync::Arc;
use std::path::PathBuf;

/// Shader manager for loading and caching compiled SPIR-V shaders
pub struct ShaderManager {
    device: Arc<VulkanDevice>,
    shaders: HashMap<String, Arc<ShaderModule>>,
    shader_dir: PathBuf,
}

impl ShaderManager {
    /// Create a new shader manager
    pub fn new(device: Arc<VulkanDevice>) -> Self {
        // Default shader directory
        let shader_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../shaders/spirv");

        Self {
            device,
            shaders: HashMap::new(),
            shader_dir,
        }
    }

    /// Create shader manager with custom shader directory
    pub fn with_shader_dir(device: Arc<VulkanDevice>, shader_dir: PathBuf) -> Self {
        Self {
            device,
            shaders: HashMap::new(),
            shader_dir,
        }
    }

    /// Load a shader by name (without .spv extension)
    pub fn load_shader(&mut self, name: &str) -> Result<Arc<ShaderModule>> {
        // Check if already loaded
        if let Some(shader) = self.shaders.get(name) {
            return Ok(Arc::clone(shader));
        }

        // Load from file
        let path = self.shader_dir.join(format!("{}.spv", name));

        if !path.exists() {
            return Err(VshipError::ShaderCompilationError(
                format!("Shader file not found: {:?}", path)
            ));
        }

        let shader = ShaderModule::from_file(Arc::clone(&self.device), path.to_str().unwrap())?;
        let shader_arc = Arc::new(shader);

        self.shaders.insert(name.to_string(), Arc::clone(&shader_arc));

        Ok(shader_arc)
    }

    /// Get a previously loaded shader
    pub fn get_shader(&self, name: &str) -> Option<Arc<ShaderModule>> {
        self.shaders.get(name).map(Arc::clone)
    }

    /// Preload commonly used shaders
    pub fn preload_common_shaders(&mut self) -> Result<()> {
        let common_shaders = vec![
            "gaussian_blur_error_reduce",
            "rgba8_to_xyb",
            "rgba8_to_planar",
            "rgb_to_xyb",
            "gaussian_blur",
            "downsample",
            "ssim_error",
            "reduce_sum",
        ];

        for shader_name in common_shaders {
            if let Err(e) = self.load_shader(shader_name) {
                log::warn!("Failed to preload shader '{}': {}", shader_name, e);
            } else {
                log::info!("Preloaded shader: {}", shader_name);
            }
        }

        Ok(())
    }

    /// Clear shader cache
    pub fn clear_cache(&mut self) {
        self.shaders.clear();
    }

    /// Get number of cached shaders
    pub fn cache_size(&self) -> usize {
        self.shaders.len()
    }
}

/// Helper macro to include SPIR-V shaders at compile time
#[macro_export]
macro_rules! include_shader_spirv {
    ($name:expr) => {{
        let spirv_path = concat!(env!("CARGO_MANIFEST_DIR"), "/../shaders/spirv/", $name, ".spv");
        include_bytes!(spirv_path)
    }};
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore] // Requires Vulkan device
    fn test_shader_manager_creation() {
        // This test requires a valid Vulkan device
        // Run with: cargo test -- --ignored
    }
}
