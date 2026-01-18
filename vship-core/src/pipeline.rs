use crate::buffer::BufferView;
use crate::device::VulkanDevice;
use crate::error::{Result, VshipError};
use crate::shader::ShaderModule;
use ash::vk;
use std::ffi::CString;
use std::sync::Arc;

/// Descriptor set layout binding descriptor
#[derive(Clone)]
pub struct DescriptorBinding {
    pub binding: u32,
    pub descriptor_type: vk::DescriptorType,
    pub stage_flags: vk::ShaderStageFlags,
}

/// Push constant range descriptor
#[derive(Clone)]
pub struct PushConstantRange {
    pub offset: u32,
    pub size: u32,
    pub stage_flags: vk::ShaderStageFlags,
}

/// Compute pipeline builder
pub struct PipelineBuilder {
    shader: Option<Arc<ShaderModule>>,
    entry_point: String,
    descriptor_bindings: Vec<DescriptorBinding>,
    push_constants: Vec<PushConstantRange>,
}

impl PipelineBuilder {
    /// Create a new pipeline builder
    pub fn new() -> Self {
        Self {
            shader: None,
            entry_point: "main".to_string(),
            descriptor_bindings: Vec::new(),
            push_constants: Vec::new(),
        }
    }

    /// Set the compute shader
    pub fn shader(mut self, shader: Arc<ShaderModule>) -> Self {
        self.shader = Some(shader);
        self
    }

    /// Set the shader entry point
    pub fn entry_point(mut self, entry_point: impl Into<String>) -> Self {
        self.entry_point = entry_point.into();
        self
    }

    /// Add a descriptor binding
    pub fn add_binding(
        mut self,
        binding: u32,
        descriptor_type: vk::DescriptorType,
        stage_flags: vk::ShaderStageFlags,
    ) -> Self {
        self.descriptor_bindings.push(DescriptorBinding {
            binding,
            descriptor_type,
            stage_flags,
        });
        self
    }

    /// Add a storage buffer binding
    pub fn add_storage_buffer(self, binding: u32) -> Self {
        self.add_binding(
            binding,
            vk::DescriptorType::STORAGE_BUFFER,
            vk::ShaderStageFlags::COMPUTE,
        )
    }

    /// Add a uniform buffer binding
    pub fn add_uniform_buffer(self, binding: u32) -> Self {
        self.add_binding(
            binding,
            vk::DescriptorType::UNIFORM_BUFFER,
            vk::ShaderStageFlags::COMPUTE,
        )
    }

    /// Add push constants
    pub fn add_push_constants(mut self, offset: u32, size: u32) -> Self {
        self.push_constants.push(PushConstantRange {
            offset,
            size,
            stage_flags: vk::ShaderStageFlags::COMPUTE,
        });
        self
    }

    /// Build the compute pipeline
    pub fn build(self, device: Arc<VulkanDevice>) -> Result<ComputePipeline> {
        let shader = self
            .shader
            .ok_or_else(|| VshipError::PipelineError("No shader specified".to_string()))?;

        // Create descriptor set layout
        let bindings: Vec<vk::DescriptorSetLayoutBinding> = self
            .descriptor_bindings
            .iter()
            .map(|binding| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(binding.binding)
                    .descriptor_type(binding.descriptor_type)
                    .descriptor_count(1)
                    .stage_flags(binding.stage_flags)
            })
            .collect();

        let layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);

        let descriptor_set_layout = unsafe {
            device
                .device()
                .create_descriptor_set_layout(&layout_info, None)?
        };

        // Create pipeline layout
        let push_constant_ranges: Vec<vk::PushConstantRange> = self
            .push_constants
            .iter()
            .map(|pc| {
                vk::PushConstantRange::default()
                    .offset(pc.offset)
                    .size(pc.size)
                    .stage_flags(pc.stage_flags)
            })
            .collect();

        let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(std::slice::from_ref(&descriptor_set_layout))
            .push_constant_ranges(&push_constant_ranges);

        let pipeline_layout = unsafe {
            device
                .device()
                .create_pipeline_layout(&pipeline_layout_info, None)?
        };

        // Create compute pipeline
        let entry_point = CString::new(self.entry_point)
            .map_err(|e| VshipError::PipelineError(format!("Invalid entry point: {}", e)))?;

        let stage_info = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(shader.module())
            .name(&entry_point);

        let pipeline_info = vk::ComputePipelineCreateInfo::default()
            .stage(stage_info)
            .layout(pipeline_layout);

        let pipeline = unsafe {
            device
                .device()
                .create_compute_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
                .map_err(|(_, e)| e)?[0]
        };

        // Create descriptor pool with enough capacity for frame processing
        // We need many sets per frame: ~100 for all scales and channels
        let pool_sizes = self
            .descriptor_bindings
            .iter()
            .map(|binding| {
                vk::DescriptorPoolSize::default()
                    .ty(binding.descriptor_type)
                    .descriptor_count(256) // Support many descriptor sets per frame
            })
            .collect::<Vec<_>>();

        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(256)
            .pool_sizes(&pool_sizes)
            .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET);

        let descriptor_pool = unsafe { device.device().create_descriptor_pool(&pool_info, None)? };

        Ok(ComputePipeline {
            device,
            pipeline,
            pipeline_layout,
            descriptor_set_layout,
            descriptor_pool,
            descriptor_bindings: self.descriptor_bindings,
        })
    }
}

impl Default for PipelineBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute pipeline with descriptor management
pub struct ComputePipeline {
    device: Arc<VulkanDevice>,
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,
    descriptor_bindings: Vec<DescriptorBinding>,
}

impl ComputePipeline {
    /// Get the Vulkan pipeline
    pub fn pipeline(&self) -> vk::Pipeline {
        self.pipeline
    }

    /// Get the pipeline layout
    pub fn layout(&self) -> vk::PipelineLayout {
        self.pipeline_layout
    }

    /// Allocate a descriptor set
    pub fn allocate_descriptor_set(&self) -> Result<vk::DescriptorSet> {
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(self.descriptor_pool)
            .set_layouts(std::slice::from_ref(&self.descriptor_set_layout));

        let descriptor_sets = unsafe { self.device.device().allocate_descriptor_sets(&alloc_info)? };

        Ok(descriptor_sets[0])
    }

    /// Reset the descriptor pool, freeing all allocated sets
    /// Call this between frames to reuse the pool
    pub fn reset_descriptor_pool(&self) -> Result<()> {
        unsafe {
            self.device
                .device()
                .reset_descriptor_pool(self.descriptor_pool, vk::DescriptorPoolResetFlags::empty())?;
        }
        Ok(())
    }

    /// Update descriptor set with buffer bindings
    pub fn update_descriptor_set(
        &self,
        descriptor_set: vk::DescriptorSet,
        buffers: &[(u32, BufferView)],
    ) {
        let buffer_infos: Vec<vk::DescriptorBufferInfo> = buffers
            .iter()
            .map(|(_, view)| view.descriptor_info())
            .collect();

        let writes: Vec<vk::WriteDescriptorSet> = buffers
            .iter()
            .enumerate()
            .map(|(i, (binding, _))| {
                let descriptor_type = self
                    .descriptor_bindings
                    .iter()
                    .find(|b| b.binding == *binding)
                    .map(|b| b.descriptor_type)
                    .unwrap_or(vk::DescriptorType::STORAGE_BUFFER);

                vk::WriteDescriptorSet::default()
                    .dst_set(descriptor_set)
                    .dst_binding(*binding)
                    .descriptor_type(descriptor_type)
                    .buffer_info(std::slice::from_ref(&buffer_infos[i]))
            })
            .collect();

        unsafe {
            self.device.device().update_descriptor_sets(&writes, &[]);
        }
    }

    /// Bind pipeline to command buffer
    pub fn bind(&self, command_buffer: vk::CommandBuffer) {
        unsafe {
            self.device.device().cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline,
            );
        }
    }

    /// Bind descriptor sets
    pub fn bind_descriptor_sets(
        &self,
        command_buffer: vk::CommandBuffer,
        descriptor_sets: &[vk::DescriptorSet],
    ) {
        unsafe {
            self.device.device().cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline_layout,
                0,
                descriptor_sets,
                &[],
            );
        }
    }

    /// Push constants
    pub fn push_constants<T: Copy>(
        &self,
        command_buffer: vk::CommandBuffer,
        offset: u32,
        data: &T,
    ) {
        let bytes = unsafe {
            std::slice::from_raw_parts(
                data as *const T as *const u8,
                std::mem::size_of::<T>(),
            )
        };

        unsafe {
            self.device.device().cmd_push_constants(
                command_buffer,
                self.pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                offset,
                bytes,
            );
        }
    }

    /// Dispatch compute work
    pub fn dispatch(
        &self,
        command_buffer: vk::CommandBuffer,
        group_count_x: u32,
        group_count_y: u32,
        group_count_z: u32,
    ) {
        unsafe {
            self.device.device().cmd_dispatch(
                command_buffer,
                group_count_x,
                group_count_y,
                group_count_z,
            );
        }
    }
}

impl Drop for ComputePipeline {
    fn drop(&mut self) {
        unsafe {
            self.device
                .device()
                .destroy_descriptor_pool(self.descriptor_pool, None);
            self.device
                .device()
                .destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            self.device
                .device()
                .destroy_pipeline_layout(self.pipeline_layout, None);
            self.device.device().destroy_pipeline(self.pipeline, None);
        }
    }
}
