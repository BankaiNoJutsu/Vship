// Compute context for executing GPU operations

use crate::device::VulkanDevice;
use crate::memory::{AllocatedBuffer, BufferAllocator};
use crate::pipeline::ComputePipeline;
use crate::error::{Result, VshipError};
use ash::vk;
use std::sync::Arc;

/// Compute context for GPU shader execution
pub struct ComputeContext {
    device: Arc<VulkanDevice>,
    allocator: BufferAllocator,
    command_buffers: Vec<vk::CommandBuffer>,
    fence: vk::Fence,
}

impl ComputeContext {
    /// Create a new compute context
    pub fn new(device: Arc<VulkanDevice>, instance: &ash::Instance) -> Result<Self> {
        let allocator = BufferAllocator::new(
            Arc::clone(&device),
            instance,
            device.physical_device(),
        )?;

        let command_buffers = device.allocate_command_buffers(1)?;
        let fence = device.create_fence(false)?;

        Ok(Self {
            device,
            allocator,
            command_buffers,
            fence,
        })
    }

    /// Get device
    pub fn device(&self) -> &Arc<VulkanDevice> {
        &self.device
    }

    /// Get allocator
    pub fn allocator(&self) -> &BufferAllocator {
        &self.allocator
    }

    /// Begin recording commands
    pub fn begin_commands(&self) -> Result<vk::CommandBuffer> {
        let cmd = self.command_buffers[0];

        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        unsafe {
            self.device.device().begin_command_buffer(cmd, &begin_info)?;
        }

        Ok(cmd)
    }

    /// End recording and submit commands
    pub fn submit_and_wait(&self, cmd: vk::CommandBuffer) -> Result<()> {
        unsafe {
            self.device.device().end_command_buffer(cmd)?;
        }

        // Reset fence
        self.device.reset_fence(self.fence)?;

        // Submit
        self.device.submit_compute(&[cmd], &[], &[], self.fence)?;

        // Wait for completion (10 second timeout)
        self.device.wait_for_fence(self.fence, 10_000_000_000)?;

        Ok(())
    }

    /// Execute a simple compute operation
    pub fn execute<F>(&self, f: F) -> Result<()>
    where
        F: FnOnce(vk::CommandBuffer) -> Result<()>,
    {
        let cmd = self.begin_commands()?;
        f(cmd)?;
        self.submit_and_wait(cmd)?;
        Ok(())
    }

    /// Upload data to GPU buffer
    pub fn upload_buffer<T: Copy>(
        &self,
        data: &[T],
        dst: &AllocatedBuffer,
    ) -> Result<()> {
        let size = std::mem::size_of_val(data) as u64;
        let mut staging = self.allocator.create_staging_buffer(size)?;

        // Write to staging buffer
        staging.write_data(data)?;

        // Copy to device buffer
        self.execute(|cmd| {
            unsafe {
                let region = vk::BufferCopy::default()
                    .src_offset(0)
                    .dst_offset(0)
                    .size(size);

                self.device.device().cmd_copy_buffer(
                    cmd,
                    staging.buffer(),
                    dst.buffer(),
                    &[region],
                );
            }
            Ok(())
        })?;

        Ok(())
    }

    /// Download data from GPU buffer
    pub fn download_buffer<T: Copy>(
        &self,
        src: &AllocatedBuffer,
        data: &mut [T],
    ) -> Result<()> {
        let size = std::mem::size_of_val(data) as u64;
        let mut readback = self.allocator.create_readback_buffer(size)?;

        // Copy from device to readback buffer
        self.execute(|cmd| {
            unsafe {
                let region = vk::BufferCopy::default()
                    .src_offset(0)
                    .dst_offset(0)
                    .size(size);

                self.device.device().cmd_copy_buffer(
                    cmd,
                    src.buffer(),
                    readback.buffer(),
                    &[region],
                );
            }
            Ok(())
        })?;

        // Read from readback buffer
        readback.read_data(data)?;

        Ok(())
    }

    /// Dispatch compute shader
    pub fn dispatch_shader(
        &self,
        pipeline: &ComputePipeline,
        descriptor_set: vk::DescriptorSet,
        push_constants: Option<&[u8]>,
        group_count_x: u32,
        group_count_y: u32,
        group_count_z: u32,
    ) -> Result<()> {
        self.execute(|cmd| {
            // Bind pipeline
            pipeline.bind(cmd);

            // Bind descriptor sets
            pipeline.bind_descriptor_sets(cmd, &[descriptor_set]);

            // Push constants if provided
            if let Some(constants) = push_constants {
                unsafe {
                    self.device.device().cmd_push_constants(
                        cmd,
                        pipeline.layout(),
                        vk::ShaderStageFlags::COMPUTE,
                        0,
                        constants,
                    );
                }
            }

            // Dispatch
            pipeline.dispatch(cmd, group_count_x, group_count_y, group_count_z);

            // Memory barrier to ensure writes are visible
            let barrier = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::TRANSFER_READ);

            unsafe {
                self.device.device().cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    &[barrier],
                    &[],
                    &[],
                );
            }

            Ok(())
        })
    }
}

impl Drop for ComputeContext {
    fn drop(&mut self) {
        unsafe {
            self.device.device_wait_idle().ok();
            self.device.destroy_fence(self.fence);
            self.device.free_command_buffers(&self.command_buffers);
        }
    }
}

/// Helper for computing dispatch dimensions
pub fn compute_dispatch_size(size: u32, workgroup_size: u32) -> u32 {
    (size + workgroup_size - 1) / workgroup_size
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dispatch_size_calculation() {
        assert_eq!(compute_dispatch_size(100, 16), 7);
        assert_eq!(compute_dispatch_size(256, 16), 16);
        assert_eq!(compute_dispatch_size(255, 16), 16);
        assert_eq!(compute_dispatch_size(1920, 16), 120);
    }
}
