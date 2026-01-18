use crate::device::VulkanDevice;
use crate::memory::{AllocatedBuffer, BufferAllocator, BufferUsage, MemoryLocationHint};
use crate::error::Result;
use ash::vk;
use std::sync::Arc;

/// High-level buffer interface for GPU operations
pub struct Buffer {
    device: Arc<VulkanDevice>,
    buffer: AllocatedBuffer,
}

impl Buffer {
    /// Create a new device-local storage buffer
    pub fn new_storage(
        allocator: &BufferAllocator,
        device: Arc<VulkanDevice>,
        size: u64,
    ) -> Result<Self> {
        let buffer = allocator.create_device_buffer(size, BufferUsage::STORAGE)?;
        Ok(Self { device, buffer })
    }

    /// Create a new uniform buffer
    pub fn new_uniform(
        allocator: &BufferAllocator,
        device: Arc<VulkanDevice>,
        size: u64,
    ) -> Result<Self> {
        let buffer = allocator.allocate(
            size,
            BufferUsage::UNIFORM | BufferUsage::TRANSFER_DST,
            MemoryLocationHint::CpuToGpu,
        )?;
        Ok(Self { device, buffer })
    }

    /// Get the underlying Vulkan buffer
    pub fn vk_buffer(&self) -> vk::Buffer {
        self.buffer.buffer()
    }

    /// Get buffer size
    pub fn size(&self) -> u64 {
        self.buffer.size()
    }

    /// Upload data to buffer using a staging buffer
    pub fn upload_data<T: Copy>(
        &mut self,
        allocator: &BufferAllocator,
        data: &[T],
        command_buffer: vk::CommandBuffer,
    ) -> Result<()> {
        let data_size = std::mem::size_of_val(data) as u64;
        let mut staging = allocator.create_staging_buffer(data_size)?;
        staging.write_data(data)?;

        // Record copy command
        unsafe {
            let copy_region = vk::BufferCopy::default()
                .src_offset(0)
                .dst_offset(0)
                .size(data_size);

            self.device.device().cmd_copy_buffer(
                command_buffer,
                staging.buffer(),
                self.buffer.buffer(),
                &[copy_region],
            );
        }

        Ok(())
    }

    /// Download data from buffer using a staging buffer
    pub fn download_data<T: Copy>(
        &self,
        allocator: &BufferAllocator,
        data: &mut [T],
        command_buffer: vk::CommandBuffer,
    ) -> Result<AllocatedBuffer> {
        let data_size = std::mem::size_of_val(data) as u64;
        let readback = allocator.create_readback_buffer(data_size)?;

        // Record copy command
        unsafe {
            let copy_region = vk::BufferCopy::default()
                .src_offset(0)
                .dst_offset(0)
                .size(data_size);

            self.device.device().cmd_copy_buffer(
                command_buffer,
                self.buffer.buffer(),
                readback.buffer(),
                &[copy_region],
            );
        }

        Ok(readback)
    }
}

/// Buffer view for descriptor sets
pub struct BufferView {
    buffer: vk::Buffer,
    offset: u64,
    range: u64,
}

impl BufferView {
    /// Create a new buffer view
    pub fn new(buffer: &Buffer, offset: u64, range: u64) -> Self {
        Self {
            buffer: buffer.vk_buffer(),
            offset,
            range,
        }
    }

    /// Create a full buffer view
    pub fn full(buffer: &Buffer) -> Self {
        Self {
            buffer: buffer.vk_buffer(),
            offset: 0,
            range: buffer.size(),
        }
    }

    /// Create a buffer view from an AllocatedBuffer
    pub fn from_allocated(buffer: &AllocatedBuffer) -> Self {
        Self {
            buffer: buffer.buffer(),
            offset: 0,
            range: buffer.size(),
        }
    }

    /// Get descriptor buffer info for binding
    pub fn descriptor_info(&self) -> vk::DescriptorBufferInfo {
        vk::DescriptorBufferInfo::default()
            .buffer(self.buffer)
            .offset(self.offset)
            .range(self.range)
    }
}
