use crate::device::VulkanDevice;
use crate::error::{Result, VshipError};
use ash::vk;
use gpu_allocator::vulkan::{Allocator, AllocatorCreateDesc, Allocation, AllocationCreateDesc, AllocationScheme};
use gpu_allocator::MemoryLocation;
use std::sync::{Arc, Mutex};

bitflags::bitflags! {
    /// Buffer usage flags
    pub struct BufferUsage: u32 {
        const TRANSFER_SRC = 0b0001;
        const TRANSFER_DST = 0b0010;
        const STORAGE = 0b0100;
        const UNIFORM = 0b1000;
    }
}

impl BufferUsage {
    pub fn to_vk(&self) -> vk::BufferUsageFlags {
        let mut flags = vk::BufferUsageFlags::empty();
        if self.contains(BufferUsage::TRANSFER_SRC) {
            flags |= vk::BufferUsageFlags::TRANSFER_SRC;
        }
        if self.contains(BufferUsage::TRANSFER_DST) {
            flags |= vk::BufferUsageFlags::TRANSFER_DST;
        }
        if self.contains(BufferUsage::STORAGE) {
            flags |= vk::BufferUsageFlags::STORAGE_BUFFER;
        }
        if self.contains(BufferUsage::UNIFORM) {
            flags |= vk::BufferUsageFlags::UNIFORM_BUFFER;
        }
        flags
    }
}

/// Memory location hint for buffer allocation
#[derive(Debug, Clone, Copy)]
pub enum MemoryLocationHint {
    /// GPU-only memory (fastest for GPU operations)
    GpuOnly,
    /// CPU-to-GPU transfer (upload)
    CpuToGpu,
    /// GPU-to-CPU transfer (download)
    GpuToCpu,
    /// Unknown (let allocator decide)
    Unknown,
}

impl MemoryLocationHint {
    fn to_gpu_allocator(&self) -> MemoryLocation {
        match self {
            MemoryLocationHint::GpuOnly => MemoryLocation::GpuOnly,
            MemoryLocationHint::CpuToGpu => MemoryLocation::CpuToGpu,
            MemoryLocationHint::GpuToCpu => MemoryLocation::GpuToCpu,
            MemoryLocationHint::Unknown => MemoryLocation::Unknown,
        }
    }
}

/// Buffer allocator using gpu-allocator
pub struct BufferAllocator {
    device: Arc<VulkanDevice>,
    allocator: Arc<Mutex<Allocator>>,
}

impl BufferAllocator {
    /// Create a new buffer allocator
    pub fn new(
        device: Arc<VulkanDevice>,
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
    ) -> Result<Self> {
        let allocator = Allocator::new(&AllocatorCreateDesc {
            instance: instance.clone(),
            device: device.device().clone(),
            physical_device,
            debug_settings: Default::default(),
            buffer_device_address: false,
            allocation_sizes: Default::default(),
        })?;

        Ok(Self {
            device,
            allocator: Arc::new(Mutex::new(allocator)),
        })
    }

    /// Allocate a buffer with specified size and usage
    pub fn allocate(
        &self,
        size: u64,
        usage: BufferUsage,
        location: MemoryLocationHint,
    ) -> Result<AllocatedBuffer> {
        let buffer_info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(usage.to_vk())
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let buffer = unsafe { self.device.device().create_buffer(&buffer_info, None)? };

        let requirements = unsafe { self.device.device().get_buffer_memory_requirements(buffer) };

        let allocation = self.allocator.lock().unwrap().allocate(&AllocationCreateDesc {
            name: "vship_buffer",
            requirements,
            location: location.to_gpu_allocator(),
            linear: true,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        })?;

        unsafe {
            self.device
                .device()
                .bind_buffer_memory(buffer, allocation.memory(), allocation.offset())?;
        }

        Ok(AllocatedBuffer {
            buffer,
            allocation: Some(allocation),
            size,
            device: Arc::clone(&self.device),
            allocator: Arc::clone(&self.allocator),
        })
    }

    /// Create a staging buffer for CPU-to-GPU transfers
    pub fn create_staging_buffer(&self, size: u64) -> Result<AllocatedBuffer> {
        self.allocate(
            size,
            BufferUsage::TRANSFER_SRC,
            MemoryLocationHint::CpuToGpu,
        )
    }

    /// Create a device-local buffer for GPU operations
    pub fn create_device_buffer(&self, size: u64, usage: BufferUsage) -> Result<AllocatedBuffer> {
        self.allocate(
            size,
            usage | BufferUsage::TRANSFER_DST | BufferUsage::TRANSFER_SRC,
            MemoryLocationHint::GpuOnly,
        )
    }

    /// Create a readback buffer for GPU-to-CPU transfers
    pub fn create_readback_buffer(&self, size: u64) -> Result<AllocatedBuffer> {
        self.allocate(
            size,
            BufferUsage::TRANSFER_DST,
            MemoryLocationHint::GpuToCpu,
        )
    }
}

/// An allocated buffer with associated memory
pub struct AllocatedBuffer {
    buffer: vk::Buffer,
    allocation: Option<Allocation>,
    size: u64,
    device: Arc<VulkanDevice>,
    allocator: Arc<Mutex<Allocator>>,
}

impl AllocatedBuffer {
    /// Get the Vulkan buffer handle
    pub fn buffer(&self) -> vk::Buffer {
        self.buffer
    }

    /// Get the buffer size
    pub fn size(&self) -> u64 {
        self.size
    }

    /// Map buffer memory for CPU access
    pub fn map(&mut self) -> Result<*mut u8> {
        if let Some(ref mut allocation) = self.allocation {
            allocation
                .mapped_ptr()
                .map(|ptr| ptr.as_ptr() as *mut u8)
                .ok_or_else(|| VshipError::AllocationError("Buffer not mappable".to_string()))
        } else {
            Err(VshipError::AllocationError("No allocation".to_string()))
        }
    }

    /// Write data to buffer (requires mappable memory)
    pub fn write_data<T: Copy>(&mut self, data: &[T]) -> Result<()> {
        let ptr = self.map()?;
        let size = std::mem::size_of_val(data);

        if size > self.size as usize {
            return Err(VshipError::InvalidBufferSize {
                expected: self.size as usize,
                actual: size,
            });
        }

        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr() as *const u8, ptr, size);
        }

        Ok(())
    }

    /// Read data from buffer (requires mappable memory)
    pub fn read_data<T: Copy>(&mut self, data: &mut [T]) -> Result<()> {
        let ptr = self.map()?;
        let size = std::mem::size_of_val(data);

        if size > self.size as usize {
            return Err(VshipError::InvalidBufferSize {
                expected: self.size as usize,
                actual: size,
            });
        }

        unsafe {
            std::ptr::copy_nonoverlapping(ptr, data.as_mut_ptr() as *mut u8, size);
        }

        Ok(())
    }
}

impl Drop for AllocatedBuffer {
    fn drop(&mut self) {
        unsafe {
            self.device.device().destroy_buffer(self.buffer, None);
        }
        if let Some(allocation) = self.allocation.take() {
            self.allocator.lock().unwrap().free(allocation).ok();
        }
    }
}
