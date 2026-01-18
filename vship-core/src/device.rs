use crate::error::{Result, VshipError};
use ash::vk;
use std::ffi::CStr;
use std::sync::{Arc, Mutex};

/// Vulkan instance wrapper with reference-counted lifetime
///
/// This ensures the Vulkan instance is only destroyed after all devices
/// that depend on it have been dropped.
pub struct VulkanInstance {
    #[allow(dead_code)]
    entry: ash::Entry,
    instance: ash::Instance,
}

impl VulkanInstance {
    /// Create a new Vulkan instance
    pub fn new() -> Result<Self> {
        let entry = unsafe { ash::Entry::load()? };

        let app_info = vk::ApplicationInfo::default()
            .application_name(c"Vship")
            .application_version(vk::make_api_version(0, 4, 1, 0))
            .engine_name(c"Vship")
            .engine_version(vk::make_api_version(0, 4, 1, 0))
            .api_version(vk::API_VERSION_1_3);

        let create_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info);

        let instance = unsafe { entry.create_instance(&create_info, None)? };

        Ok(Self { entry, instance })
    }

    /// Get the raw Vulkan instance
    pub fn instance(&self) -> &ash::Instance {
        &self.instance
    }

    /// Enumerate physical devices
    pub fn enumerate_physical_devices(&self) -> Result<Vec<vk::PhysicalDevice>> {
        let devices = unsafe { self.instance.enumerate_physical_devices()? };
        Ok(devices)
    }
}

impl Drop for VulkanInstance {
    fn drop(&mut self) {
        unsafe {
            self.instance.destroy_instance(None);
        }
    }
}

/// Vulkan device wrapper with compute queue support
pub struct VulkanDevice {
    // Hold a reference to the instance to keep it alive
    _instance: Arc<VulkanInstance>,
    physical_device: vk::PhysicalDevice,
    device: ash::Device,
    compute_queue_family: u32,
    compute_queue: Mutex<vk::Queue>,
    properties: vk::PhysicalDeviceProperties,
    memory_properties: vk::PhysicalDeviceMemoryProperties,
    command_pool: Mutex<vk::CommandPool>,
}

impl VulkanDevice {
    /// Create a new Vulkan device from a physical device
    pub fn new(vulkan_instance: Arc<VulkanInstance>, physical_device: vk::PhysicalDevice) -> Result<Self> {
        let instance = vulkan_instance.instance();
        let properties = unsafe { instance.get_physical_device_properties(physical_device) };
        let memory_properties = unsafe {
            instance.get_physical_device_memory_properties(physical_device)
        };

        // Find compute queue family
        let queue_families = unsafe {
            instance.get_physical_device_queue_family_properties(physical_device)
        };

        let compute_queue_family = queue_families
            .iter()
            .enumerate()
            .find(|(_, props)| props.queue_flags.contains(vk::QueueFlags::COMPUTE))
            .map(|(index, _)| index as u32)
            .ok_or(VshipError::NoComputeQueue)?;

        // Create logical device
        let queue_priorities = [1.0];
        let queue_create_info = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(compute_queue_family)
            .queue_priorities(&queue_priorities);

        let device_create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(std::slice::from_ref(&queue_create_info));

        let device = unsafe { instance.create_device(physical_device, &device_create_info, None)? };

        let compute_queue = unsafe { device.get_device_queue(compute_queue_family, 0) };

        // Create command pool
        let pool_create_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(compute_queue_family)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);

        let command_pool = unsafe { device.create_command_pool(&pool_create_info, None)? };

        let device_name = unsafe { CStr::from_ptr(properties.device_name.as_ptr()) }
            .to_string_lossy()
            .to_string();

        log::info!(
            "Created Vulkan device: {} (type: {:?})",
            device_name,
            properties.device_type
        );

        Ok(Self {
            _instance: vulkan_instance,
            physical_device,
            device,
            compute_queue_family,
            compute_queue: Mutex::new(compute_queue),
            properties,
            memory_properties,
            command_pool: Mutex::new(command_pool),
        })
    }

    /// Get the physical device
    pub fn physical_device(&self) -> vk::PhysicalDevice {
        self.physical_device
    }

    /// Get the logical device
    pub fn device(&self) -> &ash::Device {
        &self.device
    }

    /// Get the compute queue family index
    pub fn compute_queue_family(&self) -> u32 {
        self.compute_queue_family
    }

    /// Get device properties
    pub fn properties(&self) -> &vk::PhysicalDeviceProperties {
        &self.properties
    }

    /// Get memory properties
    pub fn memory_properties(&self) -> &vk::PhysicalDeviceMemoryProperties {
        &self.memory_properties
    }

    /// Get device name
    pub fn name(&self) -> String {
        unsafe { CStr::from_ptr(self.properties.device_name.as_ptr()) }
            .to_string_lossy()
            .to_string()
    }

    /// Submit commands to compute queue
    pub fn submit_compute(
        &self,
        command_buffers: &[vk::CommandBuffer],
        wait_semaphores: &[vk::Semaphore],
        signal_semaphores: &[vk::Semaphore],
        fence: vk::Fence,
    ) -> Result<()> {
        let queue = self.compute_queue.lock().unwrap();

        let wait_stages = vec![vk::PipelineStageFlags::COMPUTE_SHADER; wait_semaphores.len()];

        let submit_info = vk::SubmitInfo::default()
            .command_buffers(command_buffers)
            .wait_semaphores(wait_semaphores)
            .wait_dst_stage_mask(&wait_stages)
            .signal_semaphores(signal_semaphores);

        unsafe {
            self.device.queue_submit(*queue, &[submit_info], fence)?;
        }

        Ok(())
    }

    /// Wait for device to become idle
    pub fn wait_idle(&self) -> Result<()> {
        unsafe { self.device.device_wait_idle()? };
        Ok(())
    }

    /// Allocate command buffers
    pub fn allocate_command_buffers(&self, count: u32) -> Result<Vec<vk::CommandBuffer>> {
        let command_pool = self.command_pool.lock().unwrap();

        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(*command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(count);

        let buffers = unsafe { self.device.allocate_command_buffers(&alloc_info)? };
        Ok(buffers)
    }

    /// Free command buffers
    pub fn free_command_buffers(&self, command_buffers: &[vk::CommandBuffer]) {
        let command_pool = self.command_pool.lock().unwrap();
        unsafe {
            self.device.free_command_buffers(*command_pool, command_buffers);
        }
    }

    /// Create fence
    pub fn create_fence(&self, signaled: bool) -> Result<vk::Fence> {
        let flags = if signaled {
            vk::FenceCreateFlags::SIGNALED
        } else {
            vk::FenceCreateFlags::empty()
        };

        let create_info = vk::FenceCreateInfo::default().flags(flags);
        let fence = unsafe { self.device.create_fence(&create_info, None)? };
        Ok(fence)
    }

    /// Wait for fence
    pub fn wait_for_fence(&self, fence: vk::Fence, timeout: u64) -> Result<()> {
        unsafe { self.device.wait_for_fences(&[fence], true, timeout)? };
        Ok(())
    }

    /// Reset fence
    pub fn reset_fence(&self, fence: vk::Fence) -> Result<()> {
        unsafe { self.device.reset_fences(&[fence])? };
        Ok(())
    }

    /// Destroy fence
    pub fn destroy_fence(&self, fence: vk::Fence) {
        unsafe { self.device.destroy_fence(fence, None) };
    }
}

impl Drop for VulkanDevice {
    fn drop(&mut self) {
        unsafe {
            self.device.device_wait_idle().ok();
            let command_pool = *self.command_pool.lock().unwrap();
            self.device.destroy_command_pool(command_pool, None);
            self.device.destroy_device(None);
        }
    }
}

/// Device selector for choosing the best GPU
pub struct DeviceSelector;

impl DeviceSelector {
    /// Select the best device from available devices
    /// Prioritizes discrete GPUs over integrated ones
    pub fn select_best(devices: &[Arc<VulkanDevice>]) -> Arc<VulkanDevice> {
        devices
            .iter()
            .max_by_key(|d| {
                let props = d.properties();
                match props.device_type {
                    vk::PhysicalDeviceType::DISCRETE_GPU => 3,
                    vk::PhysicalDeviceType::INTEGRATED_GPU => 2,
                    vk::PhysicalDeviceType::VIRTUAL_GPU => 1,
                    _ => 0,
                }
            })
            .expect("No devices available")
            .clone()
    }
}
