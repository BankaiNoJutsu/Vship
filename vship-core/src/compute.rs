// Compute context for executing GPU operations

use crate::device::VulkanDevice;
use crate::memory::{AllocatedBuffer, BufferAllocator};
use crate::pipeline::ComputePipeline;
use crate::error::Result;
use ash::vk;
use std::sync::{Arc, atomic::{AtomicU64, Ordering}};

const COMMAND_BUFFER_COUNT: u32 = 4;

/// Compute context for GPU shader execution
pub struct ComputeContext {
    device: Arc<VulkanDevice>,
    allocator: BufferAllocator,
    command_buffers: Vec<vk::CommandBuffer>,
    fences: Vec<vk::Fence>,
    next_cmd: std::sync::atomic::AtomicUsize,
    timestamp_query_pool: Option<vk::QueryPool>,
    timestamp_period: f32,
    last_gpu_time_ns: AtomicU64,
    timeline_semaphore: Option<vk::Semaphore>,
    timeline_value: AtomicU64,
    staging_pool: std::sync::Mutex<Vec<AllocatedBuffer>>,
}

/// Batched command recorder for reducing submit/wait overhead
pub struct ComputeBatch<'a> {
    ctx: &'a ComputeContext,
    cmd: vk::CommandBuffer,
    cmd_index: usize,
    staging_buffers: Vec<AllocatedBuffer>,
}

impl ComputeContext {
    /// Create a new compute context
    pub fn new(device: Arc<VulkanDevice>, instance: &ash::Instance) -> Result<Self> {
        let allocator = BufferAllocator::new(
            Arc::clone(&device),
            instance,
            device.physical_device(),
        )?;

        let command_buffers = device.allocate_command_buffers(COMMAND_BUFFER_COUNT)?;
        let mut fences = Vec::with_capacity(command_buffers.len());
        for _ in 0..command_buffers.len() {
            fences.push(device.create_fence(true)?);
        }

        let timeline_semaphore = if device.supports_timeline_semaphore() {
            Some(device.create_timeline_semaphore(0)?)
        } else {
            None
        };

        let timestamp_period = device.timestamp_period();
        let timestamp_query_pool = if timestamp_period > 0.0 {
            let info = vk::QueryPoolCreateInfo::default()
                .query_type(vk::QueryType::TIMESTAMP)
                .query_count(2);
            Some(unsafe { device.device().create_query_pool(&info, None)? })
        } else {
            None
        };

        Ok(Self {
            device,
            allocator,
            command_buffers,
            fences,
            next_cmd: std::sync::atomic::AtomicUsize::new(0),
            timestamp_query_pool,
            timestamp_period,
            last_gpu_time_ns: AtomicU64::new(0),
            timeline_semaphore,
            timeline_value: AtomicU64::new(0),
            staging_pool: std::sync::Mutex::new(Vec::new()),
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

    fn acquire_staging_buffer(&self, size: u64) -> Result<AllocatedBuffer> {
        let mut pool = self.staging_pool.lock().unwrap();
        if let Some(idx) = pool.iter().position(|buf| buf.size() >= size) {
            return Ok(pool.swap_remove(idx));
        }
        drop(pool);
        self.allocator.create_staging_buffer(size)
    }

    fn reclaim_staging_buffers(&self, mut buffers: Vec<AllocatedBuffer>) {
        if buffers.is_empty() {
            return;
        }
        let mut pool = self.staging_pool.lock().unwrap();
        pool.append(&mut buffers);
    }

    /// Begin recording commands
    pub fn begin_commands(&self) -> Result<(vk::CommandBuffer, usize)> {
        let idx = self.next_cmd.fetch_add(1, Ordering::Relaxed) % self.command_buffers.len();
        let cmd = self.command_buffers[idx];

        self.device.wait_for_fence(self.fences[idx], 10_000_000_000)?;
        self.device.reset_fence(self.fences[idx])?;
        self.device.reset_command_buffer(cmd)?;

        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        unsafe {
            self.device.device().begin_command_buffer(cmd, &begin_info)?;
        }

        Ok((cmd, idx))
    }

    /// Begin a batched recording session
    pub fn begin_batch(&self) -> Result<ComputeBatch<'_>> {
        let (cmd, cmd_index) = self.begin_commands()?;
        if let Some(query_pool) = self.timestamp_query_pool {
            unsafe {
                self.device.device().cmd_reset_query_pool(cmd, query_pool, 0, 2);
                self.device.device().cmd_write_timestamp(
                    cmd,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    query_pool,
                    0,
                );
            }
        }
        Ok(ComputeBatch {
            ctx: self,
            cmd,
            cmd_index,
            staging_buffers: Vec::new(),
        })
    }

    /// End recording and submit commands
    pub fn submit_and_wait(&self, cmd: vk::CommandBuffer, cmd_index: usize) -> Result<()> {
        unsafe {
            self.device.device().end_command_buffer(cmd)?;
        }
        let fence = self.fences[cmd_index];

        // Submit
        if let Some(timeline_semaphore) = self.timeline_semaphore {
            let value = self.timeline_value.fetch_add(1, Ordering::Relaxed) + 1;
            self.device.submit_compute_with_timeline(
                &[cmd],
                timeline_semaphore,
                value,
                fence,
            )?;
            self.device.wait_timeline_semaphore(timeline_semaphore, value, 10_000_000_000)?;
        } else {
            self.device.submit_compute(&[cmd], &[], &[], fence)?;
            self.device.wait_for_fence(fence, 10_000_000_000)?;
        }

        Ok(())
    }

    /// Execute a simple compute operation
    pub fn execute<F>(&self, f: F) -> Result<()>
    where
        F: FnOnce(vk::CommandBuffer) -> Result<()>,
    {
        let (cmd, cmd_index) = self.begin_commands()?;
        f(cmd)?;
        self.submit_and_wait(cmd, cmd_index)?;
        Ok(())
    }

    /// Upload data to GPU buffer
    pub fn upload_buffer<T: Copy>(
        &self,
        data: &[T],
        dst: &AllocatedBuffer,
    ) -> Result<()> {
        let size = std::mem::size_of_val(data) as u64;
        let mut staging = self.acquire_staging_buffer(size)?;

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

        self.reclaim_staging_buffers(vec![staging]);
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

    /// Get last recorded GPU time for a batch in nanoseconds
    pub fn last_gpu_time_ns(&self) -> u64 {
        self.last_gpu_time_ns.load(Ordering::Relaxed)
    }
}

impl<'a> ComputeBatch<'a> {
    /// Record a buffer upload via staging
    pub fn record_upload_buffer<T: Copy>(
        &mut self,
        data: &[T],
        dst: &AllocatedBuffer,
    ) -> Result<()> {
        let size = std::mem::size_of_val(data) as u64;
        let mut staging = self.ctx.acquire_staging_buffer(size)?;
        staging.write_data(data)?;

        unsafe {
            let region = vk::BufferCopy::default()
                .src_offset(0)
                .dst_offset(0)
                .size(size);
            self.ctx.device.device().cmd_copy_buffer(
                self.cmd,
                staging.buffer(),
                dst.buffer(),
                &[region],
            );
        }

        self.staging_buffers.push(staging);
        Ok(())
    }

    /// Record a buffer download into a readback buffer
    pub fn record_download_buffer(
        &self,
        src: &AllocatedBuffer,
        size: u64,
    ) -> Result<AllocatedBuffer> {
        let readback = self.ctx.allocator.create_readback_buffer(size)?;

        unsafe {
            let region = vk::BufferCopy::default()
                .src_offset(0)
                .dst_offset(0)
                .size(size);
            self.ctx.device.device().cmd_copy_buffer(
                self.cmd,
                src.buffer(),
                readback.buffer(),
                &[region],
            );
        }

        Ok(readback)
    }

    /// Record a buffer-to-buffer copy with offsets
    pub fn record_copy_buffer(
        &self,
        src: &AllocatedBuffer,
        dst: &AllocatedBuffer,
        size: u64,
        src_offset: u64,
        dst_offset: u64,
    ) {
        unsafe {
            let region = vk::BufferCopy::default()
                .src_offset(src_offset)
                .dst_offset(dst_offset)
                .size(size);
            self.ctx.device.device().cmd_copy_buffer(
                self.cmd,
                src.buffer(),
                dst.buffer(),
                &[region],
            );
        }
    }

    /// Ensure transfer writes are visible to compute shaders
    pub fn record_transfer_to_compute_barrier(&self) {
        let barrier = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE | vk::AccessFlags::TRANSFER_READ)
            .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);

        unsafe {
            self.ctx.device.device().cmd_pipeline_barrier(
                self.cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[barrier],
                &[],
                &[],
            );
        }
    }

    /// Ensure transfer writes are visible to subsequent transfer reads
    pub fn record_transfer_to_transfer_barrier(&self) {
        let barrier = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::TRANSFER_READ);

        unsafe {
            self.ctx.device.device().cmd_pipeline_barrier(
                self.cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[barrier],
                &[],
                &[],
            );
        }
    }

    /// Record a compute dispatch
    pub fn record_dispatch_shader(
        &self,
        pipeline: &ComputePipeline,
        descriptor_set: vk::DescriptorSet,
        push_constants: Option<&[u8]>,
        group_count_x: u32,
        group_count_y: u32,
        group_count_z: u32,
    ) {
        pipeline.bind(self.cmd);
        pipeline.bind_descriptor_sets(self.cmd, &[descriptor_set]);

        if let Some(constants) = push_constants {
            unsafe {
                self.ctx.device.device().cmd_push_constants(
                    self.cmd,
                    pipeline.layout(),
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    constants,
                );
            }
        }

        pipeline.dispatch(self.cmd, group_count_x, group_count_y, group_count_z);

        let barrier = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::TRANSFER_READ);

        unsafe {
            self.ctx.device.device().cmd_pipeline_barrier(
                self.cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[barrier],
                &[],
                &[],
            );
        }
    }

    /// End recording, submit, and wait for completion
    pub fn finish_and_wait(self) -> Result<u64> {
        if let Some(query_pool) = self.ctx.timestamp_query_pool {
            unsafe {
                self.ctx.device.device().cmd_write_timestamp(
                    self.cmd,
                    vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                    query_pool,
                    1,
                );
            }
        }

        self.ctx.submit_and_wait(self.cmd, self.cmd_index)?;

        let mut gpu_time_ns = 0u64;
        if let Some(query_pool) = self.ctx.timestamp_query_pool {
            let mut timestamps = [0u64; 2];
            let result = unsafe {
                self.ctx.device.device().get_query_pool_results(
                    query_pool,
                    0,
                    &mut timestamps,
                    vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
                )
            };
            if result.is_ok() {
                let ticks = timestamps[1].saturating_sub(timestamps[0]);
                gpu_time_ns = (ticks as f64 * self.ctx.timestamp_period as f64) as u64;
            }
        }

        self.ctx.last_gpu_time_ns.store(gpu_time_ns, Ordering::Relaxed);
        self.ctx.reclaim_staging_buffers(self.staging_buffers);
        Ok(gpu_time_ns)
    }
}

impl Drop for ComputeContext {
    fn drop(&mut self) {
        unsafe {
            self.device.wait_idle().ok();
            for fence in &self.fences {
                self.device.destroy_fence(*fence);
            }
            self.device.free_command_buffers(&self.command_buffers);
            if let Some(query_pool) = self.timestamp_query_pool {
                self.device.device().destroy_query_pool(query_pool, None);
            }
            if let Some(semaphore) = self.timeline_semaphore {
                self.device.device().destroy_semaphore(semaphore, None);
            }
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
