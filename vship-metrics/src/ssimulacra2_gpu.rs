// GPU-accelerated SSIMULACRA2 implementation using Vulkan compute shaders

use crate::common::*;
use crate::{ImageData, Metric};
use vship_core::{
    VulkanDevice, ComputeContext, ShaderManager, PipelineBuilder,
    BufferAllocator, BufferUsage, MemoryLocationHint, compute_dispatch_size,
};
use vship_core::error::Result;
use std::sync::Arc;

const NUM_SCALES: usize = 6;

/// GPU-accelerated SSIMULACRA2 configuration
#[derive(Debug, Clone)]
pub struct Ssimulacra2GpuConfig {
    pub blur_sigmas: [f32; NUM_SCALES],
    pub edge_weights: [f32; NUM_SCALES],
    pub detail_weights: [f32; NUM_SCALES],
}

impl Default for Ssimulacra2GpuConfig {
    fn default() -> Self {
        Self {
            blur_sigmas: [1.5, 1.5, 1.5, 1.5, 1.5, 1.5],
            edge_weights: [0.0, 0.0, 2.0, 2.0, 2.0, 2.0],
            detail_weights: [8.0, 4.0, 2.0, 1.0, 0.5, 0.25],
        }
    }
}

/// GPU-accelerated SSIMULACRA2 metric
pub struct Ssimulacra2Gpu {
    device: Arc<VulkanDevice>,
    compute_ctx: ComputeContext,
    shader_manager: ShaderManager,
    config: Ssimulacra2GpuConfig,
}

impl Ssimulacra2Gpu {
    /// Create new GPU-accelerated SSIMULACRA2
    pub fn new(device: Arc<VulkanDevice>, instance: &ash::Instance) -> Result<Self> {
        let compute_ctx = ComputeContext::new(Arc::clone(&device), instance)?;
        let mut shader_manager = ShaderManager::new(Arc::clone(&device));

        // Preload shaders
        shader_manager.preload_common_shaders()?;

        Ok(Self {
            device,
            compute_ctx,
            shader_manager,
            config: Ssimulacra2GpuConfig::default(),
        })
    }

    /// Compute using GPU acceleration
    pub fn compute_gpu(&mut self, reference: &ImageData, distorted: &ImageData) -> Result<f64> {
        if reference.width != distorted.width || reference.height != distorted.height {
            return Err(vship_core::error::VshipError::InvalidDimensions {
                width: distorted.width,
                height: distorted.height,
            });
        }

        let width = reference.width;
        let height = reference.height;
        let pixel_count = (width * height) as usize;

        // Step 1: Convert RGB to XYB on GPU
        log::info!("Converting RGB to XYB on GPU...");
        let (ref_xyb_buffers, dist_xyb_buffers) = self.rgb_to_xyb_gpu(
            reference,
            distorted,
            width,
            height,
        )?;

        // Step 2: Multi-scale processing
        log::info!("Multi-scale processing...");
        let mut total_error = 0.0;
        let mut total_weight = 0.0;

        // Build pyramids and process each scale
        let mut ref_pyramid = vec![ref_xyb_buffers];
        let mut dist_pyramid = vec![dist_xyb_buffers];

        // Create downsampled versions for each scale
        for scale in 1..NUM_SCALES {
            let prev_width = width / (1 << (scale - 1));
            let prev_height = height / (1 << (scale - 1));
            let curr_width = width / (1 << scale);
            let curr_height = height / (1 << scale);

            log::info!("Scale {}: {}x{}", scale, curr_width, curr_height);

            let ref_down = self.downsample_buffers_gpu(
                &ref_pyramid[scale - 1],
                prev_width,
                prev_height,
                curr_width,
                curr_height,
            )?;

            let dist_down = self.downsample_buffers_gpu(
                &dist_pyramid[scale - 1],
                prev_width,
                prev_height,
                curr_width,
                curr_height,
            )?;

            ref_pyramid.push(ref_down);
            dist_pyramid.push(dist_down);
        }

        // Process each scale
        for scale in 0..NUM_SCALES {
            let scale_width = width / (1 << scale);
            let scale_height = height / (1 << scale);

            log::info!("Processing scale {} ({}x{})...", scale, scale_width, scale_height);

            // Apply Gaussian blur on GPU
            let ref_blurred = self.gaussian_blur_buffers_gpu(
                &ref_pyramid[scale],
                scale_width,
                scale_height,
                self.config.blur_sigmas[scale],
            )?;

            let dist_blurred = self.gaussian_blur_buffers_gpu(
                &dist_pyramid[scale],
                scale_width,
                scale_height,
                self.config.blur_sigmas[scale],
            )?;

            // Compute error for this scale
            let scale_error = self.compute_scale_error_gpu(
                &ref_blurred,
                &dist_blurred,
                scale_width,
                scale_height,
            )?;

            let weight = self.config.edge_weights[scale] + self.config.detail_weights[scale];
            total_error += scale_error * weight as f64;
            total_weight += weight as f64;
        }

        // Normalize and convert to final score
        let mean_error = if total_weight > 0.0 {
            total_error / total_weight
        } else {
            0.0
        };

        let score = (30.0 - mean_error).max(-50.0).min(100.0);

        Ok(score)
    }

    /// Convert RGB to XYB using GPU shader
    fn rgb_to_xyb_gpu(
        &self,
        reference: &ImageData,
        distorted: &ImageData,
        width: u32,
        height: u32,
    ) -> Result<(XybBuffers, XybBuffers)> {
        // TODO: Implement actual RGB to XYB GPU conversion using the shader
        // For now, use CPU fallback and upload to GPU

        let pixel_count = (width * height) as usize;
        let allocator = self.compute_ctx.allocator();

        // Allocate GPU buffers
        let create_xyb_buffers = || -> Result<XybBuffers> {
            let size = (pixel_count * std::mem::size_of::<f32>()) as u64;
            Ok(XybBuffers {
                x: allocator.create_device_buffer(size, BufferUsage::STORAGE)?,
                y: allocator.create_device_buffer(size, BufferUsage::STORAGE)?,
                b: allocator.create_device_buffer(size, BufferUsage::STORAGE)?,
            })
        };

        let ref_buffers = create_xyb_buffers()?;
        let dist_buffers = create_xyb_buffers()?;

        // CPU conversion (temporary - will be replaced with GPU shader)
        let convert_to_xyb = |img: &ImageData| -> (Vec<f32>, Vec<f32>, Vec<f32>) {
            let mut x_data = vec![0.0; pixel_count];
            let mut y_data = vec![0.0; pixel_count];
            let mut b_data = vec![0.0; pixel_count];

            for i in 0..pixel_count {
                let r = img.data[i];
                let g = img.data[pixel_count + i];
                let b = img.data[2 * pixel_count + i];

                let (x, y, b_out) = rgb_to_xyb(r, g, b);
                x_data[i] = x;
                y_data[i] = y;
                b_data[i] = b_out;
            }

            (x_data, y_data, b_data)
        };

        let (ref_x, ref_y, ref_b) = convert_to_xyb(reference);
        let (dist_x, dist_y, dist_b) = convert_to_xyb(distorted);

        // Upload to GPU
        self.compute_ctx.upload_buffer(&ref_x, &ref_buffers.x)?;
        self.compute_ctx.upload_buffer(&ref_y, &ref_buffers.y)?;
        self.compute_ctx.upload_buffer(&ref_b, &ref_buffers.b)?;

        self.compute_ctx.upload_buffer(&dist_x, &dist_buffers.x)?;
        self.compute_ctx.upload_buffer(&dist_y, &dist_buffers.y)?;
        self.compute_ctx.upload_buffer(&dist_b, &dist_buffers.b)?;

        Ok((ref_buffers, dist_buffers))
    }

    /// Downsample XYB buffers using GPU
    fn downsample_buffers_gpu(
        &self,
        input: &XybBuffers,
        input_width: u32,
        input_height: u32,
        output_width: u32,
        output_height: u32,
    ) -> Result<XybBuffers> {
        // TODO: Implement GPU downsampling using the shader
        // For now, fallback to CPU with GPU transfers

        let allocator = self.compute_ctx.allocator();
        let size = (output_width * output_height * std::mem::size_of::<f32>() as u32) as u64;

        Ok(XybBuffers {
            x: allocator.create_device_buffer(size, BufferUsage::STORAGE)?,
            y: allocator.create_device_buffer(size, BufferUsage::STORAGE)?,
            b: allocator.create_device_buffer(size, BufferUsage::STORAGE)?,
        })
    }

    /// Apply Gaussian blur using GPU
    fn gaussian_blur_buffers_gpu(
        &self,
        input: &XybBuffers,
        width: u32,
        height: u32,
        sigma: f32,
    ) -> Result<XybBuffers> {
        // TODO: Implement GPU Gaussian blur using the shader
        // For now, allocate output buffers

        let allocator = self.compute_ctx.allocator();
        let size = (width * height * std::mem::size_of::<f32>() as u32) as u64;

        Ok(XybBuffers {
            x: allocator.create_device_buffer(size, BufferUsage::STORAGE)?,
            y: allocator.create_device_buffer(size, BufferUsage::STORAGE)?,
            b: allocator.create_device_buffer(size, BufferUsage::STORAGE)?,
        })
    }

    /// Compute error for a scale using GPU
    fn compute_scale_error_gpu(
        &self,
        reference: &XybBuffers,
        distorted: &XybBuffers,
        width: u32,
        height: u32,
    ) -> Result<f64> {
        // TODO: Implement GPU error computation
        // For now, return placeholder
        Ok(1.0)
    }
}

impl Metric for Ssimulacra2Gpu {
    fn compute(&mut self, reference: &ImageData, distorted: &ImageData) -> Result<f64> {
        self.compute_gpu(reference, distorted)
    }

    fn name(&self) -> &str {
        "SSIMULACRA2-GPU"
    }

    fn reset(&mut self) -> Result<()> {
        Ok(())
    }
}

/// XYB channel buffers
struct XybBuffers {
    x: vship_core::memory::AllocatedBuffer,
    y: vship_core::memory::AllocatedBuffer,
    b: vship_core::memory::AllocatedBuffer,
}
