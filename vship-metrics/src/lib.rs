// Vship Metrics - Psychovisual image and video quality metrics
// Implements SSIMULACRA2, Butteraugli, and CVVDP

pub mod ssimulacra2;
pub mod ssimulacra2_gpu;
pub mod butteraugli;
pub mod cvvdp;
pub mod common;

#[cfg(test)]
mod tests;

pub use ssimulacra2::Ssimulacra2;
pub use ssimulacra2_gpu::Ssimulacra2Gpu;
pub use butteraugli::Butteraugli;
pub use cvvdp::Cvvdp;

/// GPU compute mode options for metrics that support them
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComputeMode {
    /// Single command buffer per frame (higher GPU utilization)
    SingleBatch,
    /// Per-step batches with waits (baseline behavior)
    LegacyBatched,
}

/// Reduction mode for aggregations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReduceMode {
    /// Reduce on GPU and read back a single value
    Gpu,
    /// Read back full buffer and reduce on CPU (debug/compare)
    Cpu,
}

use vship_core::error::Result;
use vship_core::VshipContext;
use std::sync::Arc;

/// Metric trait for all psychovisual metrics
pub trait Metric {
    /// Compute metric score between reference and distorted images
    fn compute(&mut self, reference: &ImageData, distorted: &ImageData) -> Result<f64>;

    /// Get metric name
    fn name(&self) -> &str;

    /// Reset metric state (useful for video sequences)
    fn reset(&mut self) -> Result<()>;

    /// Last GPU time for a frame in nanoseconds (if applicable)
    fn gpu_time_ns(&self) -> Option<u64> {
        None
    }

    /// Set compute mode (if supported by the metric)
    fn set_compute_mode(&mut self, _mode: ComputeMode) {}

    /// Set reduction mode (if supported by the metric)
    fn set_reduce_mode(&mut self, _mode: ReduceMode) {}

    /// Compute metric from packed RGBA8 inputs (default converts to f32 RGB)
    fn compute_rgba8(
        &mut self,
        reference: &ImageDataRgba8,
        distorted: &ImageDataRgba8,
    ) -> Result<f64> {
        let ref_f32 = ImageData::from_rgba8(
            reference.width,
            reference.height,
            &reference.data,
        )?;
        let dist_f32 = ImageData::from_rgba8(
            distorted.width,
            distorted.height,
            &distorted.data,
        )?;
        self.compute(&ref_f32, &dist_f32)
    }
}

/// Image data container
#[derive(Clone)]
pub struct ImageData {
    pub width: u32,
    pub height: u32,
    pub data: Vec<f32>, // Planar RGB or YUV data
    pub format: ImageFormat,
}

/// Image format
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageFormat {
    RGB,
    YUV420,
    YUV444,
}

impl ImageData {
    /// Create new image data
    pub fn new(width: u32, height: u32, format: ImageFormat) -> Self {
        let size = match format {
            ImageFormat::RGB | ImageFormat::YUV444 => (width * height * 3) as usize,
            ImageFormat::YUV420 => {
                let y_size = (width * height) as usize;
                let uv_size = ((width / 2) * (height / 2)) as usize;
                y_size + 2 * uv_size
            }
        };

        Self {
            width,
            height,
            data: vec![0.0; size],
            format,
        }
    }

    /// Create from f32 slice
    pub fn from_f32(
        width: u32,
        height: u32,
        data: &[f32],
        format: ImageFormat,
    ) -> Result<Self> {
        let mut img = Self::new(width, height, format);
        img.data.copy_from_slice(data);
        Ok(img)
    }

    /// Convert to linear RGB (if not already)
    pub fn to_linear_rgb(&self) -> Self {
        // For now, assume data is already in linear RGB
        // TODO: Implement YUV to RGB conversion
        self.clone()
    }

    /// Convert RGBA8 packed data to planar f32 RGB
    pub fn from_rgba8(width: u32, height: u32, data: &[u8]) -> Result<Self> {
        let pixel_count = (width * height) as usize;
        let expected_len = pixel_count * 4;
        if data.len() != expected_len {
            return Err(vship_core::error::VshipError::InvalidBufferSize {
                expected: expected_len,
                actual: data.len(),
            });
        }

        let mut image = Self::new(width, height, ImageFormat::RGB);
        for i in 0..pixel_count {
            let src = i * 4;
            image.data[i] = data[src] as f32 / 255.0;
            image.data[pixel_count + i] = data[src + 1] as f32 / 255.0;
            image.data[2 * pixel_count + i] = data[src + 2] as f32 / 255.0;
        }
        Ok(image)
    }
}

/// Packed RGBA8 image data (interleaved RGBA bytes)
#[derive(Clone)]
pub struct ImageDataRgba8 {
    pub width: u32,
    pub height: u32,
    pub data: Vec<u8>,
}

impl ImageDataRgba8 {
    /// Create a new RGBA8 buffer filled with zeros
    pub fn new(width: u32, height: u32) -> Self {
        let len = (width * height * 4) as usize;
        Self {
            width,
            height,
            data: vec![0u8; len],
        }
    }
}

/// Vship metrics context
pub struct MetricsContext {
    vship_ctx: Arc<VshipContext>,
}

impl MetricsContext {
    /// Create new metrics context
    pub fn new() -> Result<Self> {
        let vship_ctx = Arc::new(VshipContext::new()?);
        Ok(Self { vship_ctx })
    }

    /// Create SSIMULACRA2 metric (GPU-accelerated by default)
    pub fn create_ssimulacra2(&self) -> Result<Ssimulacra2Gpu> {
        Ssimulacra2Gpu::new(self.vship_ctx.default_device(), self.vship_ctx.instance())
    }

    /// Create SSIMULACRA2 metric (CPU fallback)
    pub fn create_ssimulacra2_cpu(&self) -> Result<Ssimulacra2> {
        Ssimulacra2::new(self.vship_ctx.default_device())
    }

    /// Create Butteraugli metric
    pub fn create_butteraugli(&self) -> Result<Butteraugli> {
        Butteraugli::new(self.vship_ctx.default_device())
    }

    /// Create CVVDP metric
    pub fn create_cvvdp(&self) -> Result<Cvvdp> {
        Cvvdp::new(self.vship_ctx.default_device())
    }
}

