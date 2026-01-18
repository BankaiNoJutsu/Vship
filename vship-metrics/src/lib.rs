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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_context_creation() {
        let ctx = MetricsContext::new();
        assert!(ctx.is_ok());
    }
}
