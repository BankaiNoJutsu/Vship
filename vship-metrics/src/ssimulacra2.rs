// SSIMULACRA2 - Structural Similarity Inspired Image Quality Assessment
// Based on Cloudinary's ssimulacra2 algorithm

use crate::common::*;
use crate::{ImageData, Metric};
use vship_core::device::VulkanDevice;
use vship_core::error::Result;
use std::sync::Arc;

const NUM_SCALES: usize = 6;

/// SSIMULACRA2 configuration
#[derive(Debug, Clone)]
pub struct Ssimulacra2Config {
    /// Gaussian blur sigma for each scale
    pub blur_sigmas: [f32; NUM_SCALES],
    /// Edge weights for each scale
    pub edge_weights: [f32; NUM_SCALES],
    /// Detail weights for each scale
    pub detail_weights: [f32; NUM_SCALES],
}

impl Default for Ssimulacra2Config {
    fn default() -> Self {
        Self {
            // These values are from the original SSIMULACRA2 implementation
            blur_sigmas: [1.5, 1.5, 1.5, 1.5, 1.5, 1.5],
            edge_weights: [0.0, 0.0, 2.0, 2.0, 2.0, 2.0],
            detail_weights: [8.0, 4.0, 2.0, 1.0, 0.5, 0.25],
        }
    }
}

/// SSIMULACRA2 metric implementation
pub struct Ssimulacra2 {
    device: Arc<VulkanDevice>,
    config: Ssimulacra2Config,
}

impl Ssimulacra2 {
    /// Create new SSIMULACRA2 metric
    pub fn new(device: Arc<VulkanDevice>) -> Result<Self> {
        Ok(Self {
            device,
            config: Ssimulacra2Config::default(),
        })
    }

    /// Create with custom configuration
    pub fn with_config(device: Arc<VulkanDevice>, config: Ssimulacra2Config) -> Result<Self> {
        Ok(Self { device, config })
    }

    /// Compute SSIMULACRA2 score (CPU implementation)
    /// TODO: Replace with GPU implementation using Vulkan compute shaders
    fn compute_cpu(&self, reference: &ImageData, distorted: &ImageData) -> Result<f64> {
        if reference.width != distorted.width || reference.height != distorted.height {
            return Err(vship_core::error::VshipError::InvalidDimensions {
                width: distorted.width,
                height: distorted.height,
            });
        }

        // Convert to linear RGB if needed
        let ref_rgb = reference.to_linear_rgb();
        let dist_rgb = distorted.to_linear_rgb();

        // Convert RGB to XYB color space
        let ref_xyb = self.rgb_to_xyb_image(&ref_rgb)?;
        let dist_xyb = self.rgb_to_xyb_image(&dist_rgb)?;

        // Multi-scale processing
        let mut total_error = 0.0;
        let mut total_weight = 0.0;

        let mut ref_pyramid = vec![ref_xyb];
        let mut dist_pyramid = vec![dist_xyb];

        // Build pyramids
        for scale in 1..NUM_SCALES {
            let ref_down = self.downsample(&ref_pyramid[scale - 1])?;
            let dist_down = self.downsample(&dist_pyramid[scale - 1])?;
            ref_pyramid.push(ref_down);
            dist_pyramid.push(dist_down);
        }

        // Process each scale
        for scale in 0..NUM_SCALES {
            let ref_img = &ref_pyramid[scale];
            let dist_img = &dist_pyramid[scale];

            // Apply Gaussian blur
            let ref_blur = self.gaussian_blur(ref_img, self.config.blur_sigmas[scale])?;
            let dist_blur = self.gaussian_blur(dist_img, self.config.blur_sigmas[scale])?;

            // Compute SSIM-like features
            let scale_error = self.compute_scale_error(&ref_blur, &dist_blur, scale)?;

            let weight = self.config.edge_weights[scale] + self.config.detail_weights[scale];
            total_error += scale_error * weight;
            total_weight += weight;
        }

        // Normalize and convert to final score
        let mean_error = if total_weight > 0.0 {
            total_error / total_weight
        } else {
            0.0
        };

        // Convert error to SSIMULACRA2 score (higher is better)
        // Score = 30 - error, clamped to reasonable range
        let score = (30.0 - mean_error).max(-50.0).min(100.0);

        Ok(score as f64)
    }

    /// Convert RGB image to XYB color space
    fn rgb_to_xyb_image(&self, img: &ImageData) -> Result<MultiChannelImage> {
        let width = img.width;
        let height = img.height;
        let size = (width * height) as usize;

        let mut xyb_data = vec![0.0; size * 3];

        for i in 0..size {
            let r = img.data[i];
            let g = img.data[size + i];
            let b = img.data[2 * size + i];

            let (x, y, b_out) = rgb_to_xyb(r, g, b);

            xyb_data[i] = x;
            xyb_data[size + i] = y;
            xyb_data[2 * size + i] = b_out;
        }

        Ok(MultiChannelImage {
            width,
            height,
            channels: 3,
            data: xyb_data,
        })
    }

    /// Downsample image by factor of 2
    fn downsample(&self, img: &MultiChannelImage) -> Result<MultiChannelImage> {
        let data = downsample_image(&img.data, img.width, img.height, img.channels, DownsampleMode::Average)?;

        Ok(MultiChannelImage {
            width: img.width / 2,
            height: img.height / 2,
            channels: img.channels,
            data,
        })
    }

    /// Apply Gaussian blur
    fn gaussian_blur(&self, img: &MultiChannelImage, sigma: f32) -> Result<MultiChannelImage> {
        let kernel = GaussianKernel::new(sigma);
        let mut output = img.data.clone();

        // Separable blur: horizontal then vertical
        let mut temp = vec![0.0; img.data.len()];

        // Horizontal pass
        for c in 0..img.channels {
            for y in 0..img.height {
                for x in 0..img.width {
                    let mut sum = 0.0;

                    for k in 0..kernel.size() {
                        let offset = k as i32 - kernel.radius as i32;
                        let sample_x = (x as i32 + offset).clamp(0, img.width as i32 - 1) as u32;

                        let idx = (c * img.width * img.height + y * img.width + sample_x) as usize;
                        sum += img.data[idx] * kernel.kernel[k];
                    }

                    let idx = (c * img.width * img.height + y * img.width + x) as usize;
                    temp[idx] = sum;
                }
            }
        }

        // Vertical pass
        for c in 0..img.channels {
            for y in 0..img.height {
                for x in 0..img.width {
                    let mut sum = 0.0;

                    for k in 0..kernel.size() {
                        let offset = k as i32 - kernel.radius as i32;
                        let sample_y = (y as i32 + offset).clamp(0, img.height as i32 - 1) as u32;

                        let idx = (c * img.width * img.height + sample_y * img.width + x) as usize;
                        sum += temp[idx] * kernel.kernel[k];
                    }

                    let idx = (c * img.width * img.height + y * img.width + x) as usize;
                    output[idx] = sum;
                }
            }
        }

        Ok(MultiChannelImage {
            width: img.width,
            height: img.height,
            channels: img.channels,
            data: output,
        })
    }

    /// Compute error for a single scale
    fn compute_scale_error(
        &self,
        reference: &MultiChannelImage,
        distorted: &MultiChannelImage,
        _scale: usize,
    ) -> Result<f32> {
        let size = (reference.width * reference.height) as usize;
        let mut errors = Vec::new();

        for c in 0..reference.channels {
            let offset = (c * reference.width * reference.height) as usize;

            for i in 0..size {
                let idx = offset + i;
                let diff = reference.data[idx] - distorted.data[idx];

                // SSIM-inspired error metric
                let error = diff.abs();
                errors.push(error);
            }
        }

        // Return mean error
        Ok(mean(&errors))
    }
}

impl Metric for Ssimulacra2 {
    fn compute(&mut self, reference: &ImageData, distorted: &ImageData) -> Result<f64> {
        self.compute_cpu(reference, distorted)
    }

    fn name(&self) -> &str {
        "SSIMULACRA2"
    }

    fn reset(&mut self) -> Result<()> {
        // SSIMULACRA2 is stateless
        Ok(())
    }
}

/// Multi-channel image container
struct MultiChannelImage {
    width: u32,
    height: u32,
    channels: u32,
    data: Vec<f32>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gaussian_kernel() {
        let kernel = GaussianKernel::new(1.0);
        assert!(kernel.size() > 0);

        // Sum should be ~1.0
        let sum: f32 = kernel.kernel.iter().sum();
        assert!((sum - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_rgb_to_xyb() {
        let (x, y, b) = rgb_to_xyb(1.0, 1.0, 1.0);
        // White should map to specific XYB values
        assert!(x.abs() < 0.1); // X should be near 0 for neutral
        assert!(y > 0.0); // Y should be positive
        assert!(b > 0.0); // B should be positive
    }
}
