// Butteraugli - Psychovisual image difference metric
// Based on Google's libjxl butteraugli algorithm

use crate::{ImageData, Metric};
use vship_core::device::VulkanDevice;
use vship_core::error::Result;
use std::sync::Arc;

/// Butteraugli metric implementation
pub struct Butteraugli {
    device: Arc<VulkanDevice>,
    intensity_target: f32,
}

impl Butteraugli {
    /// Create new Butteraugli metric
    pub fn new(device: Arc<VulkanDevice>) -> Result<Self> {
        Ok(Self {
            device,
            intensity_target: 80.0, // Default: 80 nits
        })
    }

    /// Create with custom intensity target
    pub fn with_intensity_target(device: Arc<VulkanDevice>, intensity_target: f32) -> Result<Self> {
        Ok(Self {
            device,
            intensity_target,
        })
    }

    /// Compute Butteraugli distance (CPU implementation)
    /// TODO: Replace with GPU implementation using Vulkan compute shaders
    fn compute_cpu(&self, reference: &ImageData, distorted: &ImageData) -> Result<f64> {
        if reference.width != distorted.width || reference.height != distorted.height {
            return Err(vship_core::error::VshipError::InvalidDimensions {
                width: distorted.width,
                height: distorted.height,
            });
        }

        // Simplified Butteraugli implementation
        // The full implementation involves:
        // 1. Opsin dynamics (color space transformation)
        // 2. Frequency decomposition (low, mid, high frequency bands)
        // 3. Psychovisual masking model
        // 4. Malta (asymmetric) difference computation
        // 5. Edge detection and preservation

        let ref_rgb = reference.to_linear_rgb();
        let dist_rgb = distorted.to_linear_rgb();

        let size = (reference.width * reference.height) as usize;
        let mut total_diff = 0.0f64;

        // Simple perceptual difference (placeholder)
        for i in 0..(size * 3) {
            let diff = (ref_rgb.data[i] - dist_rgb.data[i]).abs();
            // Apply simple psychovisual weighting
            let weighted_diff = diff as f64 * self.perceptual_weight(ref_rgb.data[i]);
            total_diff += weighted_diff;
        }

        let mean_diff = total_diff / (size * 3) as f64;

        // Scale to approximate Butteraugli range
        let score = mean_diff * 100.0;

        Ok(score)
    }

    /// Simple perceptual weighting function
    fn perceptual_weight(&self, value: f32) -> f64 {
        // Higher weight for mid-tones, lower for very dark/bright regions
        let normalized = value / self.intensity_target;
        let weight = 1.0 - (normalized - 0.5).abs() * 0.5;
        weight.max(0.5) as f64
    }

    /// Compute distortion map (for visualization)
    /// TODO: Implement full distortion map generation
    pub fn compute_distortion_map(
        &self,
        reference: &ImageData,
        distorted: &ImageData,
    ) -> Result<Vec<f32>> {
        let size = (reference.width * reference.height) as usize;
        let mut distortion_map = vec![0.0; size];

        let ref_rgb = reference.to_linear_rgb();
        let dist_rgb = distorted.to_linear_rgb();

        for i in 0..size {
            let mut diff_sum = 0.0;
            for c in 0..3 {
                let idx = c * size + i;
                let diff = (ref_rgb.data[idx] - dist_rgb.data[idx]).abs();
                diff_sum += diff;
            }
            distortion_map[i] = diff_sum / 3.0;
        }

        Ok(distortion_map)
    }
}

impl Metric for Butteraugli {
    fn compute(&mut self, reference: &ImageData, distorted: &ImageData) -> Result<f64> {
        self.compute_cpu(reference, distorted)
    }

    fn name(&self) -> &str {
        "Butteraugli"
    }

    fn reset(&mut self) -> Result<()> {
        // Butteraugli is stateless
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_perceptual_weight() {
        let ctx = match vship_core::VshipContext::new() {
            Ok(ctx) => ctx,
            Err(_) => return,
        };
        let metric = Butteraugli::new(ctx.default_device()).unwrap();

        // Mid-tones should have higher weight
        let mid_weight = metric.perceptual_weight(40.0);
        let dark_weight = metric.perceptual_weight(5.0);
        let bright_weight = metric.perceptual_weight(75.0);

        assert!(mid_weight >= dark_weight);
        assert!(mid_weight >= bright_weight);
    }
}
