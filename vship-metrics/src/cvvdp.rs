// CVVDP - Colored Video Visual Difference Predictor
// Based on University of Cambridge's ColorVideoVDP algorithm

use crate::{ImageData, Metric};
use vship_core::device::VulkanDevice;
use vship_core::error::Result;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Display configuration for CVVDP
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DisplayConfig {
    pub name: String,
    pub resolution: (u32, u32),
    pub diagonal_size_inches: f32,
    pub viewing_distance_meters: f32,
    pub peak_luminance: f32,
    pub contrast_ratio: f32,
}

impl DisplayConfig {
    /// Full HD display at typical viewing distance
    pub fn fhd() -> Self {
        Self {
            name: "FHD".to_string(),
            resolution: (1920, 1080),
            diagonal_size_inches: 24.0,
            viewing_distance_meters: 0.6,
            peak_luminance: 200.0,
            contrast_ratio: 1000.0,
        }
    }

    /// 4K display
    pub fn uhd() -> Self {
        Self {
            name: "4K".to_string(),
            resolution: (3840, 2160),
            diagonal_size_inches: 55.0,
            viewing_distance_meters: 1.5,
            peak_luminance: 400.0,
            contrast_ratio: 5000.0,
        }
    }

    /// HDR display
    pub fn hdr() -> Self {
        Self {
            name: "HDR".to_string(),
            resolution: (3840, 2160),
            diagonal_size_inches: 55.0,
            viewing_distance_meters: 1.5,
            peak_luminance: 1000.0,
            contrast_ratio: 100000.0,
        }
    }
}

/// CVVDP metric implementation
pub struct Cvvdp {
    device: Arc<VulkanDevice>,
    display_config: DisplayConfig,
    frame_buffer: Vec<ImageData>,
    temporal_enabled: bool,
}

impl Cvvdp {
    /// Create new CVVDP metric with default FHD display
    pub fn new(device: Arc<VulkanDevice>) -> Result<Self> {
        Ok(Self {
            device,
            display_config: DisplayConfig::fhd(),
            frame_buffer: Vec::new(),
            temporal_enabled: true,
        })
    }

    /// Create with custom display configuration
    pub fn with_display(device: Arc<VulkanDevice>, display_config: DisplayConfig) -> Result<Self> {
        Ok(Self {
            device,
            display_config,
            frame_buffer: Vec::new(),
            temporal_enabled: true,
        })
    }

    /// Enable/disable temporal processing
    pub fn set_temporal_processing(&mut self, enabled: bool) {
        self.temporal_enabled = enabled;
    }

    /// Compute CVVDP score (CPU implementation)
    /// TODO: Replace with GPU implementation using Vulkan compute shaders
    fn compute_cpu(&mut self, reference: &ImageData, distorted: &ImageData) -> Result<f64> {
        if reference.width != distorted.width || reference.height != distorted.height {
            return Err(vship_core::error::VshipError::InvalidDimensions {
                width: distorted.width,
                height: distorted.height,
            });
        }

        // Simplified CVVDP implementation
        // The full implementation involves:
        // 1. Display model transformation
        // 2. Spatial frequency decomposition (Laplacian pyramid)
        // 3. Contrast Sensitivity Function (CSF) modeling
        // 4. Temporal filtering (sustained and transient channels)
        // 5. Masking model
        // 6. Probability summation

        let ref_rgb = reference.to_linear_rgb();
        let dist_rgb = distorted.to_linear_rgb();

        // Apply display model
        let ref_display = self.apply_display_model(&ref_rgb)?;
        let dist_display = self.apply_display_model(&dist_rgb)?;

        // Compute spatial differences
        let spatial_diff = self.compute_spatial_difference(&ref_display, &dist_display)?;

        // Apply CSF (Contrast Sensitivity Function)
        let csf_weighted = self.apply_csf(spatial_diff)?;

        // Temporal processing if enabled
        let final_score = if self.temporal_enabled {
            self.frame_buffer.push(reference.clone());
            if self.frame_buffer.len() > 5 {
                self.frame_buffer.remove(0);
            }

            // Simple temporal weighting (placeholder)
            csf_weighted * 0.9
        } else {
            csf_weighted
        };

        // Convert to Q-score (Just Objectionable Difference units)
        // Higher score = more visible difference
        let q_score = final_score * 10.0;

        Ok(q_score)
    }

    /// Apply display model transformation
    fn apply_display_model(&self, image: &ImageData) -> Result<ImageData> {
        let mut output = image.clone();

        // Apply peak luminance scaling
        let scale = self.display_config.peak_luminance / 100.0;

        for pixel in &mut output.data {
            *pixel *= scale;

            // Apply contrast ratio clipping
            let min_luminance = self.display_config.peak_luminance / self.display_config.contrast_ratio;
            *pixel = pixel.max(min_luminance / 100.0);
        }

        Ok(output)
    }

    /// Compute spatial difference
    fn compute_spatial_difference(
        &self,
        reference: &ImageData,
        distorted: &ImageData,
    ) -> Result<f64> {
        let size = (reference.width * reference.height) as usize;
        let mut total_diff = 0.0;

        for i in 0..(size * 3) {
            let diff = (reference.data[i] - distorted.data[i]).abs();
            total_diff += diff as f64;
        }

        Ok(total_diff / (size * 3) as f64)
    }

    /// Apply Contrast Sensitivity Function
    fn apply_csf(&self, diff: f64) -> Result<f64> {
        // Simplified CSF model
        // Real CSF considers spatial frequency, viewing distance, and luminance

        // Calculate viewing angle per pixel
        let diagonal_pixels = ((self.display_config.resolution.0.pow(2)
            + self.display_config.resolution.1.pow(2)) as f32)
            .sqrt();

        let diagonal_size_meters = self.display_config.diagonal_size_inches * 0.0254;

        let pixels_per_degree = diagonal_pixels * self.display_config.viewing_distance_meters
            / (diagonal_size_meters * (std::f32::consts::PI / 180.0).tan());

        // CSF peaks around 4-8 cycles per degree
        let csf_scale = pixels_per_degree / 4.0;

        Ok(diff * csf_scale.max(0.5) as f64)
    }

    /// Reset per-scene state (for video processing)
    pub fn reset_scene(&mut self) {
        self.frame_buffer.clear();
    }
}

impl Metric for Cvvdp {
    fn compute(&mut self, reference: &ImageData, distorted: &ImageData) -> Result<f64> {
        self.compute_cpu(reference, distorted)
    }

    fn name(&self) -> &str {
        "CVVDP"
    }

    fn reset(&mut self) -> Result<()> {
        self.frame_buffer.clear();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_display_configs() {
        let fhd = DisplayConfig::fhd();
        assert_eq!(fhd.resolution, (1920, 1080));

        let uhd = DisplayConfig::uhd();
        assert_eq!(uhd.resolution, (3840, 2160));

        let hdr = DisplayConfig::hdr();
        assert!(hdr.peak_luminance > uhd.peak_luminance);
    }
}
