// Common utilities for all metrics

use vship_core::error::Result;

/// Gaussian blur kernel generator
pub struct GaussianKernel {
    pub kernel: Vec<f32>,
    pub radius: usize,
}

impl GaussianKernel {
    /// Create a Gaussian kernel with given sigma
    pub fn new(sigma: f32) -> Self {
        // Radius is typically 3*sigma
        let radius = (3.0 * sigma).ceil() as usize;
        let size = 2 * radius + 1;

        let mut kernel = vec![0.0; size];
        let mut sum = 0.0;

        for i in 0..size {
            let x = i as f32 - radius as f32;
            kernel[i] = (-x * x / (2.0 * sigma * sigma)).exp();
            sum += kernel[i];
        }

        // Normalize
        for k in &mut kernel {
            *k /= sum;
        }

        Self { kernel, radius }
    }

    /// Get kernel size
    pub fn size(&self) -> usize {
        2 * self.radius + 1
    }
}

/// Downsampling modes
#[derive(Debug, Clone, Copy)]
pub enum DownsampleMode {
    Average,
    Bilinear,
    Lanczos,
}

/// Image downsampling utility
pub fn downsample_image(
    input: &[f32],
    width: u32,
    height: u32,
    channels: u32,
    mode: DownsampleMode,
) -> Result<Vec<f32>> {
    let out_width = width / 2;
    let out_height = height / 2;
    let mut output = vec![0.0; (out_width * out_height * channels) as usize];

    match mode {
        DownsampleMode::Average => {
            for c in 0..channels {
                for y in 0..out_height {
                    for x in 0..out_width {
                        let sx = x * 2;
                        let sy = y * 2;

                        let idx = |x: u32, y: u32| -> usize {
                            (c * width * height + y * width + x) as usize
                        };

                        let sum = input[idx(sx, sy)]
                            + input[idx(sx + 1, sy)]
                            + input[idx(sx, sy + 1)]
                            + input[idx(sx + 1, sy + 1)];

                        let out_idx = (c * out_width * out_height + y * out_width + x) as usize;
                        output[out_idx] = sum * 0.25;
                    }
                }
            }
        }
        _ => {
            // TODO: Implement other downsampling modes
            return downsample_image(input, width, height, channels, DownsampleMode::Average);
        }
    }

    Ok(output)
}

/// Image upsampling utility
pub fn upsample_image(
    input: &[f32],
    width: u32,
    height: u32,
    channels: u32,
) -> Result<Vec<f32>> {
    let out_width = width * 2;
    let out_height = height * 2;
    let mut output = vec![0.0; (out_width * out_height * channels) as usize];

    for c in 0..channels {
        for y in 0..out_height {
            for x in 0..out_width {
                let sx = x / 2;
                let sy = y / 2;

                let in_idx = (c * width * height + sy * width + sx) as usize;
                let out_idx = (c * out_width * out_height + y * out_width + x) as usize;

                output[out_idx] = input[in_idx];
            }
        }
    }

    Ok(output)
}

/// Calculate mean of values
pub fn mean(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f32>() / values.len() as f32
}

/// Calculate standard deviation
pub fn std_dev(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }

    let mean = mean(values);
    let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / values.len() as f32;
    variance.sqrt()
}

/// Linear to sRGB conversion
pub fn linear_to_srgb(linear: f32) -> f32 {
    if linear <= 0.0031308 {
        12.92 * linear
    } else {
        1.055 * linear.powf(1.0 / 2.4) - 0.055
    }
}

/// sRGB to linear conversion
pub fn srgb_to_linear(srgb: f32) -> f32 {
    if srgb <= 0.04045 {
        srgb / 12.92
    } else {
        ((srgb + 0.055) / 1.055).powf(2.4)
    }
}

/// RGB to XYB color space (used by SSIMULACRA2)
pub fn rgb_to_xyb(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    // Approximate opsin absorbance
    let mixed_r = 0.3 * r + 0.622 * g + 0.078 * b;
    let mixed_g = 0.23 * r + 0.692 * g + 0.078 * b;
    let mixed_b = 0.243 * r + 0.419 * g + 0.338 * b;

    // Gamma correction
    let gamma = |v: f32| -> f32 {
        if v > 0.0 {
            v.cbrt()
        } else {
            -(-v).cbrt()
        }
    };

    let l = gamma(mixed_r);
    let m = gamma(mixed_g);
    let s = gamma(mixed_b);

    // Convert to XYB
    let x = 0.5 * (l - m);
    let y = 0.5 * (l + m);
    let b_out = s;

    (x, y, b_out)
}
