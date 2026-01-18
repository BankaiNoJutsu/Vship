// Integration tests for vship-metrics

#[cfg(test)]
mod tests {
    use crate::common::*;
    use crate::{ImageData, ImageFormat};

    #[test]
    fn test_gaussian_kernel_creation() {
        let kernel = GaussianKernel::new(1.0);

        // Kernel should have non-zero size
        assert!(kernel.size() > 0);

        // Kernel sum should be approximately 1.0
        let sum: f32 = kernel.kernel.iter().sum();
        assert!((sum - 1.0).abs() < 0.001);

        // Kernel should be symmetric
        let half = kernel.radius;
        for i in 0..half {
            let diff = (kernel.kernel[i] - kernel.kernel[kernel.size() - 1 - i]).abs();
            assert!(diff < 0.0001, "Kernel not symmetric at index {}", i);
        }
    }

    #[test]
    fn test_gaussian_kernel_different_sigmas() {
        let k1 = GaussianKernel::new(0.5);
        let k2 = GaussianKernel::new(2.0);

        // Larger sigma should produce larger kernel
        assert!(k2.size() > k1.size());

        // Both should still be normalized
        let sum1: f32 = k1.kernel.iter().sum();
        let sum2: f32 = k2.kernel.iter().sum();
        assert!((sum1 - 1.0).abs() < 0.001);
        assert!((sum2 - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_downsample_image() {
        let width = 8;
        let height = 8;
        let channels = 3;

        // Create test image with known pattern
        let mut input = vec![0.0f32; (width * height * channels) as usize];

        // Fill with checkerboard pattern
        for c in 0..channels {
            for y in 0..height {
                for x in 0..width {
                    let idx = (c * width * height + y * width + x) as usize;
                    input[idx] = ((x + y) % 2) as f32;
                }
            }
        }

        // Downsample
        let output = downsample_image(&input, width, height, channels, DownsampleMode::Average).unwrap();

        // Output should be half the size
        let expected_size = ((width / 2) * (height / 2) * channels) as usize;
        assert_eq!(output.len(), expected_size);

        // Values should be averages (0.5 for checkerboard)
        for &val in &output {
            assert!((val - 0.5).abs() < 0.01, "Expected ~0.5, got {}", val);
        }
    }

    #[test]
    fn test_upsample_image() {
        let width = 4;
        let height = 4;
        let channels = 1;

        // Create small test image
        let input: Vec<f32> = (0..16).map(|i| i as f32).collect();

        // Upsample
        let output = upsample_image(&input, width, height, channels).unwrap();

        // Output should be 2x size
        let expected_size = ((width * 2) * (height * 2) * channels) as usize;
        assert_eq!(output.len(), expected_size);
    }

    #[test]
    fn test_mean_calculation() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let m = mean(&values);
        assert_eq!(m, 3.0);
    }

    #[test]
    fn test_mean_empty() {
        let values: Vec<f32> = vec![];
        let m = mean(&values);
        assert_eq!(m, 0.0);
    }

    #[test]
    fn test_std_dev_calculation() {
        let values = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let sd = std_dev(&values);

        // Expected std dev is 2.0
        assert!((sd - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_rgb_to_xyb_conversion() {
        // Test white
        let (x, y, b) = rgb_to_xyb(1.0, 1.0, 1.0);
        assert!(x.abs() < 0.1); // X should be near 0 for neutral
        assert!(y > 0.0); // Y should be positive
        assert!(b > 0.0); // B should be positive

        // Test black
        let (x, y, b) = rgb_to_xyb(0.0, 0.0, 0.0);
        assert_eq!(x, 0.0);
        assert_eq!(y, 0.0);
        assert_eq!(b, 0.0);

        // Test red
        let (x, y, b) = rgb_to_xyb(1.0, 0.0, 0.0);
        // Red should have non-zero X and Y
        assert!(x != 0.0 || y != 0.0);
    }

    #[test]
    fn test_srgb_to_linear_roundtrip() {
        let test_values = vec![0.0, 0.25, 0.5, 0.75, 1.0];

        for &val in &test_values {
            let linear = srgb_to_linear(val);
            let back = linear_to_srgb(linear);

            assert!((val - back).abs() < 0.001, "Roundtrip failed for {}", val);
        }
    }

    #[test]
    fn test_image_data_creation() {
        let width = 640;
        let height = 480;

        // Test RGB format
        let img_rgb = ImageData::new(width, height, ImageFormat::RGB);
        assert_eq!(img_rgb.width, width);
        assert_eq!(img_rgb.height, height);
        assert_eq!(img_rgb.data.len(), (width * height * 3) as usize);

        // Test YUV420 format
        let img_yuv = ImageData::new(width, height, ImageFormat::YUV420);
        let expected_len = (width * height) as usize // Y plane
            + 2 * ((width / 2) * (height / 2)) as usize; // U and V planes
        assert_eq!(img_yuv.data.len(), expected_len);
    }

    #[test]
    fn test_image_data_from_f32() {
        let width = 4;
        let height = 4;
        let data: Vec<f32> = (0..48).map(|i| i as f32 / 48.0).collect();

        let img = ImageData::from_f32(width, height, &data, ImageFormat::RGB).unwrap();

        assert_eq!(img.width, width);
        assert_eq!(img.height, height);
        assert_eq!(img.data.len(), data.len());
        assert_eq!(img.data[0], 0.0);
        assert!((img.data[47] - 47.0 / 48.0).abs() < 0.001);
    }
}
