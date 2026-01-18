// Simple example demonstrating metric computation with Vship

use vship_metrics::{MetricsContext, Metric, ImageData, ImageFormat};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    env_logger::init();

    println!("Vship Simple Metric Example");
    println!("============================\n");

    // Create Vship context (initializes Vulkan)
    println!("Initializing Vship...");
    let ctx = MetricsContext::new()?;
    println!("✓ Vulkan initialized\n");

    // Create SSIMULACRA2 metric
    println!("Creating SSIMULACRA2 metric...");
    let mut ssimulacra2 = ctx.create_ssimulacra2()?;
    println!("✓ Metric created\n");

    // Create test images (1920x1080)
    let width = 1920;
    let height = 1080;

    println!("Creating test images ({}x{})...", width, height);
    let mut reference = ImageData::new(width, height, ImageFormat::RGB);
    let mut distorted = ImageData::new(width, height, ImageFormat::RGB);

    // Fill with test pattern
    for i in 0..reference.data.len() {
        reference.data[i] = (i % 256) as f32 / 255.0;
        distorted.data[i] = ((i + 10) % 256) as f32 / 255.0;
    }
    println!("✓ Test images created\n");

    // Compute metric
    println!("Computing SSIMULACRA2 score...");
    let score = ssimulacra2.compute(&reference, &distorted)?;
    println!("✓ Score: {:.4}\n", score);

    // Try other metrics
    println!("Testing Butteraugli...");
    let mut butteraugli = ctx.create_butteraugli()?;
    let butter_score = butteraugli.compute(&reference, &distorted)?;
    println!("✓ Butteraugli score: {:.4}\n", butter_score);

    println!("Testing CVVDP...");
    let mut cvvdp = ctx.create_cvvdp()?;
    let cvvdp_score = cvvdp.compute(&reference, &distorted)?;
    println!("✓ CVVDP score: {:.4}\n", cvvdp_score);

    println!("All metrics computed successfully!");

    Ok(())
}
