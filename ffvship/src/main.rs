// FFVship - CLI tool for computing video quality metrics
// Supports SSIMULACRA2, Butteraugli, and CVVDP metrics

mod video;
#[cfg(feature = "ffmpeg")]
mod ffmpeg_decoder;

use clap::{Parser, ValueEnum};
use anyhow::{Result, Context};
use vship_metrics::{MetricsContext, Metric};
use std::path::PathBuf;
use video::VideoReader;

#[derive(Debug, Clone, Copy, ValueEnum)]
enum MetricType {
    /// SSIMULACRA2 metric
    Ssimulacra2,
    /// Butteraugli metric
    Butteraugli,
    /// CVVDP metric
    Cvvdp,
}

#[derive(Parser)]
#[command(name = "ffvship")]
#[command(about = "GPU-accelerated video quality metrics", long_about = None)]
#[command(version = "4.1.0")]
struct Cli {
    /// Reference video file
    #[arg(short, long)]
    reference: PathBuf,

    /// Distorted video file
    #[arg(short, long)]
    distorted: PathBuf,

    /// Metric to compute
    #[arg(short, long, value_enum, default_value = "ssimulacra2")]
    metric: MetricType,

    /// Output file for scores (JSON format)
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Start frame (0-indexed)
    #[arg(long, default_value = "0")]
    start_frame: usize,

    /// End frame (exclusive, 0 = end of video)
    #[arg(long, default_value = "0")]
    end_frame: usize,

    /// Frame step (process every Nth frame)
    #[arg(long, default_value = "1")]
    frame_step: usize,

    /// Verbose output
    #[arg(short, long)]
    verbose: bool,

    /// GPU device index
    #[arg(long, default_value = "0")]
    device: usize,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    // Initialize logging
    if cli.verbose {
        env_logger::Builder::from_default_env()
            .filter_level(log::LevelFilter::Info)
            .init();
    } else {
        env_logger::Builder::from_default_env()
            .filter_level(log::LevelFilter::Warn)
            .init();
    }

    println!("FFVship 4.1.0 (Rust/Vulkan)");
    println!("─────────────────────────────");

    // Initialize Vship
    println!("Initializing Vulkan...");
    let ctx = MetricsContext::new()
        .context("Failed to initialize Vship context")?;

    // Create metric
    println!("Creating {:?} metric...", cli.metric);
    let mut metric: Box<dyn Metric> = match cli.metric {
        MetricType::Ssimulacra2 => Box::new(ctx.create_ssimulacra2()?),
        MetricType::Butteraugli => Box::new(ctx.create_butteraugli()?),
        MetricType::Cvvdp => Box::new(ctx.create_cvvdp()?),
    };

    // Open video files
    println!("Opening reference video: {:?}", cli.reference);
    let mut ref_reader = VideoReader::open(&cli.reference)?;

    println!("Opening distorted video: {:?}", cli.distorted);
    let mut dist_reader = VideoReader::open(&cli.distorted)?;

    // Validate dimensions match
    if ref_reader.width() != dist_reader.width() || ref_reader.height() != dist_reader.height() {
        anyhow::bail!(
            "Video dimensions mismatch: reference {}x{}, distorted {}x{}",
            ref_reader.width(),
            ref_reader.height(),
            dist_reader.width(),
            dist_reader.height()
        );
    }

    println!("Video resolution: {}x{}", ref_reader.width(), ref_reader.height());
    println!("Frame count: {}", ref_reader.frame_count());

    // Process frames
    println!("\nProcessing frames...");
    let effective_end = if cli.end_frame == 0 {
        ref_reader.frame_count().min(dist_reader.frame_count())
    } else {
        cli.end_frame
    };

    let total_frames = (effective_end - cli.start_frame) / cli.frame_step;
    println!("Processing {} frames ({}..{}, step {})",
             total_frames, cli.start_frame, effective_end, cli.frame_step);

    let mut scores = Vec::new();
    let mut frame_numbers = Vec::new();

    for (i, frame_num) in (cli.start_frame..effective_end).step_by(cli.frame_step).enumerate() {
        if cli.verbose {
            print!("\rFrame {}/{}: {}", i + 1, total_frames, frame_num);
            use std::io::Write;
            std::io::stdout().flush()?;
        }

        let ref_frame = ref_reader.read_frame(frame_num)
            .context(format!("Failed to read reference frame {}", frame_num))?;

        let dist_frame = dist_reader.read_frame(frame_num)
            .context(format!("Failed to read distorted frame {}", frame_num))?;

        let score = metric.compute(&ref_frame, &dist_frame)
            .context(format!("Failed to compute metric for frame {}", frame_num))?;

        scores.push(score);
        frame_numbers.push(frame_num);
    }

    if cli.verbose {
        println!();
    }

    // Compute statistics
    let mean_score = scores.iter().sum::<f64>() / scores.len() as f64;
    let min_score = scores.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max_score = scores.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

    // Compute standard deviation
    let variance = scores.iter()
        .map(|s| (s - mean_score).powi(2))
        .sum::<f64>() / scores.len() as f64;
    let std_dev = variance.sqrt();

    println!("\nResults:");
    println!("─────────────────────────────");
    println!("{} Statistics:", metric.name());
    println!("  Mean:   {:.4}", mean_score);
    println!("  Min:    {:.4}", min_score);
    println!("  Max:    {:.4}", max_score);
    println!("  StdDev: {:.4}", std_dev);
    println!("  Frames: {}", scores.len());

    // Save results if output file specified
    if let Some(output_path) = cli.output {
        let results = serde_json::json!({
            "metric": metric.name(),
            "reference": cli.reference,
            "distorted": cli.distorted,
            "version": "4.1.0",
            "statistics": {
                "mean": mean_score,
                "min": min_score,
                "max": max_score,
                "std_dev": std_dev,
                "frame_count": scores.len(),
            },
            "per_frame_scores": scores.iter().enumerate().map(|(i, &score)| {
                serde_json::json!({
                    "frame": frame_numbers[i],
                    "score": score,
                })
            }).collect::<Vec<_>>(),
        });

        std::fs::write(&output_path, serde_json::to_string_pretty(&results)?)?;
        println!("\nResults saved to: {:?}", output_path);
    }

    Ok(())
}
