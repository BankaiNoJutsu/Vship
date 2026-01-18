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

    // Determine frame range
    let effective_end = if cli.end_frame == 0 {
        ref_reader.frame_count().min(dist_reader.frame_count())
    } else {
        cli.end_frame
    };

    let total_frames = (effective_end - cli.start_frame + cli.frame_step - 1) / cli.frame_step;

    // Check if we need streaming mode (large frame range)
    let frame_count = effective_end - cli.start_frame;
    let use_streaming = frame_count > 500;  // ~12GB threshold for 1080p

    #[cfg(feature = "ffmpeg")]
    if use_streaming {
        println!("\nUsing streaming mode for {} frames...", frame_count);
        ref_reader.enable_streaming()?;
        dist_reader.enable_streaming()?;
    } else {
        println!("\nLoading frames {} to {}...", cli.start_frame, effective_end);
        ref_reader.load_frame_range(cli.start_frame, effective_end)?;
        dist_reader.load_frame_range(cli.start_frame, effective_end)?;
    }

    // Process frames
    println!("\nProcessing {} frames...\n", total_frames);

    let mut scores = Vec::new();
    let mut frame_numbers = Vec::new();

    // Progress tracking
    let start_time = std::time::Instant::now();
    let mut last_update = start_time;
    let mut frames_since_update = 0;
    let mut current_fps = 0.0f64;
    let mut running_sum = 0.0f64;
    let mut last_score = 0.0f64;

    // Helper to print progress with scores
    let print_progress = |processed: usize, total: usize, fps: f64, elapsed: std::time::Duration,
                          last: f64, avg: f64| {
        let percent = (processed as f64 / total as f64 * 100.0).min(100.0);
        let eta_secs = if fps > 0.0 {
            (total - processed) as f64 / fps
        } else {
            0.0
        };
        let eta_min = (eta_secs / 60.0) as u32;
        let eta_sec = (eta_secs % 60.0) as u32;
        let elapsed_min = elapsed.as_secs() / 60;
        let elapsed_sec = elapsed.as_secs() % 60;

        print!("\r[{:>3.0}%] {}/{} | {:.1} fps | Score: {:.1} (avg {:.1}) | {:02}:{:02} ETA {:02}:{:02}  ",
               percent, processed, total, fps, last, avg, elapsed_min, elapsed_sec, eta_min, eta_sec);
        use std::io::Write;
        std::io::stdout().flush().ok();
    };

    #[cfg(feature = "ffmpeg")]
    if use_streaming {
        // Streaming mode: decode and process frames on-the-fly
        let mut current_frame = 0;
        let mut processed = 0;

        while let Some(ref_frame) = ref_reader.decode_next()? {
            // Get corresponding distorted frame
            let dist_frame = match dist_reader.decode_next()? {
                Some(f) => f,
                None => {
                    log::warn!("Distorted video ended before reference");
                    break;
                }
            };

            // Skip frames before start
            if current_frame < cli.start_frame {
                current_frame += 1;
                continue;
            }

            // Stop if past end
            if current_frame >= effective_end {
                break;
            }

            // Only process at step intervals
            if (current_frame - cli.start_frame) % cli.frame_step == 0 {
                let score = metric.compute(&ref_frame, &dist_frame)
                    .context(format!("Failed to compute metric for frame {}", current_frame))?;

                scores.push(score);
                frame_numbers.push(current_frame);
                processed += 1;
                frames_since_update += 1;
                running_sum += score;
                last_score = score;

                // Update progress every 100ms or every frame if slow
                let now = std::time::Instant::now();
                if now.duration_since(last_update).as_millis() >= 100 || processed == 1 {
                    let dt = now.duration_since(last_update).as_secs_f64();
                    if dt > 0.0 {
                        current_fps = frames_since_update as f64 / dt;
                    }
                    let avg_score = running_sum / processed as f64;
                    print_progress(processed, total_frames, current_fps, now.duration_since(start_time),
                                   last_score, avg_score);
                    last_update = now;
                    frames_since_update = 0;
                }
            }

            current_frame += 1;
        }
    } else {
        // Cached mode: frames already loaded
        for (i, frame_num) in (cli.start_frame..effective_end).step_by(cli.frame_step).enumerate() {
            let ref_frame = ref_reader.read_frame(frame_num)
                .context(format!("Failed to read reference frame {}", frame_num))?;

            let dist_frame = dist_reader.read_frame(frame_num)
                .context(format!("Failed to read distorted frame {}", frame_num))?;

            let score = metric.compute(&ref_frame, &dist_frame)
                .context(format!("Failed to compute metric for frame {}", frame_num))?;

            scores.push(score);
            frame_numbers.push(frame_num);
            frames_since_update += 1;
            running_sum += score;
            last_score = score;

            // Update progress every 100ms or every frame if slow
            let now = std::time::Instant::now();
            if now.duration_since(last_update).as_millis() >= 100 || i == 0 {
                let dt = now.duration_since(last_update).as_secs_f64();
                if dt > 0.0 {
                    current_fps = frames_since_update as f64 / dt;
                }
                let avg_score = running_sum / (i + 1) as f64;
                print_progress(i + 1, total_frames, current_fps, now.duration_since(start_time),
                               last_score, avg_score);
                last_update = now;
                frames_since_update = 0;
            }
        }
    }

    #[cfg(not(feature = "ffmpeg"))]
    for (i, frame_num) in (cli.start_frame..effective_end).step_by(cli.frame_step).enumerate() {
        let ref_frame = ref_reader.read_frame(frame_num)
            .context(format!("Failed to read reference frame {}", frame_num))?;

        let dist_frame = dist_reader.read_frame(frame_num)
            .context(format!("Failed to read distorted frame {}", frame_num))?;

        let score = metric.compute(&ref_frame, &dist_frame)
            .context(format!("Failed to compute metric for frame {}", frame_num))?;

        scores.push(score);
        frame_numbers.push(frame_num);
        frames_since_update += 1;
        running_sum += score;
        last_score = score;

        // Update progress
        let now = std::time::Instant::now();
        if now.duration_since(last_update).as_millis() >= 100 || i == 0 {
            let dt = now.duration_since(last_update).as_secs_f64();
            if dt > 0.0 {
                current_fps = frames_since_update as f64 / dt;
            }
            let avg_score = running_sum / (i + 1) as f64;
            print_progress(i + 1, total_frames, current_fps, now.duration_since(start_time),
                           last_score, avg_score);
            last_update = now;
            frames_since_update = 0;
        }
    }

    // Final progress update
    let total_elapsed = start_time.elapsed();
    let avg_fps = scores.len() as f64 / total_elapsed.as_secs_f64();
    let final_avg = if scores.is_empty() { 0.0 } else { running_sum / scores.len() as f64 };
    println!("\r[100%] {} frames | {:.1} fps | Avg score: {:.2} | {:.1}s                           ",
             scores.len(), avg_fps, final_avg, total_elapsed.as_secs_f64());

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
