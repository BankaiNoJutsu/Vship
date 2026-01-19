// FFVship - CLI tool for computing video quality metrics
// Supports SSIMULACRA2, Butteraugli, and CVVDP metrics

mod video;
#[cfg(feature = "ffmpeg")]
mod ffmpeg_decoder;

use clap::{Parser, ValueEnum};
use anyhow::{Result, Context};
use vship_metrics::{ComputeMode, ReduceMode, ImageData, ImageDataRgba8, MetricsContext, Metric};
use std::sync::mpsc::{self, SyncSender, Receiver};
use std::cell::Cell;
use std::thread;
use std::time::Instant;
use std::path::PathBuf;
use video::VideoReader;

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum MetricType {
    /// SSIMULACRA2 metric
    Ssimulacra2,
    /// Butteraugli metric
    Butteraugli,
    /// CVVDP metric
    Cvvdp,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum GpuMode {
    /// Single command buffer per frame
    Single,
    /// Per-step batches with waits
    Legacy,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum ReduceModeCli {
    /// Reduce on GPU and read back a single value
    Gpu,
    /// Read back full buffer and reduce on CPU
    Cpu,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum InputFormat {
    /// Packed RGBA8 with GPU normalization
    Rgba8,
    /// Convert to planar f32 on CPU (legacy)
    F32,
}

impl From<ReduceModeCli> for ReduceMode {
    fn from(mode: ReduceModeCli) -> Self {
        match mode {
            ReduceModeCli::Gpu => ReduceMode::Gpu,
            ReduceModeCli::Cpu => ReduceMode::Cpu,
        }
    }
}

impl From<GpuMode> for ComputeMode {
    fn from(mode: GpuMode) -> Self {
        match mode {
            GpuMode::Single => ComputeMode::SingleBatch,
            GpuMode::Legacy => ComputeMode::LegacyBatched,
        }
    }
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

    /// GPU compute mode
    #[arg(long, value_enum, default_value = "single")]
    gpu_mode: GpuMode,

    /// Number of frames to keep in flight
    #[arg(long, default_value = "1")]
    in_flight: usize,

    /// Max in-flight frames when adaptive mode is enabled (in_flight=0)
    #[arg(long, default_value = "4")]
    in_flight_max: usize,

    /// Reduction mode
    #[arg(long, value_enum, default_value = "gpu")]
    reduce_mode: ReduceModeCli,

    /// Input pixel format for comparison
    #[arg(long, value_enum, default_value = "rgba8")]
    input_format: InputFormat,

    /// Frames per GPU submit (single mode only)
    #[arg(long, default_value = "1")]
    batch_frames: usize,
}

struct Job {
    frame_num: usize,
    ref_frame: ImageDataRgba8,
    dist_frame: ImageDataRgba8,
}

struct JobResult {
    frame_num: usize,
    score: Option<f64>,
    gpu_time_ns: u64,
    wall_ns: u64,
    error: Option<String>,
}

fn create_metric_instance(
    ctx: &MetricsContext,
    metric: MetricType,
    gpu_mode: GpuMode,
    reduce_mode: ReduceModeCli,
) -> Result<Box<dyn Metric>> {
    let mut metric: Box<dyn Metric> = match metric {
        MetricType::Ssimulacra2 => Box::new(ctx.create_ssimulacra2()?),
        MetricType::Butteraugli => Box::new(ctx.create_butteraugli()?),
        MetricType::Cvvdp => Box::new(ctx.create_cvvdp()?),
    };
    metric.set_compute_mode(gpu_mode.into());
    metric.set_reduce_mode(reduce_mode.into());
    Ok(metric)
}

fn spawn_workers(
    ctx: std::sync::Arc<MetricsContext>,
    metric: MetricType,
    gpu_mode: GpuMode,
    reduce_mode: ReduceModeCli,
    input_format: InputFormat,
    count: usize,
) -> (Vec<SyncSender<Job>>, Receiver<JobResult>) {
    let (result_tx, result_rx) = mpsc::channel::<JobResult>();
    let mut senders = Vec::with_capacity(count);

    for _ in 0..count {
        let (job_tx, job_rx) = mpsc::sync_channel::<Job>(1);
        let result_tx = result_tx.clone();
        let metric_type = metric;
        let gpu_mode = gpu_mode;
        let input_format = input_format;
        let ctx = std::sync::Arc::clone(&ctx);

        thread::spawn(move || {
            let mut metric = match create_metric_instance(&ctx, metric_type, gpu_mode, reduce_mode) {
                Ok(metric) => metric,
                Err(err) => {
                    let _ = result_tx.send(JobResult {
                        frame_num: 0,
                        score: None,
                        gpu_time_ns: 0,
                        wall_ns: 0,
                        error: Some(err.to_string()),
                    });
                    return;
                }
            };

            for job in job_rx {
                let start = Instant::now();
                let score = match input_format {
                    InputFormat::Rgba8 => metric.compute_rgba8(&job.ref_frame, &job.dist_frame),
                    InputFormat::F32 => {
                        let ref_f32 = ImageData::from_rgba8(
                            job.ref_frame.width,
                            job.ref_frame.height,
                            &job.ref_frame.data,
                        );
                        let dist_f32 = ImageData::from_rgba8(
                            job.dist_frame.width,
                            job.dist_frame.height,
                            &job.dist_frame.data,
                        );
                        match (ref_f32, dist_f32) {
                            (Ok(ref_f32), Ok(dist_f32)) => metric.compute(&ref_f32, &dist_f32),
                            (Err(err), _) | (_, Err(err)) => Err(err),
                        }
                    }
                };
                let wall_ns = start.elapsed().as_nanos() as u64;
                let gpu_time_ns = metric.gpu_time_ns().unwrap_or(0);
                let result = match score {
                    Ok(score) => JobResult {
                        frame_num: job.frame_num,
                        score: Some(score),
                        gpu_time_ns,
                        wall_ns,
                        error: None,
                    },
                    Err(err) => JobResult {
                        frame_num: job.frame_num,
                        score: None,
                        gpu_time_ns,
                        wall_ns,
                        error: Some(err.to_string()),
                    },
                };
                if result_tx.send(result).is_err() {
                    break;
                }
            }
        });

        senders.push(job_tx);
    }

    (senders, result_rx)
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
    let ctx = std::sync::Arc::new(
        MetricsContext::new().context("Failed to initialize Vship context")?
    );
    let ctx_ref = std::sync::Arc::clone(&ctx);
    let use_inflight = cli.in_flight != 1;
    let adaptive_inflight = cli.in_flight == 0;
    let use_batch = cli.batch_frames > 1;

    if use_batch {
        if cli.metric != MetricType::Ssimulacra2
            || cli.input_format != InputFormat::Rgba8
            || cli.reduce_mode != ReduceModeCli::Gpu
            || cli.gpu_mode != GpuMode::Single
            || use_inflight
        {
            anyhow::bail!(
                "--batch-frames requires --metric ssimulacra2 --input-format rgba8 --reduce-mode gpu --gpu-mode single and --in-flight 1"
            );
        }
    }

    // Create metric
    println!("Creating {:?} metric...", cli.metric);
    let mut metric: Option<Box<dyn Metric>> = None;
    let mut batch_metric: Option<vship_metrics::Ssimulacra2Gpu> = None;

    if use_batch {
        let mut metric_impl = ctx_ref.create_ssimulacra2()?;
        metric_impl.set_compute_mode(cli.gpu_mode.into());
        metric_impl.set_reduce_mode(cli.reduce_mode.into());
        batch_metric = Some(metric_impl);
    } else {
        let mut metric_impl: Box<dyn Metric> = match cli.metric {
            MetricType::Ssimulacra2 => Box::new(ctx_ref.create_ssimulacra2()?),
            MetricType::Butteraugli => Box::new(ctx_ref.create_butteraugli()?),
            MetricType::Cvvdp => Box::new(ctx_ref.create_cvvdp()?),
        };
        metric_impl.set_compute_mode(cli.gpu_mode.into());
        metric_impl.set_reduce_mode(cli.reduce_mode.into());
        metric = Some(metric_impl);
    }
    println!("GPU mode: {:?}", cli.gpu_mode);
    println!("Reduce mode: {:?}", cli.reduce_mode);
    println!("Input format: {:?}", cli.input_format);
    if use_batch {
        println!("Batch frames: {}", cli.batch_frames);
    }
    let metric_name = if use_batch {
        batch_metric.as_ref().unwrap().name().to_string()
    } else {
        metric.as_ref().unwrap().name().to_string()
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
    let mut last_gpu_usage: Option<f64> = None;

    // Helper to print progress with scores
    let print_progress = |processed: usize, total: usize, fps: f64, elapsed: std::time::Duration,
                          last: f64, avg: f64, gpu_usage: Option<f64>| {
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

        match gpu_usage {
            Some(gpu) => print!(
                "\r[{:>3.0}%] {}/{} | {:.1} fps | GPU: {:>4.1}% | Score: {:.1} (avg {:.1}) | {:02}:{:02} ETA {:02}:{:02}  ",
                percent, processed, total, fps, gpu, last, avg, elapsed_min, elapsed_sec, eta_min, eta_sec
            ),
            None => print!(
                "\r[{:>3.0}%] {}/{} | {:.1} fps | GPU:  --  | Score: {:.1} (avg {:.1}) | {:02}:{:02} ETA {:02}:{:02}  ",
                percent, processed, total, fps, last, avg, elapsed_min, elapsed_sec, eta_min, eta_sec
            ),
        }
        use std::io::Write;
        std::io::stdout().flush().ok();
    };


    #[cfg(feature = "ffmpeg")]
    if use_streaming && !use_inflight {
        if use_batch {
            let mut current_frame = 0usize;
            let mut processed = 0usize;
            let mut batch_refs: Vec<ImageDataRgba8> = Vec::with_capacity(cli.batch_frames);
            let mut batch_dists: Vec<ImageDataRgba8> = Vec::with_capacity(cli.batch_frames);
            let mut batch_frames: Vec<usize> = Vec::with_capacity(cli.batch_frames);
            let metric = batch_metric.as_mut().unwrap();

            let mut flush_batch = |batch_refs: &mut Vec<ImageDataRgba8>,
                                   batch_dists: &mut Vec<ImageDataRgba8>,
                                   batch_frames: &mut Vec<usize>| -> Result<()> {
                if batch_refs.is_empty() {
                    return Ok(());
                }
                let frame_start = std::time::Instant::now();
                let scores_batch = metric
                    .compute_batch_rgba8(batch_refs, batch_dists)
                    .context("Failed to compute metric for batch")?;
                let frame_time = frame_start.elapsed();

                for (idx, score) in scores_batch.iter().enumerate() {
                    let frame_num = batch_frames[idx];
                    scores.push(*score);
                    frame_numbers.push(frame_num);
                    processed += 1;
                    frames_since_update += 1;
                    running_sum += *score;
                    last_score = *score;
                }

                let per_frame_ns = frame_time.as_nanos() as f64 / scores_batch.len() as f64;
                last_gpu_usage = metric.gpu_time_ns().and_then(|ns| {
                    if per_frame_ns > 0.0 {
                        Some((ns as f64 / per_frame_ns * 100.0).min(100.0))
                    } else {
                        None
                    }
                });

                let now = std::time::Instant::now();
                if now.duration_since(last_update).as_millis() >= 100 || processed == 1 {
                    let dt = now.duration_since(last_update).as_secs_f64();
                    if dt > 0.0 {
                        current_fps = frames_since_update as f64 / dt;
                    }
                    let avg_score = running_sum / processed as f64;
                    print_progress(processed, total_frames, current_fps, now.duration_since(start_time),
                                   last_score, avg_score, last_gpu_usage);
                    last_update = now;
                    frames_since_update = 0;
                }

                batch_refs.clear();
                batch_dists.clear();
                batch_frames.clear();

                Ok(())
            };

            while let Some(ref_frame) = ref_reader.decode_next()? {
                let dist_frame = match dist_reader.decode_next()? {
                    Some(f) => f,
                    None => {
                        log::warn!("Distorted video ended before reference");
                        break;
                    }
                };

                if current_frame < cli.start_frame {
                    current_frame += 1;
                    continue;
                }

                if current_frame >= effective_end {
                    break;
                }

                if (current_frame - cli.start_frame) % cli.frame_step == 0 {
                    batch_refs.push(ref_frame);
                    batch_dists.push(dist_frame);
                    batch_frames.push(current_frame);

                    if batch_refs.len() >= cli.batch_frames {
                        flush_batch(&mut batch_refs, &mut batch_dists, &mut batch_frames)?;
                    }
                }

                current_frame += 1;
            }

            flush_batch(&mut batch_refs, &mut batch_dists, &mut batch_frames)?;
        } else {
            let metric = metric.as_mut().unwrap();
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
                    let frame_start = std::time::Instant::now();
                    let score = match cli.input_format {
                        InputFormat::Rgba8 => metric.compute_rgba8(&ref_frame, &dist_frame),
                        InputFormat::F32 => {
                            let ref_f32 = ImageData::from_rgba8(
                                ref_frame.width,
                                ref_frame.height,
                                &ref_frame.data,
                            );
                            let dist_f32 = ImageData::from_rgba8(
                                dist_frame.width,
                                dist_frame.height,
                                &dist_frame.data,
                            );
                            match (ref_f32, dist_f32) {
                                (Ok(ref_f32), Ok(dist_f32)) => metric.compute(&ref_f32, &dist_f32),
                                (Err(err), _) | (_, Err(err)) => Err(err),
                            }
                        }
                    }
                    .context(format!("Failed to compute metric for frame {}", current_frame))?;
                    let frame_time = frame_start.elapsed();

                    scores.push(score);
                    frame_numbers.push(current_frame);
                    processed += 1;
                    frames_since_update += 1;
                    running_sum += score;
                    last_score = score;
                    last_gpu_usage = metric.gpu_time_ns().and_then(|ns| {
                        let total_ns = frame_time.as_nanos() as f64;
                        if total_ns > 0.0 {
                            Some((ns as f64 / total_ns * 100.0).min(100.0))
                        } else {
                            None
                        }
                    });

                    // Update progress every 100ms or every frame if slow
                    let now = std::time::Instant::now();
                    if now.duration_since(last_update).as_millis() >= 100 || processed == 1 {
                        let dt = now.duration_since(last_update).as_secs_f64();
                        if dt > 0.0 {
                            current_fps = frames_since_update as f64 / dt;
                        }
                        let avg_score = running_sum / processed as f64;
                        print_progress(processed, total_frames, current_fps, now.duration_since(start_time),
                                       last_score, avg_score, last_gpu_usage);
                        last_update = now;
                        frames_since_update = 0;
                    }
                }

                current_frame += 1;
            }
        }
    } else if !use_inflight {
        if use_batch {
            let metric = batch_metric.as_mut().unwrap();
            let mut processed = 0usize;
            let mut batch_refs: Vec<ImageDataRgba8> = Vec::with_capacity(cli.batch_frames);
            let mut batch_dists: Vec<ImageDataRgba8> = Vec::with_capacity(cli.batch_frames);
            let mut batch_frames: Vec<usize> = Vec::with_capacity(cli.batch_frames);

            let mut flush_batch = |batch_refs: &mut Vec<ImageDataRgba8>,
                                   batch_dists: &mut Vec<ImageDataRgba8>,
                                   batch_frames: &mut Vec<usize>| -> Result<()> {
                if batch_refs.is_empty() {
                    return Ok(());
                }
                let frame_start = std::time::Instant::now();
                let scores_batch = metric
                    .compute_batch_rgba8(batch_refs, batch_dists)
                    .context("Failed to compute metric for batch")?;
                let frame_time = frame_start.elapsed();

                for (idx, score) in scores_batch.iter().enumerate() {
                    let frame_num = batch_frames[idx];
                    scores.push(*score);
                    frame_numbers.push(frame_num);
                    processed += 1;
                    frames_since_update += 1;
                    running_sum += *score;
                    last_score = *score;
                }

                let per_frame_ns = frame_time.as_nanos() as f64 / scores_batch.len() as f64;
                last_gpu_usage = metric.gpu_time_ns().and_then(|ns| {
                    if per_frame_ns > 0.0 {
                        Some((ns as f64 / per_frame_ns * 100.0).min(100.0))
                    } else {
                        None
                    }
                });

                let now = std::time::Instant::now();
                if now.duration_since(last_update).as_millis() >= 100 || processed == 1 {
                    let dt = now.duration_since(last_update).as_secs_f64();
                    if dt > 0.0 {
                        current_fps = frames_since_update as f64 / dt;
                    }
                    let avg_score = running_sum / processed as f64;
                    print_progress(processed, total_frames, current_fps, now.duration_since(start_time),
                                   last_score, avg_score, last_gpu_usage);
                    last_update = now;
                    frames_since_update = 0;
                }

                batch_refs.clear();
                batch_dists.clear();
                batch_frames.clear();

                Ok(())
            };

            for frame_num in (cli.start_frame..effective_end).step_by(cli.frame_step) {
                let ref_frame = ref_reader.read_frame(frame_num)
                    .context(format!("Failed to read reference frame {}", frame_num))?;

                let dist_frame = dist_reader.read_frame(frame_num)
                    .context(format!("Failed to read distorted frame {}", frame_num))?;

                batch_refs.push(ref_frame);
                batch_dists.push(dist_frame);
                batch_frames.push(frame_num);

                if batch_refs.len() >= cli.batch_frames {
                    flush_batch(&mut batch_refs, &mut batch_dists, &mut batch_frames)?;
                }
            }

            flush_batch(&mut batch_refs, &mut batch_dists, &mut batch_frames)?;
        } else {
            let metric = metric.as_mut().unwrap();
            // Cached mode: frames already loaded
            for (i, frame_num) in (cli.start_frame..effective_end).step_by(cli.frame_step).enumerate() {
                let ref_frame = ref_reader.read_frame(frame_num)
                    .context(format!("Failed to read reference frame {}", frame_num))?;

                let dist_frame = dist_reader.read_frame(frame_num)
                    .context(format!("Failed to read distorted frame {}", frame_num))?;

                let frame_start = std::time::Instant::now();
                let score = match cli.input_format {
                    InputFormat::Rgba8 => metric.compute_rgba8(&ref_frame, &dist_frame),
                    InputFormat::F32 => {
                        let ref_f32 = ImageData::from_rgba8(
                            ref_frame.width,
                            ref_frame.height,
                            &ref_frame.data,
                        );
                        let dist_f32 = ImageData::from_rgba8(
                            dist_frame.width,
                            dist_frame.height,
                            &dist_frame.data,
                        );
                        match (ref_f32, dist_f32) {
                            (Ok(ref_f32), Ok(dist_f32)) => metric.compute(&ref_f32, &dist_f32),
                            (Err(err), _) | (_, Err(err)) => Err(err),
                        }
                    }
                }
                .context(format!("Failed to compute metric for frame {}", frame_num))?;
                let frame_time = frame_start.elapsed();

                scores.push(score);
                frame_numbers.push(frame_num);
                frames_since_update += 1;
                running_sum += score;
                last_score = score;
                last_gpu_usage = metric.gpu_time_ns().and_then(|ns| {
                    let total_ns = frame_time.as_nanos() as f64;
                    if total_ns > 0.0 {
                        Some((ns as f64 / total_ns * 100.0).min(100.0))
                    } else {
                        None
                    }
                });

                // Update progress every 100ms or every frame if slow
                let now = std::time::Instant::now();
                if now.duration_since(last_update).as_millis() >= 100 || i == 0 {
                    let dt = now.duration_since(last_update).as_secs_f64();
                    if dt > 0.0 {
                        current_fps = frames_since_update as f64 / dt;
                    }
                    let avg_score = running_sum / (i + 1) as f64;
                    print_progress(i + 1, total_frames, current_fps, now.duration_since(start_time),
                                   last_score, avg_score, last_gpu_usage);
                    last_update = now;
                    frames_since_update = 0;
                }
            }
        }
    } else {
        let max_workers = if adaptive_inflight {
            cli.in_flight_max.max(1)
        } else {
            cli.in_flight.max(1)
        };
        let mut active_workers = if adaptive_inflight { 1 } else { max_workers };
        let gpu_sum = Cell::new(0.0f64);
        let gpu_samples = Cell::new(0usize);
        let gpu_target_low = 55.0f64;
        let gpu_target_high = 80.0f64;
        let adjust_after = 60usize;

        let (job_senders, result_rx) = spawn_workers(
            std::sync::Arc::clone(&ctx),
            cli.metric,
            cli.gpu_mode,
            cli.reduce_mode,
            cli.input_format,
            max_workers,
        );
        let mut next_worker = 0;
        let mut pending = 0usize;
        let mut processed = 0usize;
        let mut results: Vec<(usize, f64)> = Vec::with_capacity(total_frames);

        let mut handle_result = |result: JobResult| -> Result<()> {
            if let Some(err) = result.error {
                anyhow::bail!("Failed to compute metric for frame {}: {}", result.frame_num, err);
            }
            let score = result.score.unwrap_or(0.0);
            results.push((result.frame_num, score));
            processed += 1;
            frames_since_update += 1;
            running_sum += score;
            last_score = score;
            if result.wall_ns > 0 && result.gpu_time_ns > 0 {
                let usage = ((result.gpu_time_ns as f64 / result.wall_ns as f64) * 100.0).min(100.0);
                last_gpu_usage = Some(usage);
                if adaptive_inflight {
                    gpu_sum.set(gpu_sum.get() + usage);
                    gpu_samples.set(gpu_samples.get() + 1);
                }
            }

            let now = std::time::Instant::now();
            if now.duration_since(last_update).as_millis() >= 100 || processed == 1 {
                let dt = now.duration_since(last_update).as_secs_f64();
                if dt > 0.0 {
                    current_fps = frames_since_update as f64 / dt;
                }
                let avg_score = running_sum / processed as f64;
                print_progress(processed, total_frames, current_fps, now.duration_since(start_time),
                               last_score, avg_score, last_gpu_usage);
                last_update = now;
                frames_since_update = 0;
            }
            Ok(())
        };

        #[cfg(feature = "ffmpeg")]
        if use_streaming {
            let mut current_frame = 0usize;
            while let Some(ref_frame) = ref_reader.decode_next()? {
                let dist_frame = match dist_reader.decode_next()? {
                    Some(f) => f,
                    None => {
                        log::warn!("Distorted video ended before reference");
                        break;
                    }
                };

                if current_frame < cli.start_frame {
                    current_frame += 1;
                    continue;
                }

                if current_frame >= effective_end {
                    break;
                }

                if (current_frame - cli.start_frame) % cli.frame_step == 0 {
                    let job = Job {
                        frame_num: current_frame,
                        ref_frame,
                        dist_frame,
                    };
                    job_senders[next_worker].send(job).map_err(|_| {
                        anyhow::anyhow!("Worker channel closed while submitting frame {}", current_frame)
                    })?;
                    next_worker = (next_worker + 1) % active_workers;
                    pending += 1;

                    if pending >= active_workers {
                        let result = result_rx.recv().context("Worker channel closed unexpectedly")?;
                        handle_result(result)?;
                        if adaptive_inflight && gpu_samples.get() >= adjust_after {
                            let avg = gpu_sum.get() / gpu_samples.get() as f64;
                            if avg < gpu_target_low && active_workers < max_workers {
                                active_workers += 1;
                                log::info!("Increasing in-flight to {}", active_workers);
                            } else if avg > gpu_target_high && active_workers > 1 {
                                active_workers -= 1;
                                log::info!("Decreasing in-flight to {}", active_workers);
                            }
                            next_worker %= active_workers;
                            gpu_sum.set(0.0);
                            gpu_samples.set(0);
                        }
                        pending -= 1;
                    }
                }

                current_frame += 1;
            }
        } else {
            for frame_num in (cli.start_frame..effective_end).step_by(cli.frame_step) {
                let ref_frame = ref_reader.read_frame(frame_num)
                    .context(format!("Failed to read reference frame {}", frame_num))?;
                let dist_frame = dist_reader.read_frame(frame_num)
                    .context(format!("Failed to read distorted frame {}", frame_num))?;
                let job = Job {
                    frame_num,
                    ref_frame,
                    dist_frame,
                };
                job_senders[next_worker].send(job).map_err(|_| {
                    anyhow::anyhow!("Worker channel closed while submitting frame {}", frame_num)
                })?;
                next_worker = (next_worker + 1) % active_workers;
                pending += 1;

                if pending >= active_workers {
                    let result = result_rx.recv().context("Worker channel closed unexpectedly")?;
                    handle_result(result)?;
                    if adaptive_inflight && gpu_samples.get() >= adjust_after {
                        let avg = gpu_sum.get() / gpu_samples.get() as f64;
                        if avg < gpu_target_low && active_workers < max_workers {
                            active_workers += 1;
                            log::info!("Increasing in-flight to {}", active_workers);
                        } else if avg > gpu_target_high && active_workers > 1 {
                            active_workers -= 1;
                            log::info!("Decreasing in-flight to {}", active_workers);
                        }
                        next_worker %= active_workers;
                        gpu_sum.set(0.0);
                        gpu_samples.set(0);
                    }
                    pending -= 1;
                }
            }
        }

        while pending > 0 {
            let result = result_rx.recv().context("Worker channel closed unexpectedly")?;
            handle_result(result)?;
            if adaptive_inflight && gpu_samples.get() >= adjust_after {
                let avg = gpu_sum.get() / gpu_samples.get() as f64;
                if avg < gpu_target_low && active_workers < max_workers {
                    active_workers += 1;
                    log::info!("Increasing in-flight to {}", active_workers);
                } else if avg > gpu_target_high && active_workers > 1 {
                    active_workers -= 1;
                    log::info!("Decreasing in-flight to {}", active_workers);
                }
                next_worker %= active_workers;
                gpu_sum.set(0.0);
                gpu_samples.set(0);
            }
            pending -= 1;
        }

        results.sort_by_key(|(frame, _)| *frame);
        scores = results.iter().map(|(_, score)| *score).collect();
        frame_numbers = results.iter().map(|(frame, _)| *frame).collect();
    }

    #[cfg(not(feature = "ffmpeg"))]
    if !use_inflight {
        if use_batch {
            let metric = batch_metric.as_mut().unwrap();
            let mut processed = 0usize;
            let mut batch_refs: Vec<ImageDataRgba8> = Vec::with_capacity(cli.batch_frames);
            let mut batch_dists: Vec<ImageDataRgba8> = Vec::with_capacity(cli.batch_frames);
            let mut batch_frames: Vec<usize> = Vec::with_capacity(cli.batch_frames);

            let mut flush_batch = |batch_refs: &mut Vec<ImageDataRgba8>,
                                   batch_dists: &mut Vec<ImageDataRgba8>,
                                   batch_frames: &mut Vec<usize>| -> Result<()> {
                if batch_refs.is_empty() {
                    return Ok(());
                }
                let frame_start = std::time::Instant::now();
                let scores_batch = metric
                    .compute_batch_rgba8(batch_refs, batch_dists)
                    .context("Failed to compute metric for batch")?;
                let frame_time = frame_start.elapsed();

                for (idx, score) in scores_batch.iter().enumerate() {
                    let frame_num = batch_frames[idx];
                    scores.push(*score);
                    frame_numbers.push(frame_num);
                    processed += 1;
                    frames_since_update += 1;
                    running_sum += *score;
                    last_score = *score;
                }

                let per_frame_ns = frame_time.as_nanos() as f64 / scores_batch.len() as f64;
                last_gpu_usage = metric.gpu_time_ns().and_then(|ns| {
                    if per_frame_ns > 0.0 {
                        Some((ns as f64 / per_frame_ns * 100.0).min(100.0))
                    } else {
                        None
                    }
                });

                let now = std::time::Instant::now();
                if now.duration_since(last_update).as_millis() >= 100 || processed == 1 {
                    let dt = now.duration_since(last_update).as_secs_f64();
                    if dt > 0.0 {
                        current_fps = frames_since_update as f64 / dt;
                    }
                    let avg_score = running_sum / processed as f64;
                    print_progress(processed, total_frames, current_fps, now.duration_since(start_time),
                                   last_score, avg_score, last_gpu_usage);
                    last_update = now;
                    frames_since_update = 0;
                }

                batch_refs.clear();
                batch_dists.clear();
                batch_frames.clear();

                Ok(())
            };

            for frame_num in (cli.start_frame..effective_end).step_by(cli.frame_step) {
                let ref_frame = ref_reader.read_frame(frame_num)
                    .context(format!("Failed to read reference frame {}", frame_num))?;

                let dist_frame = dist_reader.read_frame(frame_num)
                    .context(format!("Failed to read distorted frame {}", frame_num))?;

                batch_refs.push(ref_frame);
                batch_dists.push(dist_frame);
                batch_frames.push(frame_num);

                if batch_refs.len() >= cli.batch_frames {
                    flush_batch(&mut batch_refs, &mut batch_dists, &mut batch_frames)?;
                }
            }

            flush_batch(&mut batch_refs, &mut batch_dists, &mut batch_frames)?;
        } else {
            let metric = metric.as_mut().unwrap();
            for (i, frame_num) in (cli.start_frame..effective_end).step_by(cli.frame_step).enumerate() {
                let ref_frame = ref_reader.read_frame(frame_num)
                    .context(format!("Failed to read reference frame {}", frame_num))?;

                let dist_frame = dist_reader.read_frame(frame_num)
                    .context(format!("Failed to read distorted frame {}", frame_num))?;

                let frame_start = std::time::Instant::now();
                let score = match cli.input_format {
                    InputFormat::Rgba8 => metric.compute_rgba8(&ref_frame, &dist_frame),
                    InputFormat::F32 => {
                        let ref_f32 = ImageData::from_rgba8(
                            ref_frame.width,
                            ref_frame.height,
                            &ref_frame.data,
                        );
                        let dist_f32 = ImageData::from_rgba8(
                            dist_frame.width,
                            dist_frame.height,
                            &dist_frame.data,
                        );
                        match (ref_f32, dist_f32) {
                            (Ok(ref_f32), Ok(dist_f32)) => metric.compute(&ref_f32, &dist_f32),
                            (Err(err), _) | (_, Err(err)) => Err(err),
                        }
                    }
                }
                .context(format!("Failed to compute metric for frame {}", frame_num))?;
                let frame_time = frame_start.elapsed();

                scores.push(score);
                frame_numbers.push(frame_num);
                frames_since_update += 1;
                running_sum += score;
                last_score = score;
                last_gpu_usage = metric.gpu_time_ns().and_then(|ns| {
                    let total_ns = frame_time.as_nanos() as f64;
                    if total_ns > 0.0 {
                        Some((ns as f64 / total_ns * 100.0).min(100.0))
                    } else {
                        None
                    }
                });

                // Update progress
                let now = std::time::Instant::now();
                if now.duration_since(last_update).as_millis() >= 100 || i == 0 {
                    let dt = now.duration_since(last_update).as_secs_f64();
                    if dt > 0.0 {
                        current_fps = frames_since_update as f64 / dt;
                    }
                    let avg_score = running_sum / (i + 1) as f64;
                    print_progress(i + 1, total_frames, current_fps, now.duration_since(start_time),
                                   last_score, avg_score, last_gpu_usage);
                    last_update = now;
                    frames_since_update = 0;
                }
            }
        }
    } else {
        let max_workers = if adaptive_inflight {
            cli.in_flight_max.max(1)
        } else {
            cli.in_flight.max(1)
        };
        let mut active_workers = if adaptive_inflight { 1 } else { max_workers };
        let gpu_sum = Cell::new(0.0f64);
        let gpu_samples = Cell::new(0usize);
        let gpu_target_low = 55.0f64;
        let gpu_target_high = 80.0f64;
        let adjust_after = 60usize;

        let (job_senders, result_rx) = spawn_workers(
            std::sync::Arc::clone(&ctx),
            cli.metric,
            cli.gpu_mode,
            cli.reduce_mode,
            cli.input_format,
            max_workers,
        );
        let mut next_worker = 0;
        let mut pending = 0usize;
        let mut processed = 0usize;
        let mut results: Vec<(usize, f64)> = Vec::with_capacity(total_frames);

        let mut handle_result = |result: JobResult| -> Result<()> {
            if let Some(err) = result.error {
                anyhow::bail!("Failed to compute metric for frame {}: {}", result.frame_num, err);
            }
            let score = result.score.unwrap_or(0.0);
            results.push((result.frame_num, score));
            processed += 1;
            frames_since_update += 1;
            running_sum += score;
            last_score = score;
            if result.wall_ns > 0 && result.gpu_time_ns > 0 {
                let usage = ((result.gpu_time_ns as f64 / result.wall_ns as f64) * 100.0).min(100.0);
                last_gpu_usage = Some(usage);
                if adaptive_inflight {
                    gpu_sum.set(gpu_sum.get() + usage);
                    gpu_samples.set(gpu_samples.get() + 1);
                }
            }

            let now = std::time::Instant::now();
            if now.duration_since(last_update).as_millis() >= 100 || processed == 1 {
                let dt = now.duration_since(last_update).as_secs_f64();
                if dt > 0.0 {
                    current_fps = frames_since_update as f64 / dt;
                }
                let avg_score = running_sum / processed as f64;
                print_progress(processed, total_frames, current_fps, now.duration_since(start_time),
                               last_score, avg_score, last_gpu_usage);
                last_update = now;
                frames_since_update = 0;
            }
            Ok(())
        };

        for frame_num in (cli.start_frame..effective_end).step_by(cli.frame_step) {
            let ref_frame = ref_reader.read_frame(frame_num)
                .context(format!("Failed to read reference frame {}", frame_num))?;
            let dist_frame = dist_reader.read_frame(frame_num)
                .context(format!("Failed to read distorted frame {}", frame_num))?;
            let job = Job {
                frame_num,
                ref_frame,
                dist_frame,
            };
            job_senders[next_worker].send(job).map_err(|_| {
                anyhow::anyhow!("Worker channel closed while submitting frame {}", frame_num)
            })?;
            next_worker = (next_worker + 1) % active_workers;
            pending += 1;

            if pending >= active_workers {
                let result = result_rx.recv().context("Worker channel closed unexpectedly")?;
                handle_result(result)?;
                if adaptive_inflight && gpu_samples.get() >= adjust_after {
                    let avg = gpu_sum.get() / gpu_samples.get() as f64;
                    if avg < gpu_target_low && active_workers < max_workers {
                        active_workers += 1;
                        log::info!("Increasing in-flight to {}", active_workers);
                    } else if avg > gpu_target_high && active_workers > 1 {
                        active_workers -= 1;
                        log::info!("Decreasing in-flight to {}", active_workers);
                    }
                    next_worker %= active_workers;
                    gpu_sum.set(0.0);
                    gpu_samples.set(0);
                }
                pending -= 1;
            }
        }

        while pending > 0 {
            let result = result_rx.recv().context("Worker channel closed unexpectedly")?;
            handle_result(result)?;
            if adaptive_inflight && gpu_samples.get() >= adjust_after {
                let avg = gpu_sum.get() / gpu_samples.get() as f64;
                if avg < gpu_target_low && active_workers < max_workers {
                    active_workers += 1;
                    log::info!("Increasing in-flight to {}", active_workers);
                } else if avg > gpu_target_high && active_workers > 1 {
                    active_workers -= 1;
                    log::info!("Decreasing in-flight to {}", active_workers);
                }
                next_worker %= active_workers;
                gpu_sum.set(0.0);
                gpu_samples.set(0);
            }
            pending -= 1;
        }

        results.sort_by_key(|(frame, _)| *frame);
        scores = results.iter().map(|(_, score)| *score).collect();
        frame_numbers = results.iter().map(|(frame, _)| *frame).collect();
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
    println!("{} Statistics:", metric_name);
    println!("  Mean:   {:.4}", mean_score);
    println!("  Min:    {:.4}", min_score);
    println!("  Max:    {:.4}", max_score);
    println!("  StdDev: {:.4}", std_dev);
    println!("  Frames: {}", scores.len());

    // Save results if output file specified
    if let Some(output_path) = cli.output {
        let results = serde_json::json!({
            "metric": metric_name,
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
