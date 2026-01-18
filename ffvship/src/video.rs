// Video file reading and decoding using FFmpeg

use anyhow::{Result, Context};
use vship_metrics::{ImageData, ImageFormat};
use std::path::Path;

#[cfg(feature = "ffmpeg")]
use crate::ffmpeg_decoder::FfmpegDecoder;

/// Video reader wrapper with streaming support
pub struct VideoReader {
    #[cfg(feature = "ffmpeg")]
    decoder: FfmpegDecoder,
    #[cfg(feature = "ffmpeg")]
    frame_cache: Option<Vec<ImageData>>,
    #[cfg(feature = "ffmpeg")]
    cache_start_frame: usize,
    #[cfg(feature = "ffmpeg")]
    streaming_mode: bool,

    #[cfg(not(feature = "ffmpeg"))]
    width: u32,
    #[cfg(not(feature = "ffmpeg"))]
    height: u32,
    #[cfg(not(feature = "ffmpeg"))]
    frame_count: usize,
    #[cfg(not(feature = "ffmpeg"))]
    fps: f64,
}

impl VideoReader {
    /// Open a video file
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();

        #[cfg(feature = "ffmpeg")]
        {
            let decoder = FfmpegDecoder::open(path)?;
            Ok(Self {
                decoder,
                frame_cache: None,
                cache_start_frame: 0,
                streaming_mode: false,
            })
        }

        #[cfg(not(feature = "ffmpeg"))]
        {
            log::warn!("FFmpeg integration not enabled, using placeholder");
            log::info!("Video file: {:?}", path);
            log::info!("To enable FFmpeg, see ffvship/src/ffmpeg_decoder.rs");

            Ok(Self {
                width: 1920,
                height: 1080,
                frame_count: 100,
                fps: 24.0,
            })
        }
    }

    /// Get video width
    pub fn width(&self) -> u32 {
        #[cfg(feature = "ffmpeg")]
        { self.decoder.width() }

        #[cfg(not(feature = "ffmpeg"))]
        { self.width }
    }

    /// Get video height
    pub fn height(&self) -> u32 {
        #[cfg(feature = "ffmpeg")]
        { self.decoder.height() }

        #[cfg(not(feature = "ffmpeg"))]
        { self.height }
    }

    /// Get total frame count
    pub fn frame_count(&self) -> usize {
        #[cfg(feature = "ffmpeg")]
        {
            if let Some(ref cache) = self.frame_cache {
                cache.len()
            } else {
                self.decoder.frame_count()
            }
        }

        #[cfg(not(feature = "ffmpeg"))]
        { self.frame_count }
    }

    /// Get frames per second
    pub fn fps(&self) -> f64 {
        #[cfg(feature = "ffmpeg")]
        { self.decoder.fps() }

        #[cfg(not(feature = "ffmpeg"))]
        { self.fps }
    }

    /// Load a range of frames into cache (call this once before processing)
    /// For large videos, consider using streaming mode instead
    #[cfg(feature = "ffmpeg")]
    pub fn load_frame_range(&mut self, start: usize, end: usize) -> Result<()> {
        // Check if the range is too large (more than 500 frames or ~12GB for 1080p)
        let frame_count = end.saturating_sub(start);
        if frame_count > 500 {
            log::info!("Large frame range detected ({}), using streaming mode", frame_count);
            self.enable_streaming()?;
            return Ok(());
        }

        if self.frame_cache.is_none() {
            let frames = self.decoder.decode_frame_range(start, end)?;
            self.cache_start_frame = start;
            self.frame_cache = Some(frames);
        }
        Ok(())
    }

    /// Enable streaming mode (decode frames on-demand, no caching)
    #[cfg(feature = "ffmpeg")]
    pub fn enable_streaming(&mut self) -> Result<()> {
        self.streaming_mode = true;
        self.frame_cache = None;
        self.decoder.init_streaming()?;
        Ok(())
    }

    /// Decode the next frame in streaming mode
    #[cfg(feature = "ffmpeg")]
    pub fn decode_next(&mut self) -> Result<Option<ImageData>> {
        if !self.streaming_mode {
            anyhow::bail!("Not in streaming mode. Call enable_streaming() first");
        }
        self.decoder.decode_next_frame()
    }

    /// Check if in streaming mode
    #[cfg(feature = "ffmpeg")]
    pub fn is_streaming(&self) -> bool {
        self.streaming_mode
    }

    /// Reset streaming position
    #[cfg(feature = "ffmpeg")]
    pub fn reset_streaming(&mut self) {
        self.decoder.reset_streaming();
    }

    /// Read a specific frame
    pub fn read_frame(&mut self, frame_num: usize) -> Result<ImageData> {
        #[cfg(feature = "ffmpeg")]
        {
            let cache = self.frame_cache.as_ref()
                .context("Frames not loaded. Call load_frame_range() first")?;

            // Convert absolute frame number to cache index
            if frame_num < self.cache_start_frame {
                anyhow::bail!("Frame {} not in cache (cache starts at {})", frame_num, self.cache_start_frame);
            }

            let cache_idx = frame_num - self.cache_start_frame;
            if cache_idx >= cache.len() {
                anyhow::bail!("Frame {} out of range (cache has {} frames starting at {})",
                             frame_num, cache.len(), self.cache_start_frame);
            }

            Ok(cache[cache_idx].clone())
        }

        #[cfg(not(feature = "ffmpeg"))]
        {
            if frame_num >= self.frame_count {
                anyhow::bail!("Frame {} out of range (total: {})", frame_num, self.frame_count);
            }
            // Return placeholder black frame
            Ok(ImageData::new(self.width, self.height, ImageFormat::RGB))
        }
    }

    /// Read next frame in sequence
    pub fn read_next_frame(&mut self) -> Result<Option<ImageData>> {
        #[cfg(feature = "ffmpeg")]
        {
            // Not implemented with cache-based approach
            // Use read_frame() with explicit frame numbers instead
            Ok(None)
        }

        #[cfg(not(feature = "ffmpeg"))]
        {
            Ok(None)
        }
    }
}

/// Frame iterator for sequential video processing
pub struct FrameIterator {
    reader: VideoReader,
    current_frame: usize,
    end_frame: usize,
    step: usize,
}

impl FrameIterator {
    /// Create a new frame iterator
    pub fn new(reader: VideoReader, start: usize, end: usize, step: usize) -> Self {
        let end_frame = if end == 0 { reader.frame_count() } else { end };

        Self {
            reader,
            current_frame: start,
            end_frame,
            step,
        }
    }

    /// Get video info
    pub fn video_info(&self) -> VideoInfo {
        VideoInfo {
            width: self.reader.width(),
            height: self.reader.height(),
            frame_count: self.reader.frame_count(),
            fps: self.reader.fps(),
        }
    }
}

impl Iterator for FrameIterator {
    type Item = Result<(usize, ImageData)>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_frame >= self.end_frame {
            return None;
        }

        let frame_num = self.current_frame;
        let result = self.reader.read_frame(frame_num)
            .map(|data| (frame_num, data));

        self.current_frame += self.step;

        Some(result)
    }
}

/// Video information
#[derive(Debug, Clone)]
pub struct VideoInfo {
    pub width: u32,
    pub height: u32,
    pub frame_count: usize,
    pub fps: f64,
}

impl std::fmt::Display for VideoInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}x{} @ {:.2} fps ({} frames)",
            self.width, self.height, self.fps, self.frame_count
        )
    }
}

/// Process two videos frame-by-frame with a metric
pub fn process_video_pair<F>(
    reference_path: &Path,
    distorted_path: &Path,
    start_frame: usize,
    end_frame: usize,
    frame_step: usize,
    mut metric_fn: F,
) -> Result<Vec<f64>>
where
    F: FnMut(&ImageData, &ImageData) -> Result<f64>,
{
    let mut ref_reader = VideoReader::open(reference_path)?;
    let mut dist_reader = VideoReader::open(distorted_path)?;

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

    let mut scores = Vec::new();
    let effective_end = if end_frame == 0 {
        ref_reader.frame_count().min(dist_reader.frame_count())
    } else {
        end_frame
    };

    for frame_num in (start_frame..effective_end).step_by(frame_step) {
        let ref_frame = ref_reader.read_frame(frame_num)
            .context(format!("Failed to read reference frame {}", frame_num))?;

        let dist_frame = dist_reader.read_frame(frame_num)
            .context(format!("Failed to read distorted frame {}", frame_num))?;

        let score = metric_fn(&ref_frame, &dist_frame)
            .context(format!("Failed to compute metric for frame {}", frame_num))?;

        scores.push(score);
    }

    Ok(scores)
}

// TODO: Implement actual FFmpeg integration
// The current implementation is a placeholder that will be replaced with
// actual ffmpeg-next bindings once the core metric functionality is working.
//
// FFmpeg integration will include:
// - Hardware-accelerated decoding where available
// - Format conversion (YUV to RGB)
// - Color space handling
// - Efficient frame seeking
