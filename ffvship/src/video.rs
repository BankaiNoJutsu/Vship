// Video file reading and decoding using FFmpeg

use anyhow::{Result, Context};
use vship_metrics::{ImageData, ImageFormat};
use std::path::Path;

#[cfg(feature = "ffmpeg")]
use crate::ffmpeg_decoder::FfmpegDecoder;

/// Video reader wrapper
pub struct VideoReader {
    #[cfg(feature = "ffmpeg")]
    decoder: FfmpegDecoder,

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
            Ok(Self { decoder })
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
        { self.decoder.frame_count() }

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

    /// Read a specific frame
    pub fn read_frame(&mut self, frame_num: usize) -> Result<ImageData> {
        #[cfg(feature = "ffmpeg")]
        {
            self.decoder.read_frame(frame_num)
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
            self.decoder.read_next_frame()
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
