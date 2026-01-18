// FFmpeg video decoder implementation

use anyhow::{Result, Context};
use vship_metrics::ImageDataRgba8;
use std::path::Path;

/// FFmpeg-based video decoder with streaming support
pub struct FfmpegDecoder {
    width: u32,
    height: u32,
    frame_count: usize,
    fps: f64,
    input_path: std::path::PathBuf,
    // Streaming state
    current_frame: usize,
    input: Option<ffmpeg_next::format::context::Input>,
    decoder: Option<ffmpeg_next::decoder::Video>,
    video_stream_index: usize,
    scaler: Option<ffmpeg_next::software::scaling::Context>,
    rgba_frame: Option<ffmpeg_next::util::frame::Video>,
}

impl FfmpegDecoder {
    /// Open a video file
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();

        // Initialize FFmpeg (idempotent)
        ffmpeg_next::init()
            .context("Failed to initialize FFmpeg")?;

        // Open input file to get metadata
        let input = ffmpeg_next::format::input(&path)
            .context("Failed to open video file")?;

        // Find the video stream
        let stream = input
            .streams()
            .best(ffmpeg_next::media::Type::Video)
            .context("No video stream found in file")?;

        let context_decoder = ffmpeg_next::codec::context::Context::from_parameters(stream.parameters())
            .context("Failed to create codec context")?;

        let decoder = context_decoder
            .decoder()
            .video()
            .context("Failed to create video decoder")?;

        let width = decoder.width();
        let height = decoder.height();

        // Get frame rate
        let avg_frame_rate = stream.avg_frame_rate();
        let fps = avg_frame_rate.numerator() as f64 / avg_frame_rate.denominator() as f64;

        // Estimate frame count from duration and fps
        let duration_secs = input.duration() as f64 / f64::from(ffmpeg_next::ffi::AV_TIME_BASE);
        let frame_count = (duration_secs * fps) as usize;

        log::info!("Opened video: {}x{} @ {:.2} fps (~{} frames)",
                   width, height, fps, frame_count);

        Ok(Self {
            width,
            height,
            frame_count,
            fps,
            input_path: path.to_path_buf(),
            current_frame: 0,
            input: None,
            decoder: None,
            video_stream_index: 0,
            scaler: None,
            rgba_frame: None,
        })
    }

    /// Initialize streaming decoder (call before streaming frames)
    pub fn init_streaming(&mut self) -> Result<()> {
        if self.input.is_some() {
            return Ok(()); // Already initialized
        }

        let input = ffmpeg_next::format::input(&self.input_path)
            .context("Failed to open video file for streaming")?;

        let stream = input
            .streams()
            .best(ffmpeg_next::media::Type::Video)
            .context("No video stream found")?;

        self.video_stream_index = stream.index();

        let context_decoder = ffmpeg_next::codec::context::Context::from_parameters(stream.parameters())
            .context("Failed to create codec context")?;

        let decoder = context_decoder
            .decoder()
            .video()
            .context("Failed to create video decoder")?;

        self.input = Some(input);
        self.decoder = Some(decoder);
        self.current_frame = 0;

        Ok(())
    }

    /// Decode the next frame in sequence (streaming mode)
    pub fn decode_next_frame(&mut self) -> Result<Option<ImageDataRgba8>> {
        self.init_streaming()?;

        let input = self.input.as_mut().unwrap();
        let decoder = self.decoder.as_mut().unwrap();

        // Try to get a frame from already-sent packets
        let mut decoded_frame = ffmpeg_next::util::frame::Video::empty();
        if decoder.receive_frame(&mut decoded_frame).is_ok() {
            self.current_frame += 1;
            return Ok(Some(self.convert_frame_to_rgba8(&decoded_frame)?));
        }

        // Need to send more packets
        for (stream, packet) in input.packets() {
            if stream.index() == self.video_stream_index {
                decoder.send_packet(&packet)
                    .context("Failed to send packet to decoder")?;

                if decoder.receive_frame(&mut decoded_frame).is_ok() {
                    self.current_frame += 1;
                    return Ok(Some(self.convert_frame_to_rgba8(&decoded_frame)?));
                }
            }
        }

        // Flush decoder
        decoder.send_eof().ok();
        if decoder.receive_frame(&mut decoded_frame).is_ok() {
            self.current_frame += 1;
            return Ok(Some(self.convert_frame_to_rgba8(&decoded_frame)?));
        }

        Ok(None) // End of video
    }

    /// Get current frame position in streaming mode
    pub fn current_frame_position(&self) -> usize {
        self.current_frame
    }

    /// Reset streaming to beginning
    pub fn reset_streaming(&mut self) {
        self.input = None;
        self.decoder = None;
        self.current_frame = 0;
    }

    /// Get video width
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Get video height
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Get total frame count
    pub fn frame_count(&self) -> usize {
        self.frame_count
    }

    /// Get frames per second
    pub fn fps(&self) -> f64 {
        self.fps
    }

    /// Decode a range of frames (start..end)
    /// Much more memory-efficient than decoding all frames
    pub fn decode_frame_range(&mut self, start: usize, end: usize) -> Result<Vec<ImageDataRgba8>> {
        let actual_end = end.min(self.frame_count);
        let num_frames = actual_end.saturating_sub(start);

        log::info!("Decoding frames {} to {} ({} frames)...", start, actual_end, num_frames);

        let mut input = ffmpeg_next::format::input(&self.input_path)
            .context("Failed to open video file")?;

        let stream = input
            .streams()
            .best(ffmpeg_next::media::Type::Video)
            .context("No video stream found")?;

        let video_stream_index = stream.index();

        let context_decoder = ffmpeg_next::codec::context::Context::from_parameters(stream.parameters())
            .context("Failed to create codec context")?;

        let mut decoder = context_decoder
            .decoder()
            .video()
            .context("Failed to create video decoder")?;

        let mut frames = Vec::new();
        frames.reserve(num_frames);

        let mut current_frame_idx = 0;

        // Process packets
        for (stream, packet) in input.packets() {
            if stream.index() == video_stream_index {
                decoder.send_packet(&packet)
                    .context("Failed to send packet to decoder")?;

                let mut decoded_frame = ffmpeg_next::util::frame::Video::empty();

                while decoder.receive_frame(&mut decoded_frame).is_ok() {
                    // Only keep frames in the requested range
                    if current_frame_idx >= start && current_frame_idx < actual_end {
                        let image = self.convert_frame_to_rgba8(&decoded_frame)?;
                        frames.push(image);

                        if frames.len() % 50 == 0 {
                            log::info!("Decoded {} / {} frames...", frames.len(), num_frames);
                        }
                    }

                    current_frame_idx += 1;

                    // Stop if we've decoded all frames we need
                    if current_frame_idx >= actual_end {
                        log::info!("Decoded {} total frames", frames.len());
                        return Ok(frames);
                    }
                }
            }
        }

        // Flush decoder
        decoder.send_eof().ok();
        let mut decoded_frame = ffmpeg_next::util::frame::Video::empty();
        while decoder.receive_frame(&mut decoded_frame).is_ok() {
            if current_frame_idx >= start && current_frame_idx < actual_end {
                let image = self.convert_frame_to_rgba8(&decoded_frame)?;
                frames.push(image);
            }
            current_frame_idx += 1;

            if current_frame_idx >= actual_end {
                break;
            }
        }

        log::info!("Decoded {} total frames", frames.len());
        Ok(frames)
    }

    /// Convert an FFmpeg frame to packed RGBA8
    fn convert_frame_to_rgba8(
        &mut self,
        frame: &ffmpeg_next::util::frame::Video,
    ) -> Result<ImageDataRgba8> {
        if self.scaler.is_none() {
            let scaler = ffmpeg_next::software::scaling::Context::get(
                frame.format(),
                frame.width(),
                frame.height(),
                ffmpeg_next::util::format::Pixel::RGBA,
                self.width,
                self.height,
                ffmpeg_next::software::scaling::Flags::BILINEAR,
            ).context("Failed to create scaler")?;
            self.scaler = Some(scaler);
        }

        let rgba_frame = self.rgba_frame.get_or_insert_with(ffmpeg_next::util::frame::Video::empty);
        self.scaler
            .as_mut()
            .unwrap()
            .run(frame, rgba_frame)
            .context("Failed to scale frame")?;

        let data = rgba_frame.data(0);
        let stride = rgba_frame.stride(0) as usize;
        let row_bytes = self.width as usize * 4;

        let mut image = ImageDataRgba8::new(self.width, self.height);
        for y in 0..self.height as usize {
            let src = y * stride;
            let dst = y * row_bytes;
            image.data[dst..dst + row_bytes]
                .copy_from_slice(&data[src..src + row_bytes]);
        }

        Ok(image)
    }
}

/// FFmpeg initialization
pub fn init_ffmpeg() -> Result<()> {
    ffmpeg_next::init()
        .context("Failed to initialize FFmpeg")?;

    log::info!("FFmpeg initialized successfully");

    Ok(())
}
