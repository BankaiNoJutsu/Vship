use thiserror::Error;

#[derive(Error, Debug)]
pub enum VshipError {
    #[error("Vulkan error: {0}")]
    VulkanError(#[from] ash::vk::Result),

    #[error("Failed to load Vulkan library: {0}")]
    LoadingError(#[from] ash::LoadingError),

    #[error("No compatible Vulkan device found")]
    NoDeviceFound,

    #[error("Device does not support compute operations")]
    NoComputeQueue,

    #[error("Shader compilation failed: {0}")]
    ShaderCompilationError(String),

    #[error("Invalid buffer size: expected {expected}, got {actual}")]
    InvalidBufferSize { expected: usize, actual: usize },

    #[error("Invalid color space: {0}")]
    InvalidColorSpace(String),

    #[error("Invalid image dimensions: {width}x{height}")]
    InvalidDimensions { width: u32, height: u32 },

    #[error("Memory allocation failed: {0}")]
    AllocationError(String),

    #[error("GPU memory allocator error: {0}")]
    GpuAllocatorError(#[from] gpu_allocator::AllocationError),

    #[error("Pipeline creation failed: {0}")]
    PipelineError(String),

    #[error("Command buffer error: {0}")]
    CommandBufferError(String),

    #[error("Synchronization error: {0}")]
    SyncError(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Other error: {0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, VshipError>;
