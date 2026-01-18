// C FFI bindings for Vship
// Provides a C-compatible API for interoperability with VapourSynth and other C code

use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_double, c_int, c_uint, c_void};
use std::ptr;
use std::sync::Arc;
use vship_core::{VshipContext, error::VshipError};
use vship_metrics::{MetricsContext, Metric, ImageData, ImageFormat};
use vship_metrics::{Ssimulacra2, Butteraugli, Cvvdp};

// Opaque handle types
#[repr(C)]
pub struct VshipHandle {
    _private: [u8; 0],
}

#[repr(C)]
pub struct VshipMetricHandle {
    _private: [u8; 0],
}

// Error codes
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum VshipErrorCode {
    Success = 0,
    InvalidHandle = -1,
    InvalidParameter = -2,
    NoDevice = -3,
    ComputeError = -4,
    InvalidDimensions = -5,
    OutOfMemory = -6,
    Unknown = -100,
}

// Metric types
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum VshipMetricType {
    Ssimulacra2 = 0,
    Butteraugli = 1,
    Cvvdp = 2,
}

// Image format enum
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum VshipImageFormat {
    RGB = 0,
    YUV420 = 1,
    YUV444 = 2,
}

impl From<VshipImageFormat> for ImageFormat {
    fn from(format: VshipImageFormat) -> Self {
        match format {
            VshipImageFormat::RGB => ImageFormat::RGB,
            VshipImageFormat::YUV420 => ImageFormat::YUV420,
            VshipImageFormat::YUV444 => ImageFormat::YUV444,
        }
    }
}

// Internal handle wrappers
struct VshipContextWrapper {
    context: MetricsContext,
}

enum MetricWrapper {
    Ssimulacra2(Ssimulacra2),
    Butteraugli(Butteraugli),
    Cvvdp(Cvvdp),
}

impl MetricWrapper {
    fn compute(&mut self, reference: &ImageData, distorted: &ImageData) -> Result<f64, VshipError> {
        match self {
            MetricWrapper::Ssimulacra2(m) => m.compute(reference, distorted),
            MetricWrapper::Butteraugli(m) => m.compute(reference, distorted),
            MetricWrapper::Cvvdp(m) => m.compute(reference, distorted),
        }
    }

    fn reset(&mut self) -> Result<(), VshipError> {
        match self {
            MetricWrapper::Ssimulacra2(m) => m.reset(),
            MetricWrapper::Butteraugli(m) => m.reset(),
            MetricWrapper::Cvvdp(m) => m.reset(),
        }
    }
}

// Helper to convert error to error code
fn error_to_code(err: VshipError) -> VshipErrorCode {
    match err {
        VshipError::NoDeviceFound => VshipErrorCode::NoDevice,
        VshipError::InvalidDimensions { .. } => VshipErrorCode::InvalidDimensions,
        VshipError::AllocationError(_) | VshipError::GpuAllocatorError(_) => VshipErrorCode::OutOfMemory,
        _ => VshipErrorCode::Unknown,
    }
}

/// Initialize Vship context
///
/// Returns an opaque handle to the Vship context, or NULL on failure.
/// The handle must be freed with vship_destroy() when done.
#[no_mangle]
pub extern "C" fn vship_init() -> *mut VshipHandle {
    match MetricsContext::new() {
        Ok(context) => {
            let wrapper = Box::new(VshipContextWrapper { context });
            Box::into_raw(wrapper) as *mut VshipHandle
        }
        Err(e) => {
            eprintln!("Failed to initialize Vship: {}", e);
            ptr::null_mut()
        }
    }
}

/// Destroy Vship context
///
/// Frees all resources associated with the context handle.
#[no_mangle]
pub extern "C" fn vship_destroy(handle: *mut VshipHandle) {
    if !handle.is_null() {
        unsafe {
            let _ = Box::from_raw(handle as *mut VshipContextWrapper);
        }
    }
}

/// Create a metric handler
///
/// Returns an opaque handle to the metric, or NULL on failure.
/// The handle must be freed with vship_metric_destroy() when done.
#[no_mangle]
pub extern "C" fn vship_metric_create(
    handle: *const VshipHandle,
    metric_type: VshipMetricType,
) -> *mut VshipMetricHandle {
    if handle.is_null() {
        return ptr::null_mut();
    }

    let wrapper = unsafe { &*(handle as *const VshipContextWrapper) };

    let metric = match metric_type {
        VshipMetricType::Ssimulacra2 => {
            wrapper.context.create_ssimulacra2().ok().map(MetricWrapper::Ssimulacra2)
        }
        VshipMetricType::Butteraugli => {
            wrapper.context.create_butteraugli().ok().map(MetricWrapper::Butteraugli)
        }
        VshipMetricType::Cvvdp => {
            wrapper.context.create_cvvdp().ok().map(MetricWrapper::Cvvdp)
        }
    };

    match metric {
        Some(m) => Box::into_raw(Box::new(m)) as *mut VshipMetricHandle,
        None => ptr::null_mut(),
    }
}

/// Destroy a metric handler
#[no_mangle]
pub extern "C" fn vship_metric_destroy(handle: *mut VshipMetricHandle) {
    if !handle.is_null() {
        unsafe {
            let _ = Box::from_raw(handle as *mut MetricWrapper);
        }
    }
}

/// Compute metric score
///
/// Computes the metric score between reference and distorted images.
/// Returns the score in the `score` output parameter.
///
/// Returns VshipErrorCode::Success on success, or an error code on failure.
#[no_mangle]
pub extern "C" fn vship_metric_compute(
    handle: *mut VshipMetricHandle,
    ref_data: *const f32,
    ref_width: c_uint,
    ref_height: c_uint,
    ref_format: VshipImageFormat,
    dist_data: *const f32,
    dist_width: c_uint,
    dist_height: c_uint,
    dist_format: VshipImageFormat,
    score: *mut c_double,
) -> VshipErrorCode {
    if handle.is_null() || ref_data.is_null() || dist_data.is_null() || score.is_null() {
        return VshipErrorCode::InvalidHandle;
    }

    if ref_width == 0 || ref_height == 0 || dist_width == 0 || dist_height == 0 {
        return VshipErrorCode::InvalidDimensions;
    }

    let metric = unsafe { &mut *(handle as *mut MetricWrapper) };

    // Calculate expected data size
    let ref_size = match ref_format.into() {
        ImageFormat::RGB | ImageFormat::YUV444 => (ref_width * ref_height * 3) as usize,
        ImageFormat::YUV420 => {
            let y_size = (ref_width * ref_height) as usize;
            let uv_size = ((ref_width / 2) * (ref_height / 2)) as usize;
            y_size + 2 * uv_size
        }
    };

    let dist_size = match dist_format.into() {
        ImageFormat::RGB | ImageFormat::YUV444 => (dist_width * dist_height * 3) as usize,
        ImageFormat::YUV420 => {
            let y_size = (dist_width * dist_height) as usize;
            let uv_size = ((dist_width / 2) * (dist_height / 2)) as usize;
            y_size + 2 * uv_size
        }
    };

    // Create ImageData from raw pointers
    let ref_slice = unsafe { std::slice::from_raw_parts(ref_data, ref_size) };
    let dist_slice = unsafe { std::slice::from_raw_parts(dist_data, dist_size) };

    let reference = match ImageData::from_f32(
        ref_width,
        ref_height,
        ref_slice,
        ref_format.into(),
    ) {
        Ok(img) => img,
        Err(e) => return error_to_code(e),
    };

    let distorted = match ImageData::from_f32(
        dist_width,
        dist_height,
        dist_slice,
        dist_format.into(),
    ) {
        Ok(img) => img,
        Err(e) => return error_to_code(e),
    };

    // Compute metric
    match metric.compute(&reference, &distorted) {
        Ok(result) => {
            unsafe { *score = result };
            VshipErrorCode::Success
        }
        Err(e) => error_to_code(e),
    }
}

/// Reset metric state
///
/// Resets the metric state (useful for video sequences).
#[no_mangle]
pub extern "C" fn vship_metric_reset(handle: *mut VshipMetricHandle) -> VshipErrorCode {
    if handle.is_null() {
        return VshipErrorCode::InvalidHandle;
    }

    let metric = unsafe { &mut *(handle as *mut MetricWrapper) };

    match metric.reset() {
        Ok(()) => VshipErrorCode::Success,
        Err(e) => error_to_code(e),
    }
}

/// Get error message for error code
#[no_mangle]
pub extern "C" fn vship_error_string(code: VshipErrorCode) -> *const c_char {
    let msg = match code {
        VshipErrorCode::Success => "Success",
        VshipErrorCode::InvalidHandle => "Invalid handle",
        VshipErrorCode::InvalidParameter => "Invalid parameter",
        VshipErrorCode::NoDevice => "No Vulkan device found",
        VshipErrorCode::ComputeError => "Compute error",
        VshipErrorCode::InvalidDimensions => "Invalid image dimensions",
        VshipErrorCode::OutOfMemory => "Out of memory",
        VshipErrorCode::Unknown => "Unknown error",
    };

    msg.as_ptr() as *const c_char
}

/// Get Vship version string
#[no_mangle]
pub extern "C" fn vship_version() -> *const c_char {
    "Vship 4.1.0 (Rust/Vulkan)\0".as_ptr() as *const c_char
}
