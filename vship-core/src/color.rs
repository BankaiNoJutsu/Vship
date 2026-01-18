/// Color space definitions and conversion utilities

use serde::{Deserialize, Serialize};

/// Color primaries (matching VshipColor.h)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ColorPrimaries {
    BT709 = 1,
    Unspecified = 2,
    BT470M = 4,
    BT470BG = 5,
    BT601 = 6,
    SMPTE240M = 7,
    GenericFilm = 8,
    BT2020 = 9,
    XYZ = 10,
    SMPTE431 = 11,
    SMPTE432 = 12,
    EBU3213 = 22,
}

/// Transfer characteristics (matching VshipColor.h)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TransferCharacteristics {
    BT709 = 1,
    Unspecified = 2,
    BT470M = 4,
    BT470BG = 5,
    BT601 = 6,
    SMPTE240M = 7,
    Linear = 8,
    Log100 = 9,
    Log316 = 10,
    IEC61966_2_4 = 11,
    BT1361E = 12,
    IEC61966_2_1 = 13, // sRGB
    BT2020_10 = 14,
    BT2020_12 = 15,
    SMPTE2084 = 16, // PQ
    SMPTE428 = 17,
    ARIB_STD_B67 = 18, // HLG
}

/// Matrix coefficients (matching VshipColor.h)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MatrixCoefficients {
    Identity = 0,
    BT709 = 1,
    Unspecified = 2,
    FCC = 4,
    BT470BG = 5,
    BT601 = 6,
    SMPTE240M = 7,
    YCgCo = 8,
    BT2020NCL = 9,
    BT2020CL = 10,
    SMPTE2085 = 11,
    ChromaDerivedNCL = 12,
    ChromaDerivedCL = 13,
    ICtCp = 14,
}

/// Chroma sample location
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChromaLocation {
    Left = 0,
    Center = 1,
    TopLeft = 2,
    Top = 3,
    BottomLeft = 4,
    Bottom = 5,
}

/// Pixel format descriptor
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct PixelFormat {
    pub bits_per_sample: u8,
    pub subsampling_w: u8, // 0 = 4:4:4, 1 = 4:2:0/4:2:2
    pub subsampling_h: u8, // 0 = 4:4:4/4:2:2, 1 = 4:2:0
    pub is_float: bool,
}

impl PixelFormat {
    /// Create YUV 4:2:0 8-bit format
    pub fn yuv420_8bit() -> Self {
        Self {
            bits_per_sample: 8,
            subsampling_w: 1,
            subsampling_h: 1,
            is_float: false,
        }
    }

    /// Create YUV 4:2:0 10-bit format
    pub fn yuv420_10bit() -> Self {
        Self {
            bits_per_sample: 10,
            subsampling_w: 1,
            subsampling_h: 1,
            is_float: false,
        }
    }

    /// Create YUV 4:4:4 format
    pub fn yuv444(bits: u8) -> Self {
        Self {
            bits_per_sample: bits,
            subsampling_w: 0,
            subsampling_h: 0,
            is_float: false,
        }
    }

    /// Create RGB format
    pub fn rgb(bits: u8, is_float: bool) -> Self {
        Self {
            bits_per_sample: bits,
            subsampling_w: 0,
            subsampling_h: 0,
            is_float,
        }
    }
}

/// Color space descriptor
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ColorSpace {
    pub primaries: ColorPrimaries,
    pub transfer: TransferCharacteristics,
    pub matrix: MatrixCoefficients,
    pub full_range: bool,
}

impl ColorSpace {
    /// BT.709 (HD)
    pub fn bt709() -> Self {
        Self {
            primaries: ColorPrimaries::BT709,
            transfer: TransferCharacteristics::BT709,
            matrix: MatrixCoefficients::BT709,
            full_range: false,
        }
    }

    /// BT.2020 (UHD)
    pub fn bt2020() -> Self {
        Self {
            primaries: ColorPrimaries::BT2020,
            transfer: TransferCharacteristics::BT2020_10,
            matrix: MatrixCoefficients::BT2020NCL,
            full_range: false,
        }
    }

    /// sRGB
    pub fn srgb() -> Self {
        Self {
            primaries: ColorPrimaries::BT709,
            transfer: TransferCharacteristics::IEC61966_2_1,
            matrix: MatrixCoefficients::Identity,
            full_range: true,
        }
    }

    /// HDR10 (BT.2020 with PQ transfer)
    pub fn hdr10() -> Self {
        Self {
            primaries: ColorPrimaries::BT2020,
            transfer: TransferCharacteristics::SMPTE2084,
            matrix: MatrixCoefficients::BT2020NCL,
            full_range: false,
        }
    }

    /// HLG (Hybrid Log-Gamma)
    pub fn hlg() -> Self {
        Self {
            primaries: ColorPrimaries::BT2020,
            transfer: TransferCharacteristics::ARIB_STD_B67,
            matrix: MatrixCoefficients::BT2020NCL,
            full_range: false,
        }
    }
}

/// Image descriptor combining format and color space
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageDescriptor {
    pub width: u32,
    pub height: u32,
    pub format: PixelFormat,
    pub color_space: ColorSpace,
    pub chroma_location: ChromaLocation,
}

impl ImageDescriptor {
    /// Create a new image descriptor
    pub fn new(
        width: u32,
        height: u32,
        format: PixelFormat,
        color_space: ColorSpace,
    ) -> Self {
        Self {
            width,
            height,
            format,
            color_space,
            chroma_location: ChromaLocation::Left,
        }
    }
}
