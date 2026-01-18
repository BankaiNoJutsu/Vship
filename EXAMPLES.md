# Vship Examples

This document provides examples of using Vship in various contexts.

## Table of Contents

- [Rust API Examples](#rust-api-examples)
- [C API Examples](#c-api-examples)
- [CLI Examples](#cli-examples)
- [Advanced Usage](#advanced-usage)

## Rust API Examples

### Simple Metric Computation

See [`examples/simple_metric.rs`](examples/simple_metric.rs)

```rust
use vship_metrics::{MetricsContext, Metric, ImageData, ImageFormat};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize Vship
    let ctx = MetricsContext::new()?;

    // Create metric
    let mut ssimulacra2 = ctx.create_ssimulacra2()?;

    // Create test images
    let reference = ImageData::new(1920, 1080, ImageFormat::RGB);
    let distorted = ImageData::new(1920, 1080, ImageFormat::RGB);

    // Compute score
    let score = ssimulacra2.compute(&reference, &distorted)?;
    println!("SSIMULACRA2 score: {:.4}", score);

    Ok(())
}
```

**Build & Run**:
```bash
cargo run --example simple_metric
```

### All Three Metrics

```rust
use vship_metrics::{MetricsContext, Metric, ImageData, ImageFormat};

fn compare_with_all_metrics(
    reference: &ImageData,
    distorted: &ImageData
) -> Result<(), Box<dyn std::error::Error>> {
    let ctx = MetricsContext::new()?;

    // SSIMULACRA2
    let mut ssim2 = ctx.create_ssimulacra2()?;
    let ssim2_score = ssim2.compute(reference, distorted)?;
    println!("SSIMULACRA2: {:.4}", ssim2_score);

    // Butteraugli
    let mut butter = ctx.create_butteraugli()?;
    let butter_score = butter.compute(reference, distorted)?;
    println!("Butteraugli: {:.4}", butter_score);

    // CVVDP
    let mut cvvdp = ctx.create_cvvdp()?;
    let cvvdp_score = cvvdp.compute(reference, distorted)?;
    println!("CVVDP: {:.4}", cvvdp_score);

    Ok(())
}
```

### Custom CVVDP Display Configuration

```rust
use vship_metrics::{Cvvdp, ImageData};
use vship_metrics::cvvdp::DisplayConfig;
use vship_core::VshipContext;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = VshipContext::new()?;
    let device = ctx.default_device();

    // Create custom display configuration
    let custom_display = DisplayConfig {
        name: "My 4K Monitor".to_string(),
        resolution: (3840, 2160),
        diagonal_size_inches: 32.0,
        viewing_distance_meters: 0.8,
        peak_luminance: 600.0,  // HDR display
        contrast_ratio: 100000.0,
    };

    let mut cvvdp = Cvvdp::with_display(device, custom_display)?;

    let reference = ImageData::new(3840, 2160, ImageFormat::RGB);
    let distorted = ImageData::new(3840, 2160, ImageFormat::RGB);

    let score = cvvdp.compute(&reference, &distorted)?;
    println!("CVVDP score (custom display): {:.4}", score);

    Ok(())
}
```

## C API Examples

### Basic C Usage

See [`examples/c_api_example.c`](examples/c_api_example.c)

```c
#include <stdio.h>
#include "vship.h"

int main() {
    // Initialize
    VshipHandle* ctx = vship_init();
    if (!ctx) {
        fprintf(stderr, "Failed to initialize\n");
        return 1;
    }

    // Create metric
    VshipMetricHandle* metric = vship_metric_create(
        ctx,
        VSHIP_METRIC_TYPE_SSIMULACRA2
    );

    // Prepare images (640x480 RGB)
    uint32_t width = 640, height = 480;
    float* reference = malloc(width * height * 3 * sizeof(float));
    float* distorted = malloc(width * height * 3 * sizeof(float));

    // ... fill images ...

    // Compute metric
    double score;
    VshipErrorCode result = vship_metric_compute(
        metric,
        reference, width, height, VSHIP_IMAGE_FORMAT_RGB,
        distorted, width, height, VSHIP_IMAGE_FORMAT_RGB,
        &score
    );

    if (result == VSHIP_ERROR_CODE_SUCCESS) {
        printf("Score: %.4f\n", score);
    }

    // Cleanup
    free(reference);
    free(distorted);
    vship_metric_destroy(metric);
    vship_destroy(ctx);

    return 0;
}
```

**Build**:
```bash
# Generate C header first
cargo build --release -p vship-ffi

# Compile C example
gcc -o c_example examples/c_api_example.c \
    -I./include \
    -L./target/release \
    -lvship_ffi

# Run
LD_LIBRARY_PATH=./target/release ./c_example
```

### C++ Wrapper

```cpp
#include <iostream>
#include <vector>
#include <memory>
#include "vship.h"

class VshipMetric {
    VshipHandle* ctx_;
    VshipMetricHandle* metric_;

public:
    VshipMetric(VshipMetricType type) {
        ctx_ = vship_init();
        if (!ctx_) throw std::runtime_error("Failed to init Vship");

        metric_ = vship_metric_create(ctx_, type);
        if (!metric_) throw std::runtime_error("Failed to create metric");
    }

    ~VshipMetric() {
        vship_metric_destroy(metric_);
        vship_destroy(ctx_);
    }

    double compute(
        const std::vector<float>& ref,
        const std::vector<float>& dist,
        uint32_t width,
        uint32_t height
    ) {
        double score;
        auto result = vship_metric_compute(
            metric_,
            ref.data(), width, height, VSHIP_IMAGE_FORMAT_RGB,
            dist.data(), width, height, VSHIP_IMAGE_FORMAT_RGB,
            &score
        );

        if (result != VSHIP_ERROR_CODE_SUCCESS) {
            throw std::runtime_error("Compute failed");
        }

        return score;
    }
};

int main() {
    try {
        VshipMetric ssim(VSHIP_METRIC_TYPE_SSIMULACRA2);

        std::vector<float> ref(640 * 480 * 3, 0.5f);
        std::vector<float> dist(640 * 480 * 3, 0.6f);

        double score = ssim.compute(ref, dist, 640, 480);
        std::cout << "Score: " << score << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
```

## CLI Examples

### Basic Video Comparison

```bash
# Compute SSIMULACRA2 for entire video
ffvship -r reference.mp4 -d encoded.mp4 -m ssimulacra2

# Save results to JSON
ffvship -r reference.mp4 -d encoded.mp4 -m ssimulacra2 -o results.json

# Verbose output with frame-by-frame progress
ffvship -r reference.mp4 -d encoded.mp4 -m ssimulacra2 -v
```

### Selective Frame Processing

```bash
# Process frames 100-200
ffvship -r ref.mp4 -d dist.mp4 --start-frame 100 --end-frame 200

# Process every 10th frame (faster, less accurate)
ffvship -r ref.mp4 -d dist.mp4 --frame-step 10

# Combine: frames 1000-2000, every 5th frame
ffvship -r ref.mp4 -d dist.mp4 \
    --start-frame 1000 \
    --end-frame 2000 \
    --frame-step 5 \
    -o sample_results.json
```

### Different Metrics

```bash
# SSIMULACRA2 (default)
ffvship -r ref.mp4 -d dist.mp4

# Butteraugli
ffvship -r ref.mp4 -d dist.mp4 -m butteraugli

# CVVDP
ffvship -r ref.mp4 -d dist.mp4 -m cvvdp
```

### GPU Selection

```bash
# Use first GPU (default)
ffvship -r ref.mp4 -d dist.mp4 --device 0

# Use second GPU
ffvship -r ref.mp4 -d dist.mp4 --device 1
```

### Output Analysis

The JSON output includes detailed statistics:

```json
{
  "metric": "SSIMULACRA2",
  "reference": "reference.mp4",
  "distorted": "encoded.mp4",
  "version": "4.1.0",
  "statistics": {
    "mean": 85.23,
    "min": 78.45,
    "max": 92.11,
    "std_dev": 3.42,
    "frame_count": 1234
  },
  "per_frame_scores": [
    {"frame": 0, "score": 85.2},
    {"frame": 1, "score": 85.5},
    ...
  ]
}
```

## Advanced Usage

### Batch Processing

```bash
#!/bin/bash
# Process multiple encodings

reference="original.mp4"
metrics=("ssimulacra2" "butteraugli" "cvvdp")

for encoded in encoded_*.mp4; do
    for metric in "${metrics[@]}"; do
        output="${encoded%.mp4}_${metric}.json"
        echo "Processing $encoded with $metric..."
        ffvship -r "$reference" -d "$encoded" -m "$metric" -o "$output"
    done
done
```

### Integration with Encoding Pipelines

```bash
# av1an integration example
av1an -i input.mkv \
      --encoder aom \
      --vmaf \
      --vmaf-path /path/to/ffvship \
      --vmaf-res 1920x1080
```

### Python Integration (via C API)

```python
import ctypes
import numpy as np

# Load library
vship = ctypes.CDLL('./target/release/libvship_ffi.so')

# Define types
class VshipHandle(ctypes.c_void_p):
    pass

class VshipMetricHandle(ctypes.c_void_p):
    pass

# Initialize
vship.vship_init.restype = VshipHandle
ctx = vship.vship_init()

# Create metric (0 = SSIMULACRA2)
vship.vship_metric_create.argtypes = [VshipHandle, ctypes.c_int]
vship.vship_metric_create.restype = VshipMetricHandle
metric = vship.vship_metric_create(ctx, 0)

# Create test data
width, height = 640, 480
ref_data = np.random.rand(height, width, 3).astype(np.float32)
dist_data = np.random.rand(height, width, 3).astype(np.float32)

# Compute
vship.vship_metric_compute.argtypes = [
    VshipMetricHandle,
    ctypes.POINTER(ctypes.c_float), ctypes.c_uint, ctypes.c_uint, ctypes.c_int,
    ctypes.POINTER(ctypes.c_float), ctypes.c_uint, ctypes.c_uint, ctypes.c_int,
    ctypes.POINTER(ctypes.c_double)
]

score = ctypes.c_double()
result = vship.vship_metric_compute(
    metric,
    ref_data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    width, height, 0,  # 0 = RGB
    dist_data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    width, height, 0,
    ctypes.byref(score)
)

print(f"Score: {score.value:.4f}")

# Cleanup
vship.vship_metric_destroy(metric)
vship.vship_destroy(ctx)
```

## Performance Tips

1. **GPU Selection**: Use `--device` to select the fastest GPU
2. **Frame Stepping**: Use `--frame-step` for faster preview analysis
3. **Batch Processing**: Process multiple videos in parallel (different GPUs)
4. **Memory**: Ensure sufficient VRAM for high-resolution videos

## Troubleshooting Examples

### Error: No Vulkan device found

```bash
# Check Vulkan availability
vulkaninfo

# Verify GPU drivers
lspci | grep -i vga

# Update drivers and retry
sudo apt update && sudo apt upgrade
```

### Error: Video dimensions mismatch

```bash
# Check video resolutions
ffprobe -v error -select_streams v:0 \
    -show_entries stream=width,height \
    -of csv=s=x:p=0 reference.mp4

# Resize if needed before processing
```

## Next Steps

- Review [BUILDING.md](BUILDING.md) for build instructions
- Check [README.md](README.md) for feature overview
- Explore [shaders/README.md](shaders/README.md) for GPU implementation details
