# Vship Vulkan Compute Shaders

This directory contains GLSL compute shaders for GPU-accelerated image and video quality metrics.

## Structure

```
shaders/
├── glsl/         # GLSL source shaders (.comp)
├── spirv/        # Compiled SPIR-V binaries (.spv)
├── Makefile      # Build system for compiling shaders
└── compile_shaders.sh  # Compilation script
```

## Shaders

### rgb_to_xyb.comp
Converts RGB color space to XYB (opsin absorbance model) used by SSIMULACRA2.
- Input: 3 separate R, G, B buffers
- Output: 3 separate X, Y, B buffers
- Workgroup size: 16x16

### gaussian_blur.comp
Separable Gaussian blur for multi-scale processing.
- Supports both horizontal and vertical passes
- Configurable kernel radius via push constants
- Workgroup size: 256 (1D)

### downsample.comp
2x downsampling using 2x2 block averaging.
- Input: Full resolution image
- Output: Half resolution image (width/2, height/2)
- Workgroup size: 16x16

### ssim_error.comp
Computes SSIM-inspired error between reference and distorted images.
- Input: Reference and distorted buffers
- Output: Per-pixel error values
- Workgroup size: 256 (1D)

## Compiling Shaders

### Prerequisites

- Vulkan SDK with `glslc` compiler
- Download from: https://vulkan.lunarg.com/

### Build Commands

Using Makefile:
```bash
cd shaders
make            # Compile all shaders
make clean      # Remove compiled SPIR-V files
make info       # Show sources and targets
```

Using script:
```bash
cd shaders
./compile_shaders.sh
```

Manual compilation:
```bash
glslc glsl/rgb_to_xyb.comp -o spirv/rgb_to_xyb.spv
```

## Usage from Rust

The compiled SPIR-V shaders are loaded at runtime by the ShaderManager:

```rust
use vship_core::ShaderManager;

let mut shader_mgr = ShaderManager::new(device);

// Load shader
let shader = shader_mgr.load_shader("rgb_to_xyb")?;

// Use in pipeline
let pipeline = PipelineBuilder::new()
    .shader(shader)
    .add_storage_buffer(0)  // Input R
    .add_storage_buffer(1)  // Input G
    .add_storage_buffer(2)  // Input B
    .add_storage_buffer(3)  // Output X
    .add_storage_buffer(4)  // Output Y
    .add_storage_buffer(5)  // Output B
    .build(device)?;
```

## Shader Interface

### Push Constants

Many shaders use push constants for dynamic parameters:

```glsl
layout(push_constant) uniform PushConstants {
    uint width;
    uint height;
    // ... other parameters
} pc;
```

### Storage Buffers

Buffers are bound using descriptor sets:

```glsl
layout(set = 0, binding = 0) readonly buffer InputBuffer {
    float data[];
};
```

## Performance Considerations

- **Workgroup Size**: Optimized for modern GPUs (16x16 for 2D, 256 for 1D)
- **Memory Access**: Coalesced reads/writes where possible
- **Separable Filters**: Gaussian blur uses separable convolution for efficiency
- **Local Memory**: Can be added for shared data between threads (future optimization)

## Adding New Shaders

1. Create GLSL source in `glsl/` directory with `.comp` extension
2. Run `make` to compile to SPIR-V
3. Add shader name to `ShaderManager::preload_common_shaders()` if frequently used
4. Document the shader interface and usage

## Validation

Shaders are validated during compilation with:
- Syntax checking
- Resource binding validation
- Workgroup size validation
- SPIR-V generation and optimization

## References

- [Vulkan Compute Shader Tutorial](https://www.khronos.org/opengl/wiki/Compute_Shader)
- [GLSL Specification](https://www.khronos.org/registry/OpenGL/specs/gl/GLSLangSpec.4.60.pdf)
- [Vulkan GLSL Extensions](https://github.com/KhronosGroup/GLSL/blob/master/extensions/khr/GL_KHR_vulkan_glsl.txt)
