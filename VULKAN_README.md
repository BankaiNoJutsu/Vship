# Vship Vulkan Backend

This document describes the Vulkan backend rewrite for Vship, providing GPU-accelerated visual fidelity metrics computation using Vulkan Compute instead of HIP/CUDA.

## Overview

The Vulkan backend provides a portable, cross-platform alternative to the HIP and CUDA backends. It uses Vulkan Compute shaders to perform the same GPU computations as the original implementation.

### Benefits

- **Cross-platform**: Works on NVIDIA, AMD, Intel, and other Vulkan-compatible GPUs
- **No proprietary SDKs**: Only requires Vulkan drivers, no CUDA or ROCm needed
- **Future-proof**: Vulkan is a modern, actively developed API
- **Portable**: Works on Windows, Linux, macOS (via MoltenVK), and even mobile devices

## Architecture

### Core Components

The Vulkan backend is organized into several key components:

#### 1. **VulkanContext** (`src/vulkan/VulkanContext.hpp/cpp`)
Manages Vulkan instance, devices, and resources:
- Device enumeration and selection
- Memory allocation and management
- Buffer creation and operations
- Pipeline management
- Synchronization primitives (fences, semaphores)

#### 2. **VulkanStream** (`src/vulkan/VulkanStream.hpp`)
Provides asynchronous command execution similar to CUDA streams:
- Command buffer management
- Asynchronous operations
- Queue submission
- Synchronization

#### 3. **VulkanKernel** (`src/vulkan/VulkanKernel.hpp`)
Kernel management and dispatch system:
- SPIR-V shader loading
- Compute pipeline creation
- Descriptor set management
- Kernel registration and caching
- Launch configuration (grid/block dimensions)

#### 4. **VulkanPreprocessor** (`src/vulkan/VulkanPreprocessor.hpp`)
API compatibility layer that maps HIP/CUDA calls to Vulkan:
- Memory operations (malloc, free, memcpy)
- Device management (set device, get device count)
- Stream operations
- Event/fence management

#### 5. **VulkanHelper** (`src/vulkan/VulkanHelper.hpp`)
Helper functions for device detection and validation

#### 6. **VulkanInit** (`src/vulkan/VulkanInit.hpp`)
Initialization and cleanup utilities

### Integration with Existing Code

The Vulkan backend integrates seamlessly with the existing codebase through:

1. **Preprocessor Abstraction** (`src/util/preprocessor.hpp`):
   - Define `USE_VULKAN` to enable Vulkan backend
   - All `hipXXX` macros are mapped to Vulkan equivalents
   - No changes needed to existing CPU-side code

2. **GPU Helper** (`src/util/gpuhelper.hpp`):
   - Conditional compilation includes Vulkan helpers when `USE_VULKAN` is defined

## Building with Vulkan

### Prerequisites

- Vulkan SDK (includes headers and loader library)
- `glslangValidator` for compiling GLSL shaders to SPIR-V
- C++17 compiler (g++, clang++, or MSVC)

### Build Steps

1. **Compile shaders**:
```bash
cd src/vulkan/shaders
bash compile_shaders.sh
```

This compiles all `.comp` (GLSL compute shader) files to `.spv` (SPIR-V) files.

2. **Build Vship with Vulkan backend**:
```bash
make buildvulkan
```

This will:
- Compile shaders automatically
- Build `libvship.so` with Vulkan backend
- Link against Vulkan library

### Manual Build

```bash
# Compile shaders
cd src/vulkan/shaders
glslangValidator -V downsample.comp -o spirv/downsample.spv
cd ../../..

# Build library
g++ src/VshipLib.cpp src/vulkan/VulkanContext.cpp \
    -g -std=c++17 -Wall \
    -DUSE_VULKAN \
    -I include -I src \
    -lvulkan \
    -shared -fPIC \
    -o libvship.so
```

## Converting Kernels to Vulkan

### Step-by-Step Migration Guide

#### 1. Convert CUDA/HIP Kernel to GLSL Compute Shader

**Original CUDA/HIP kernel** (`downsample.hpp`):
```cpp
__launch_bounds__(256)
__global__ void downsamplekernel(float* src, float* dst, int64_t width, int64_t height) {
    int64_t x = threadIdx.x + blockIdx.x*blockDim.x;
    int64_t y = threadIdx.y + blockIdx.y*blockDim.y;

    int64_t newh = (height-1)/2 + 1;
    int64_t neww = (width-1)/2 + 1;

    if (x >= neww || y >= newh) return;

    dst[y * neww + x] = /* computation */;
}
```

**GLSL Compute Shader** (`src/vulkan/shaders/downsample.comp`):
```glsl
#version 450

layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

layout(binding = 0) readonly buffer SrcBuffer {
    float data[];
} src;

layout(binding = 1) writeonly buffer DstBuffer {
    float data[];
} dst;

layout(push_constant) uniform PushConstants {
    int64_t width;
    int64_t height;
} pc;

void main() {
    int64_t x = int64_t(gl_GlobalInvocationID.x);
    int64_t y = int64_t(gl_GlobalInvocationID.y);

    int64_t newh = (pc.height - 1) / 2 + 1;
    int64_t neww = (pc.width - 1) / 2 + 1;

    if (x >= neww || y >= newh) return;

    dst.data[y * neww + x] = /* computation */;
}
```

**Key differences:**
- `threadIdx + blockIdx*blockDim` → `gl_GlobalInvocationID`
- `__global__` → compute shader with `layout(local_size_*)`
- Pointers → buffer blocks with `layout(binding = N)`
- Parameters → push constants or uniform buffers

#### 2. Create Vulkan Host Wrapper

**Create `*_vulkan.hpp`** file (e.g., `downsample_vulkan.hpp`):

```cpp
#ifdef USE_VULKAN

#include "../vulkan/VulkanKernel.hpp"

namespace ssimu2 {

inline void downsample(float* src_ptr, float* dst_ptr, int64_t width, int64_t height, hipStream_t stream) {
    using namespace vk_backend;

    // Get device buffers (hipMalloc returns VulkanBuffer*)
    VulkanBuffer* src = reinterpret_cast<VulkanBuffer*>(src_ptr);
    VulkanBuffer* dst = reinterpret_cast<VulkanBuffer*>(dst_ptr);

    // Define descriptor bindings
    std::vector<VkDescriptorSetLayoutBinding> bindings(2);
    bindings[0].binding = 0; // src
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    bindings[1].binding = 1; // dst
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    // Register kernel (once)
    static bool registered = false;
    if (!registered) {
        auto spirv = loadSPIRV("src/vulkan/shaders/spirv/downsample.spv");
        VulkanKernelRegistry::getInstance().registerKernel("downsample", spirv, bindings);
        registered = true;
    }

    // Calculate grid/block dimensions
    int64_t newh = (height-1)/2 + 1;
    int64_t neww = (width-1)/2 + 1;
    int64_t th_x = std::min((int64_t)16, neww);
    int64_t th_y = std::min((int64_t)16, newh);
    int64_t bl_x = (neww-1)/th_x + 1;
    int64_t bl_y = (newh-1)/th_y + 1;

    // Launch kernel
    VulkanKernelLauncher("downsample", stream)
        .setGrid(bl_x, bl_y, 1)
        .setBlock(th_x, th_y, 1)
        .addBuffer(src)
        .addBuffer(dst)
        .setPushConstant(0, width)
        .setPushConstant(1, height)
        .dispatch();
}

} // namespace ssimu2

#endif // USE_VULKAN
```

#### 3. Conditionally Include Vulkan Version

In the original kernel file, add:
```cpp
#ifdef USE_VULKAN
    #include "downsample_vulkan.hpp"
#else
    // Original CUDA/HIP implementation
    __global__ void downsamplekernel(...) { ... }
    void inline downsample(...) { ... }
#endif
```

### CUDA/HIP to GLSL Translation Reference

| CUDA/HIP | GLSL/Vulkan |
|----------|-------------|
| `threadIdx.x` | `gl_LocalInvocationID.x` |
| `blockIdx.x` | `gl_WorkGroupID.x` |
| `blockDim.x` | `gl_WorkGroupSize.x` |
| `gridDim.x` | `gl_NumWorkGroups.x` |
| `threadIdx + blockIdx*blockDim` | `gl_GlobalInvocationID` |
| `__global__` | `layout(local_size_x=...) in; void main()` |
| `__device__` | Regular function |
| `__shared__` | `shared` qualifier |
| `__syncthreads()` | `barrier()` |
| `float* buffer` | `layout(binding=N) buffer { float data[]; }` |
| Kernel parameters | `push_constant` or uniform buffer |

### Data Type Mapping

| CUDA/HIP | GLSL |
|----------|------|
| `float` | `float` |
| `double` | `double` |
| `float3` | `vec3` |
| `float4` | `vec4` |
| `int` | `int` |
| `unsigned int` | `uint` |
| `int64_t` | `int64_t` (requires extension) |

## Current Status

### ✅ Completed

- [x] Vulkan infrastructure (context, device, memory management)
- [x] Stream/command buffer abstraction
- [x] Kernel management and dispatch system
- [x] HIP/CUDA API compatibility layer
- [x] Build system integration
- [x] Example kernel conversion (downsample)

### 🚧 In Progress

- [ ] SSIMULACRA2 kernels:
  - [x] Downsample
  - [ ] Gaussian blur
  - [ ] makeXYB color transform
  - [ ] SSIM calculation
  - [ ] Score computation

- [ ] Butteraugli kernels:
  - [ ] Opsin dynamics
  - [ ] Malta diff
  - [ ] Frequency separation
  - [ ] Psychovisual masking
  - [ ] Diff norms

- [ ] CVVDP kernels:
  - [ ] Laplacian pyramid
  - [ ] Temporal filter
  - [ ] CSF application
  - [ ] Masking model
  - [ ] Pooling

- [ ] Color conversion kernels:
  - [ ] YUV to RGB
  - [ ] Bit depth conversion
  - [ ] Chroma upsampling
  - [ ] Transfer functions
  - [ ] Primaries conversion

### 📋 TODO

- [ ] Complete all kernel conversions
- [ ] Optimize descriptor set allocation (reuse pools)
- [ ] Implement proper timestamp queries for performance metrics
- [ ] Add peer device memory copy support
- [ ] Comprehensive testing against HIP/CUDA reference
- [ ] Performance benchmarking
- [ ] Add validation layer support for debugging

## Performance Considerations

### Optimization Tips

1. **Descriptor Set Reuse**: Currently descriptor pools are created per dispatch. Cache and reuse them.

2. **Pipeline Caching**: Pipelines are registered once and cached. Consider saving pipeline cache to disk.

3. **Memory Transfers**: Use staging buffers efficiently. The current implementation creates temporary staging buffers for each transfer.

4. **Command Buffer Recording**: Stream-based recording allows efficient batching.

5. **Specialization Constants**: Use for compile-time kernel parameters instead of push constants when possible.

## Debugging

### Validation Layers

To enable Vulkan validation layers for debugging (add to VulkanContext.cpp):

```cpp
const std::vector<const char*> validationLayers = {
    "VK_LAYER_KHRONOS_validation"
};

// In createInstance():
createInfo.enabledLayerCount = validationLayers.size();
createInfo.ppEnabledLayerNames = validationLayers.data();
```

### Shader Debugging

Compile shaders with debug info:
```bash
glslangValidator -V -g downsample.comp -o downsample.spv
```

### Runtime Debugging

Set environment variables:
```bash
export VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation
export VK_LOADER_DEBUG=all
./FFVship <options>
```

## API Compatibility

The Vulkan backend maintains API compatibility with the HIP/CUDA backends:

- Same C API (`VshipAPI.h`)
- Same VapourSynth plugin interface
- Same FFVship CLI tool
- Same metric algorithms and accuracy

Users can switch backends by simply using a different binary, without changing their code or workflows.

## Contributing

### Adding New Kernels

1. Write GLSL compute shader in `src/vulkan/shaders/*.comp`
2. Create `*_vulkan.hpp` wrapper
3. Update original kernel file with conditional compilation
4. Add shader to `compile_shaders.sh`
5. Test against reference implementation

### Testing

Ensure mathematical equivalence:
```bash
# Run with HIP/CUDA
./FFVship ref.mkv dist.mkv --metric ssimulacra2 > cuda_result.txt

# Run with Vulkan
./FFVship ref.mkv dist.mkv --metric ssimulacra2 > vulkan_result.txt

# Compare results (should be identical or within floating-point error)
diff cuda_result.txt vulkan_result.txt
```

## License

Same as Vship main project.

## References

- [Vulkan Specification](https://www.khronos.org/registry/vulkan/specs/1.3/html/)
- [Vulkan Compute Tutorial](https://github.com/Erkaman/vulkan_minimal_compute)
- [GLSL Specification](https://www.khronos.org/registry/OpenGL/specs/gl/GLSLangSpec.4.60.pdf)
- [SPIR-V Specification](https://www.khronos.org/registry/spir-v/)
