# Vulkan Rewrite Plan

This document outlines the first steps toward a Vulkan-based rewrite of the GPU
backend. The current implementation uses HIP/CUDA kernels; a Vulkan rewrite
requires restructuring into a device-agnostic compute pipeline with explicit
resource management.

## Goals

- Provide a single GPU backend that runs on Vulkan 1.2+ devices.
- Replace HIP/CUDA kernels with Vulkan compute shaders.
- Preserve the public API for FFVship, Vapoursynth, and the C API.

## High-Level Architecture

### Device/Queue Management

- Create a Vulkan instance with validation layers (development builds only).
- Select a physical device based on compute queue support and memory budget.
- Use a dedicated compute queue; optionally allow a graphics-capable queue when
  compute-only queues are unavailable.

### Resource Model

- Replace HIP/CUDA device buffers with Vulkan buffers backed by device-local
  memory.
- Use staging buffers for host-visible uploads and downloads.
- Adopt a pooled allocator for transient buffers to minimize VkDeviceMemory
  churn.

### Shader Pipeline

- Port each kernel to SPIR-V compute shaders.
- Use specialization constants for tile sizes and thread group dimensions.
- Centralize pipeline creation/caching to avoid recompilation.

### Synchronization

- Use timeline semaphores for frame-to-frame synchronization.
- Favor command buffer reuse with per-frame fences.

## Work Breakdown

1. **Backend abstraction**
   - Define a GPU backend interface that hides HIP/CUDA/Vulkan specifics.
   - Add a backend selection mechanism (compile-time or runtime).

2. **Core utilities**
   - Port color space conversions and buffer management to Vulkan.
   - Implement a basic compute dispatch wrapper.

3. **Metric ports**
   - SSIMULACRA2 compute kernels.
   - Butteraugli compute kernels.
   - CVVDP compute kernels.

4. **Validation and parity**
   - Compare Vulkan outputs against the existing GPU backend.
   - Add tolerance-based regression tests for output parity.

## Risks

- Vulkan shader porting is non-trivial and requires careful validation.
- Some algorithms assume CUDA/HIP warp semantics that must be reimplemented.
- Performance tuning (workgroup sizes, memory layout) must be revisited.

## Next Steps

- Draft the backend interface and introduce a Vulkan-only build target.
- Start with a minimal Vulkan compute proof-of-concept (e.g. color conversion).
- Integrate Vulkan device selection into the existing GPU discovery path.
