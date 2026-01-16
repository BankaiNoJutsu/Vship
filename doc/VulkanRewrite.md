# Vulkan + Rust Rewrite Plan

This document outlines the first steps toward a Vulkan-based rewrite of the GPU
backend and a Rust-based core for the library. The current implementation uses
HIP/CUDA kernels and C++ entry points; a Vulkan and Rust rewrite requires
restructuring into a device-agnostic compute pipeline with explicit resource
management plus a safe, modern host-side API.

## Goals

- Provide a single GPU backend that runs on Vulkan 1.2+ devices.
- Replace HIP/CUDA kernels with Vulkan compute shaders.
- Migrate host-side logic to Rust for safety and maintainability.
- Preserve the public API for FFVship, Vapoursynth, and the C API via FFI.

## High-Level Architecture

### Rust Host Layer

- Create a Rust crate that owns device selection, queue management, and resource
  lifetimes.
- Expose a C ABI surface for FFVship and Vapoursynth bindings.
- Keep the C API stable by mirroring current structs and error codes.

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

### Host/Shader Interface

- Define Rust-side descriptor set layouts that map to shader expectations.
- Use explicit buffer layouts with versioned structs to avoid ABI drift.
- Generate SPIR-V with shader compilation tooling checked into the build system.

### Synchronization

- Use timeline semaphores for frame-to-frame synchronization.
- Favor command buffer reuse with per-frame fences.

## Work Breakdown

1. **Backend abstraction**
   - Define a Rust GPU backend interface that hides HIP/CUDA/Vulkan specifics.
   - Add a backend selection mechanism (compile-time or runtime).

2. **Rust core**
   - Port color space conversions and buffer management to Rust.
   - Implement a basic Vulkan compute dispatch wrapper.
   - Introduce a C ABI shim for FFVship and Vapoursynth integration.

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
- FFI boundaries must be carefully designed to avoid regressions.

## Next Steps

- Draft the Rust backend interface and introduce a Vulkan-only build target.
- Start with a minimal Vulkan compute proof-of-concept (e.g. color conversion).
- Integrate Vulkan device selection into the existing GPU discovery path.
