# Vulkan SSIM2 Next Changes Plan

## Goals
- Reduce global memory traffic by fusing blur + error and doing tile-local reductions. (done)
- Cut per-frame submit overhead by batching multiple frames per submit (single queue). (done)
- Reduce CPU/transfer overhead that still dominates wall time.
- Preserve metric accuracy against the current RGBA8 and F32 paths.

## Planned Changes
1) Add a fused shader that combines vertical blur + error computation + tile-local reduction. (done)
   - New shader: `shaders/glsl/gaussian_blur_error_reduce.comp`.
   - Outputs one sum per workgroup (tile) to a compact buffer.
   - Keeps the horizontal blur pass for separability, but removes the vertical blur output and per-pixel error buffer.

2) Introduce a tiled reduction path in `Ssimulacra2Gpu`. (done)
   - Use the fused shader for `ReduceMode::Gpu`.
   - Keep the existing per-pixel error path for `ReduceMode::Cpu` (accuracy/debug).
   - Reuse existing reduce_sum pipeline to collapse tile sums to a single value.

3) Batch 2-4 frames per submit on the single compute queue. (done)
   - Increase descriptor pool capacity to allow multiple frames without resetting.
   - Add a batched RGBA8 compute path for SSIMULACRA2 that records N frames into one command buffer.
   - Expose `--batch-frames` in ffvship; enable only for SSIMULACRA2 + RGBA8 + GPU reduce + single mode.

4) Reuse staging buffers to cut per-frame upload overhead. (done)
   - Add a staging buffer pool in `ComputeContext`.
   - Acquire/reclaim staging buffers per batch instead of allocating per upload.

5) Reduce per-frame descriptor churn. (done)
   - Cache descriptor sets per pipeline/binding set and reuse them across frames.
   - Avoid descriptor pool resets on the hot path; reset only on buffer reallocation.

6) Overlap transfers with compute when a transfer queue is available.
   - Record copies on transfer queue and signal compute queue with semaphores.

7) Tighten barriers and batch dispatches.
   - Reduce per-dispatch global barriers where dependencies allow.

8) Consolidate readbacks. (done)
   - Copy reduced sums into a packed results buffer and read back once per batch.

9) Validate accuracy and report performance.
   - Run RGBA8 vs F32 comparison on the same sample.
   - Report FPS, total time, and mean/min/max/stddev parity.

## Files Expected to Change
- `vship-core/src/pipeline.rs` (descriptor pool size)
- `vship-core/src/compute.rs`
- `vship-core/src/shader_manager.rs`
- `shaders/glsl/gaussian_blur_error_reduce.comp`
- `shaders/compile.bat`
- `vship-metrics/src/ssimulacra2_gpu.rs`
- `ffvship/src/main.rs`

## Acceptance Criteria
- No score regressions (RGBA8 vs F32 parity maintained).
- Faster throughput vs current RGBA8 path.
- GPU usage stays stable on dGPU targets (no stalls or crashes).
