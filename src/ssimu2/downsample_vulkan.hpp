#ifndef DOWNSAMPLE_VULKAN_HPP
#define DOWNSAMPLE_VULKAN_HPP

#ifdef USE_VULKAN

#include "../vulkan/VulkanKernel.hpp"
#include "../vulkan/VulkanStream.hpp"
#include <algorithm>

namespace ssimu2 {

// Vulkan version of downsample function
inline void downsample(float* src_ptr, float* dst_ptr, int64_t width, int64_t height, hipStream_t stream) {
    using namespace vk_backend;

    int64_t newh = (height-1)/2 + 1;
    int64_t neww = (width-1)/2 + 1;

    // Get device buffers
    VulkanBuffer* src = reinterpret_cast<VulkanBuffer*>(src_ptr);
    VulkanBuffer* dst = reinterpret_cast<VulkanBuffer*>(dst_ptr);

    // Set up descriptor bindings for the shader
    std::vector<VkDescriptorSetLayoutBinding> bindings(2);

    // Binding 0: src buffer (readonly)
    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    // Binding 1: dst buffer (writeonly)
    bindings[1].binding = 1;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    // Load SPIR-V shader (this should be done once and cached)
    static bool kernelRegistered = false;
    if (!kernelRegistered) {
        try {
            std::vector<uint32_t> spirv = loadSPIRV("src/vulkan/shaders/spirv/downsample.spv");
            VulkanKernelRegistry::getInstance().registerKernel("downsample", spirv, bindings,
                                                               stream ? stream->getDeviceIdx() : -1);
            kernelRegistered = true;
        } catch (const VshipError& e) {
            throw;
        }
    }

    // Calculate grid and block dimensions
    int64_t th_x = std::min((int64_t)16, neww);
    int64_t th_y = std::min((int64_t)16, newh);
    int64_t bl_x = (neww-1)/th_x + 1;
    int64_t bl_y = (newh-1)/th_y + 1;

    // Launch kernel
    VulkanKernelLauncher launcher("downsample", stream);
    launcher.setGrid(bl_x, bl_y, 1)
           .setBlock(th_x, th_y, 1)
           .addBuffer(src)
           .addBuffer(dst)
           .setPushConstant(0, width)
           .setPushConstant(1, height)
           .dispatch();
}

} // namespace ssimu2

#endif // USE_VULKAN

#endif // DOWNSAMPLE_VULKAN_HPP
