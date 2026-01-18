#ifndef VULKAN_INIT_HPP
#define VULKAN_INIT_HPP

#include "VulkanContext.hpp"
#include "VulkanKernel.hpp"

namespace vk_backend {

// Initialize Vulkan context - should be called before any GPU operations
inline void initializeVulkan() {
    static bool initialized = false;
    if (!initialized) {
        g_vulkanContext.initialize();
        initialized = true;
    }
}

// Cleanup Vulkan context - should be called when done with GPU
inline void cleanupVulkan() {
    VulkanKernelRegistry::getInstance().clear();
    g_vulkanContext.cleanup();
}

// RAII wrapper for Vulkan initialization
class VulkanInitializer {
public:
    VulkanInitializer() {
        initializeVulkan();
    }
    ~VulkanInitializer() {
        cleanupVulkan();
    }
};

} // namespace vk_backend

#endif // VULKAN_INIT_HPP
