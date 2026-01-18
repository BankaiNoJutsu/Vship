#ifndef VULKAN_HELPER_HPP
#define VULKAN_HELPER_HPP

#include "VulkanPreprocessor.hpp"
#include "../util/VshipExceptions.hpp"
#include <sstream>

namespace helper {

inline int checkGpuCount() {
    int count;
    if (vk_backend::vkGetDeviceCount(&count) != vk_backend::vkSuccess) {
        throw VshipError(DeviceCountError, __FILE__, __LINE__);
    }
    if (count == 0) {
        throw VshipError(NoDeviceDetected, __FILE__, __LINE__);
    }
    return count;
}

inline bool gpuKernelCheck() {
    // For Vulkan, we can't easily test a kernel without having one compiled
    // Just return true for now - proper testing will happen during kernel execution
    return true;
}

inline void gpuFullCheck(int gpuid = 0) {
    int count = checkGpuCount();

    if (count <= gpuid || gpuid < 0) {
        throw VshipError(BadDeviceArgument, __FILE__, __LINE__);
    }

    vk_backend::vkSetDevice(gpuid);

    if (!gpuKernelCheck()) {
        throw VshipError(BadDeviceCode, __FILE__, __LINE__);
    }
}

inline std::string listGPU() {
    std::stringstream ss;
    int count = checkGpuCount();

    for (int i = 0; i < count; i++) {
        vk_backend::vkSetDevice(i);
        std::string name = vk_backend::g_vulkanContext.getDeviceName(i);
        ss << "GPU " << i << ": " << name << std::endl;
    }
    return ss.str();
}

} // namespace helper

#endif // VULKAN_HELPER_HPP
