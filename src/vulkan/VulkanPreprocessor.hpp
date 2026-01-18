#ifndef VULKAN_PREPROCESSOR_HPP
#define VULKAN_PREPROCESSOR_HPP

#include "VulkanContext.hpp"
#include "VulkanStream.hpp"
#include "VulkanInit.hpp"
#include <string>
#include <sstream>
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <vector>
#include <chrono>
#include <thread>
#include <exception>
#include <set>
#include <mutex>
#include <condition_variable>
#include <cassert>
#include <cstring>
#include <algorithm>

using uint = unsigned int;

#ifdef _WIN32
    #define aligned_alloc(a, b) _aligned_malloc(b, a)
#endif

#define GAUSSIANSIZE 8
#define SIGMA 1.5f
#define PI  3.14159265359
#define TAU 6.28318530718

enum InputMemType {UINT16, HALF, FLOAT};

// Vulkan backend type definitions
namespace vk_backend {
    using vkStream_t = VulkanStream*;
    using vkError_t = VkResult;
    using vkDeviceptr_t = VulkanBuffer*;
    using vkEvent_t = VkFence;
    using vkDeviceProp_t = VkPhysicalDeviceProperties;

    // Error codes
    constexpr vkError_t vkSuccess = VK_SUCCESS;

    // Cache config (no-op for Vulkan)
    enum vkFuncCache {
        vkFuncCachePreferShared,
        vkFuncCachePreferNone,
        vkFuncCachePreferL1,
        vkFuncCachePreferEqual
    };
}

// Vulkan API wrappers to match HIP/CUDA interface
#define hipMemcpyDtoH(x, y, z) vk_backend::vkMemcpyDtoH(x, y, z)
#define hipMemcpyHtoD(x, y, z) vk_backend::vkMemcpyHtoD(x, y, z)
#define hipMemcpyDtoHAsync(x, y, z, w) vk_backend::vkMemcpyDtoHAsync(x, y, z, w)
#define hipMemcpyHtoDAsync(x, y, z, w) vk_backend::vkMemcpyHtoDAsync(x, y, z, w)
#define hipMemcpyDtoDAsync(x, y, z, w) vk_backend::vkMemcpyDtoDAsync(x, y, z, w)
#define hipMemcpyPeer vk_backend::vkMemcpyPeer
#define hipMemcpyPeerAsync vk_backend::vkMemcpyPeerAsync
#define hipMalloc vk_backend::vkMalloc
#define hipFree vk_backend::vkFree
#define hipMallocAsync vk_backend::vkMallocAsync
#define hipFreeAsync vk_backend::vkFreeAsync
#define hipDeviceSynchronize vk_backend::vkDeviceSynchronize
#define hipSetDevice vk_backend::vkSetDevice
#define hipDeviceProp_t vk_backend::vkDeviceProp_t
#define hipGetDeviceCount vk_backend::vkGetDeviceCount
#define hipDeviceptr_t vk_backend::vkDeviceptr_t
#define hipGetDevice vk_backend::vkGetDevice
#define hipGetDeviceProperties vk_backend::vkGetDeviceProperties
#define hipError_t vk_backend::vkError_t
#define hipGetErrorString vk_backend::vkGetErrorString
#define hipStream_t vk_backend::vkStream_t
#define hipStreamAddCallback vk_backend::vkStreamAddCallback
#define hipDeviceEnablePeerAccess vk_backend::vkDeviceEnablePeerAccess
#define hipSuccess vk_backend::vkSuccess
#define hipGetLastError vk_backend::vkGetLastError
#define hipStreamCreate vk_backend::vkStreamCreate
#define hipStreamDestroy vk_backend::vkStreamDestroy
#define hipEventCreate vk_backend::vkEventCreate
#define hipEventDestroy vk_backend::vkEventDestroy
#define hipEventSynchronize vk_backend::vkEventSynchronize
#define hipEventRecord vk_backend::vkEventRecord
#define hipEvent_t vk_backend::vkEvent_t
#define hipEventElapsedTime vk_backend::vkEventElapsedTime
#define hipDeviceSetCacheConfig vk_backend::vkDeviceSetCacheConfig
#define hipFuncCachePreferShared vk_backend::vkFuncCachePreferShared
#define hipFuncCachePreferNone vk_backend::vkFuncCachePreferNone
#define hipFuncCachePreferL1 vk_backend::vkFuncCachePreferL1
#define hipFuncCachePreferEqual vk_backend::vkFuncCachePreferEqual
#define hipMemGetInfo vk_backend::vkMemGetInfo
#define hipMemsetAsync vk_backend::vkMemsetAsync
#define hipMemset vk_backend::vkMemset
#define hipHostFree vk_backend::vkHostFree
#define hipFreeHost vk_backend::vkHostFree
#define hipHostMalloc vk_backend::vkHostMalloc
#define hipStreamSynchronize vk_backend::vkStreamSynchronize
#define hipStreamWaitEvent vk_backend::vkStreamWaitEvent

namespace vk_backend {

// Memory copy operations
inline vkError_t vkMemcpyDtoH(void* dst, vkDeviceptr_t src, size_t size) {
    try {
        g_vulkanContext.copyBufferToHost(dst, *src, size);
        return VK_SUCCESS;
    } catch (...) {
        return VK_ERROR_UNKNOWN;
    }
}

inline vkError_t vkMemcpyHtoD(vkDeviceptr_t dst, const void* src, size_t size) {
    try {
        g_vulkanContext.copyHostToBuffer(*dst, src, size);
        return VK_SUCCESS;
    } catch (...) {
        return VK_ERROR_UNKNOWN;
    }
}

inline vkError_t vkMemcpyDtoHAsync(void* dst, vkDeviceptr_t src, size_t size, vkStream_t stream);
inline vkError_t vkMemcpyHtoDAsync(vkDeviceptr_t dst, const void* src, size_t size, vkStream_t stream);
inline vkError_t vkMemcpyDtoDAsync(vkDeviceptr_t dst, vkDeviceptr_t src, size_t size, vkStream_t stream);

// Peer memory operations (stub - requires extension)
inline vkError_t vkMemcpyPeer(vkDeviceptr_t dst, int dstDevice, vkDeviceptr_t src, int srcDevice, size_t size) {
    // TODO: Implement peer device copy
    return VK_ERROR_FEATURE_NOT_PRESENT;
}

inline vkError_t vkMemcpyPeerAsync(vkDeviceptr_t dst, int dstDevice, vkDeviceptr_t src, int srcDevice,
                                    size_t size, vkStream_t stream) {
    // TODO: Implement async peer device copy
    return VK_ERROR_FEATURE_NOT_PRESENT;
}

// Memory allocation
inline vkError_t vkMalloc(vkDeviceptr_t* ptr, size_t size) {
    try {
        auto buffer = g_vulkanContext.createBuffer(
            size,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
        );
        *ptr = new VulkanBuffer(buffer);
        return VK_SUCCESS;
    } catch (const VshipError& e) {
        if (e.type == OutOfVRAM) return VK_ERROR_OUT_OF_DEVICE_MEMORY;
        return VK_ERROR_UNKNOWN;
    } catch (...) {
        return VK_ERROR_UNKNOWN;
    }
}

inline vkError_t vkFree(vkDeviceptr_t ptr) {
    try {
        if (ptr) {
            g_vulkanContext.destroyBuffer(*ptr);
            delete ptr;
        }
        return VK_SUCCESS;
    } catch (...) {
        return VK_ERROR_UNKNOWN;
    }
}

inline vkError_t vkMallocAsync(vkDeviceptr_t* ptr, size_t size, vkStream_t stream);
inline vkError_t vkFreeAsync(vkDeviceptr_t ptr, vkStream_t stream);

// Device management
inline vkError_t vkDeviceSynchronize() {
    try {
        VulkanDevice* dev = g_vulkanContext.getDevice();
        vkQueueWaitIdle(dev->computeQueue);
        return VK_SUCCESS;
    } catch (...) {
        return VK_ERROR_UNKNOWN;
    }
}

inline vkError_t vkSetDevice(int device) {
    try {
        g_vulkanContext.setDevice(device);
        return VK_SUCCESS;
    } catch (const VshipError& e) {
        if (e.type == BadDeviceArgument) return VK_ERROR_DEVICE_LOST;
        return VK_ERROR_UNKNOWN;
    } catch (...) {
        return VK_ERROR_UNKNOWN;
    }
}

inline vkError_t vkGetDeviceCount(int* count) {
    try {
        initializeVulkan();  // Auto-initialize if needed
        *count = g_vulkanContext.getDeviceCount();
        return VK_SUCCESS;
    } catch (...) {
        return VK_ERROR_UNKNOWN;
    }
}

inline vkError_t vkGetDevice(int* device) {
    try {
        *device = g_vulkanContext.getCurrentDevice();
        return VK_SUCCESS;
    } catch (...) {
        return VK_ERROR_UNKNOWN;
    }
}

inline vkError_t vkGetDeviceProperties(vkDeviceProp_t* prop, int device) {
    try {
        *prop = g_vulkanContext.getDeviceProperties(device);
        return VK_SUCCESS;
    } catch (...) {
        return VK_ERROR_UNKNOWN;
    }
}

inline const char* vkGetErrorString(vkError_t error) {
    // Return a basic error string
    if (error == VK_SUCCESS) return "VK_SUCCESS";
    if (error == VK_ERROR_OUT_OF_DEVICE_MEMORY) return "VK_ERROR_OUT_OF_DEVICE_MEMORY";
    if (error == VK_ERROR_OUT_OF_HOST_MEMORY) return "VK_ERROR_OUT_OF_HOST_MEMORY";
    return "VK_ERROR_UNKNOWN";
}

inline vkError_t vkGetLastError() {
    // Vulkan doesn't have a "last error" concept like CUDA
    // Return success by default
    return VK_SUCCESS;
}

// Stream management (defined in VulkanStream.hpp)
inline vkError_t vkStreamCreate(vkStream_t* stream);
inline vkError_t vkStreamDestroy(vkStream_t stream);
inline vkError_t vkStreamSynchronize(vkStream_t stream);
inline vkError_t vkStreamWaitEvent(vkStream_t stream, vkEvent_t event);
inline vkError_t vkStreamAddCallback(vkStream_t stream, void (*callback)(void*), void* userData);

// Event management
inline vkError_t vkEventCreate(vkEvent_t* event) {
    try {
        *event = g_vulkanContext.createFence(false);
        return VK_SUCCESS;
    } catch (...) {
        return VK_ERROR_UNKNOWN;
    }
}

inline vkError_t vkEventDestroy(vkEvent_t event) {
    try {
        g_vulkanContext.destroyFence(event);
        return VK_SUCCESS;
    } catch (...) {
        return VK_ERROR_UNKNOWN;
    }
}

inline vkError_t vkEventSynchronize(vkEvent_t event) {
    try {
        g_vulkanContext.waitForFence(event);
        return VK_SUCCESS;
    } catch (...) {
        return VK_ERROR_UNKNOWN;
    }
}

inline vkError_t vkEventRecord(vkEvent_t event, vkStream_t stream);

inline vkError_t vkEventElapsedTime(float* ms, vkEvent_t start, vkEvent_t end) {
    // TODO: Implement timing with timestamps
    *ms = 0.0f;
    return VK_SUCCESS;
}

// Device configuration (no-ops for Vulkan)
inline vkError_t vkDeviceSetCacheConfig(vkFuncCache config) {
    return VK_SUCCESS;
}

inline vkError_t vkDeviceEnablePeerAccess(int peerDevice, unsigned int flags) {
    // TODO: Implement peer access
    return VK_ERROR_FEATURE_NOT_PRESENT;
}

// Memory operations
inline vkError_t vkMemGetInfo(size_t* free, size_t* total) {
    try {
        VulkanDevice* dev = g_vulkanContext.getDevice();
        // Get heap size for device local memory
        *total = 0;
        *free = 0;
        for (uint32_t i = 0; i < dev->memoryProperties.memoryHeapCount; i++) {
            if (dev->memoryProperties.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
                *total += dev->memoryProperties.memoryHeaps[i].size;
                *free += dev->memoryProperties.memoryHeaps[i].size; // Approximation
            }
        }
        return VK_SUCCESS;
    } catch (...) {
        return VK_ERROR_UNKNOWN;
    }
}

inline vkError_t vkMemset(vkDeviceptr_t ptr, int value, size_t size) {
    try {
        VkCommandBuffer cmd = g_vulkanContext.beginSingleTimeCommands();
        vkCmdFillBuffer(cmd, ptr->buffer, 0, size, static_cast<uint32_t>(value));
        g_vulkanContext.endSingleTimeCommands(cmd);
        return VK_SUCCESS;
    } catch (...) {
        return VK_ERROR_UNKNOWN;
    }
}

inline vkError_t vkMemsetAsync(vkDeviceptr_t ptr, int value, size_t size, vkStream_t stream);

// Host memory allocation
inline vkError_t vkHostMalloc(void** ptr, size_t size) {
    try {
        *ptr = aligned_alloc(64, size);
        if (!*ptr) return VK_ERROR_OUT_OF_HOST_MEMORY;
        return VK_SUCCESS;
    } catch (...) {
        return VK_ERROR_UNKNOWN;
    }
}

inline vkError_t vkHostFree(void* ptr) {
    try {
        if (ptr) free(ptr);
        return VK_SUCCESS;
    } catch (...) {
        return VK_ERROR_UNKNOWN;
    }
}

} // namespace vk_backend

#endif // VULKAN_PREPROCESSOR_HPP
