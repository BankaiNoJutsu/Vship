#ifndef VULKAN_CONTEXT_HPP
#define VULKAN_CONTEXT_HPP

#include <vulkan/vulkan.h>
#include <vector>
#include <string>
#include <memory>
#include <map>
#include <cstring>
#include "../util/VshipExceptions.hpp"

namespace vk_backend {

// Vulkan error checking macro
#define VK_CHECK(x) \
{ \
    VkResult err_vk = x; \
    if (err_vk != VK_SUCCESS) { \
        throw VshipError(HIPError, __FILE__, __LINE__, std::string("Vulkan error: ") + std::to_string(err_vk)); \
    } \
}

// Queue family indices
struct QueueFamilyIndices {
    uint32_t computeFamily = UINT32_MAX;
    uint32_t transferFamily = UINT32_MAX;

    bool isComplete() const {
        return computeFamily != UINT32_MAX;
    }
};

// Vulkan device wrapper
class VulkanDevice {
public:
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue computeQueue = VK_NULL_HANDLE;
    VkQueue transferQueue = VK_NULL_HANDLE;
    QueueFamilyIndices queueFamilyIndices;
    VkPhysicalDeviceProperties deviceProperties;
    VkPhysicalDeviceMemoryProperties memoryProperties;

    VulkanDevice() = default;
    ~VulkanDevice() {
        cleanup();
    }

    VulkanDevice(const VulkanDevice&) = delete;
    VulkanDevice& operator=(const VulkanDevice&) = delete;

    void cleanup() {
        if (device != VK_NULL_HANDLE) {
            vkDestroyDevice(device, nullptr);
            device = VK_NULL_HANDLE;
        }
        if (instance != VK_NULL_HANDLE) {
            vkDestroyInstance(instance, nullptr);
            instance = VK_NULL_HANDLE;
        }
    }
};

// Buffer wrapper
struct VulkanBuffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
    void* mapped = nullptr;

    VulkanBuffer() = default;
    VulkanBuffer(VkBuffer buf, VkDeviceMemory mem, VkDeviceSize sz)
        : buffer(buf), memory(mem), size(sz) {}
};

// Command buffer pool
struct CommandBufferPool {
    VkCommandPool pool = VK_NULL_HANDLE;
    std::vector<VkCommandBuffer> buffers;
};

// Compute pipeline wrapper
struct ComputePipeline {
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    std::vector<VkDescriptorSet> descriptorSets;
    VkShaderModule shaderModule = VK_NULL_HANDLE;
};

// Vulkan context manager - manages all Vulkan resources
class VulkanContext {
private:
    std::vector<std::unique_ptr<VulkanDevice>> devices;
    int currentDeviceIdx = 0;
    bool initialized = false;

    // Per-device resources
    std::map<int, std::vector<VulkanBuffer>> deviceBuffers;
    std::map<int, std::vector<CommandBufferPool>> deviceCommandPools;
    std::map<int, std::vector<ComputePipeline>> devicePipelines;
    std::map<int, std::vector<VkFence>> deviceFences;
    std::map<int, std::vector<VkSemaphore>> deviceSemaphores;

public:
    VulkanContext() = default;
    ~VulkanContext() {
        cleanup();
    }

    void initialize();
    void cleanup();

    int getDeviceCount() const { return static_cast<int>(devices.size()); }
    void setDevice(int deviceIdx);
    int getCurrentDevice() const { return currentDeviceIdx; }
    VulkanDevice* getDevice(int idx = -1);

    // Buffer management
    VulkanBuffer createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                              VkMemoryPropertyFlags properties, int deviceIdx = -1);
    void destroyBuffer(VulkanBuffer& buffer, int deviceIdx = -1);
    void* mapBuffer(VulkanBuffer& buffer, int deviceIdx = -1);
    void unmapBuffer(VulkanBuffer& buffer, int deviceIdx = -1);

    // Command buffer management
    VkCommandBuffer beginSingleTimeCommands(int deviceIdx = -1);
    void endSingleTimeCommands(VkCommandBuffer commandBuffer, int deviceIdx = -1);

    // Memory operations
    void copyBuffer(VulkanBuffer& src, VulkanBuffer& dst, VkDeviceSize size, int deviceIdx = -1);
    void copyBufferToHost(void* hostPtr, VulkanBuffer& deviceBuffer, VkDeviceSize size, int deviceIdx = -1);
    void copyHostToBuffer(VulkanBuffer& deviceBuffer, const void* hostPtr, VkDeviceSize size, int deviceIdx = -1);

    // Pipeline management
    ComputePipeline createComputePipeline(const std::vector<uint32_t>& spirvCode,
                                          const std::vector<VkDescriptorSetLayoutBinding>& bindings,
                                          int deviceIdx = -1);
    void destroyPipeline(ComputePipeline& pipeline, int deviceIdx = -1);

    // Synchronization
    VkFence createFence(bool signaled = false, int deviceIdx = -1);
    void destroyFence(VkFence fence, int deviceIdx = -1);
    void waitForFence(VkFence fence, int deviceIdx = -1);
    void resetFence(VkFence fence, int deviceIdx = -1);

    VkSemaphore createSemaphore(int deviceIdx = -1);
    void destroySemaphore(VkSemaphore semaphore, int deviceIdx = -1);

    // Device info
    std::string getDeviceName(int deviceIdx = -1);
    VkPhysicalDeviceProperties getDeviceProperties(int deviceIdx = -1);

private:
    void createInstance();
    void pickPhysicalDevices();
    void createLogicalDevice(VulkanDevice* device);
    QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device);
    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties, int deviceIdx = -1);
};

// Global Vulkan context
extern VulkanContext g_vulkanContext;

} // namespace vk_backend

#endif // VULKAN_CONTEXT_HPP
