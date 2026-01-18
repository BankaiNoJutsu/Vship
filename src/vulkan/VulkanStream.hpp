#ifndef VULKAN_STREAM_HPP
#define VULKAN_STREAM_HPP

#include "VulkanContext.hpp"
#include <queue>
#include <functional>
#include <thread>
#include <mutex>
#include <condition_variable>

namespace vk_backend {

// Vulkan stream - manages asynchronous command execution
class VulkanStream {
private:
    VulkanDevice* device;
    int deviceIdx;
    VkCommandPool commandPool;
    std::vector<VkCommandBuffer> commandBuffers;
    VkFence fence;
    VkSemaphore semaphore;

    // Pending operations
    struct PendingOp {
        std::function<void(VkCommandBuffer)> record;
        std::function<void()> callback;
    };
    std::queue<PendingOp> pendingOps;
    std::mutex mutex;

    bool recording;
    VkCommandBuffer currentCommandBuffer;

public:
    VulkanStream(int devIdx = -1) : deviceIdx(devIdx), recording(false), currentCommandBuffer(VK_NULL_HANDLE) {
        device = g_vulkanContext.getDevice(deviceIdx);

        // Create command pool
        VkCommandPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolInfo.queueFamilyIndex = device->queueFamilyIndices.computeFamily;
        poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

        VK_CHECK(vkCreateCommandPool(device->device, &poolInfo, nullptr, &commandPool));

        // Allocate initial command buffers
        allocateCommandBuffer();

        // Create fence and semaphore
        fence = g_vulkanContext.createFence(true, deviceIdx);
        semaphore = g_vulkanContext.createSemaphore(deviceIdx);
    }

    ~VulkanStream() {
        synchronize();

        g_vulkanContext.destroyFence(fence, deviceIdx);
        g_vulkanContext.destroySemaphore(semaphore, deviceIdx);

        if (!commandBuffers.empty()) {
            vkFreeCommandBuffers(device->device, commandPool, static_cast<uint32_t>(commandBuffers.size()),
                                commandBuffers.data());
        }

        if (commandPool != VK_NULL_HANDLE) {
            vkDestroyCommandPool(device->device, commandPool, nullptr);
        }
    }

    void allocateCommandBuffer() {
        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool = commandPool;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = 1;

        VkCommandBuffer cmdBuf;
        VK_CHECK(vkAllocateCommandBuffers(device->device, &allocInfo, &cmdBuf));
        commandBuffers.push_back(cmdBuf);
    }

    VkCommandBuffer beginRecording() {
        std::lock_guard<std::mutex> lock(mutex);

        if (recording) {
            return currentCommandBuffer;
        }

        // Wait for fence before reusing command buffer
        g_vulkanContext.waitForFence(fence, deviceIdx);
        g_vulkanContext.resetFence(fence, deviceIdx);

        if (commandBuffers.empty()) {
            allocateCommandBuffer();
        }

        currentCommandBuffer = commandBuffers[0];

        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

        VK_CHECK(vkBeginCommandBuffer(currentCommandBuffer, &beginInfo));
        recording = true;

        return currentCommandBuffer;
    }

    void endRecording() {
        std::lock_guard<std::mutex> lock(mutex);

        if (!recording) return;

        VK_CHECK(vkEndCommandBuffer(currentCommandBuffer));
        recording = false;
    }

    void submit() {
        std::lock_guard<std::mutex> lock(mutex);

        if (recording) {
            VK_CHECK(vkEndCommandBuffer(currentCommandBuffer));
            recording = false;
        }

        if (currentCommandBuffer == VK_NULL_HANDLE) return;

        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &currentCommandBuffer;

        VK_CHECK(vkQueueSubmit(device->computeQueue, 1, &submitInfo, fence));
        currentCommandBuffer = VK_NULL_HANDLE;
    }

    void synchronize() {
        std::lock_guard<std::mutex> lock(mutex);

        if (recording) {
            VK_CHECK(vkEndCommandBuffer(currentCommandBuffer));
            recording = false;

            VkSubmitInfo submitInfo{};
            submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submitInfo.commandBufferCount = 1;
            submitInfo.pCommandBuffers = &currentCommandBuffer;

            VK_CHECK(vkQueueSubmit(device->computeQueue, 1, &submitInfo, fence));
            currentCommandBuffer = VK_NULL_HANDLE;
        }

        g_vulkanContext.waitForFence(fence, deviceIdx);
    }

    void waitForEvent(VkFence event) {
        // Add a pipeline barrier or wait in the host
        g_vulkanContext.waitForFence(event, deviceIdx);
    }

    void recordEvent(VkFence event) {
        // Submit current work and signal the fence
        submit();
        // The fence from submit will act as the event
    }

    VkCommandBuffer getCommandBuffer() {
        if (!recording) {
            return beginRecording();
        }
        return currentCommandBuffer;
    }

    VulkanDevice* getDevice() const { return device; }
    int getDeviceIdx() const { return deviceIdx; }
    VkFence getFence() const { return fence; }
};

// Implementation of async operations from VulkanPreprocessor.hpp

inline vkError_t vkStreamCreate(vkStream_t* stream) {
    try {
        *stream = new VulkanStream();
        return VK_SUCCESS;
    } catch (...) {
        return VK_ERROR_UNKNOWN;
    }
}

inline vkError_t vkStreamDestroy(vkStream_t stream) {
    try {
        if (stream) delete stream;
        return VK_SUCCESS;
    } catch (...) {
        return VK_ERROR_UNKNOWN;
    }
}

inline vkError_t vkStreamSynchronize(vkStream_t stream) {
    try {
        if (stream) stream->synchronize();
        return VK_SUCCESS;
    } catch (...) {
        return VK_ERROR_UNKNOWN;
    }
}

inline vkError_t vkStreamWaitEvent(vkStream_t stream, vkEvent_t event) {
    try {
        if (stream) stream->waitForEvent(event);
        return VK_SUCCESS;
    } catch (...) {
        return VK_ERROR_UNKNOWN;
    }
}

inline vkError_t vkStreamAddCallback(vkStream_t stream, void (*callback)(void*), void* userData) {
    try {
        if (stream) {
            stream->synchronize();
            if (callback) callback(userData);
        }
        return VK_SUCCESS;
    } catch (...) {
        return VK_ERROR_UNKNOWN;
    }
}

inline vkError_t vkEventRecord(vkEvent_t event, vkStream_t stream) {
    try {
        if (stream) stream->recordEvent(event);
        return VK_SUCCESS;
    } catch (...) {
        return VK_ERROR_UNKNOWN;
    }
}

inline vkError_t vkMemcpyDtoHAsync(void* dst, vkDeviceptr_t src, size_t size, vkStream_t stream) {
    try {
        if (stream) stream->synchronize();
        g_vulkanContext.copyBufferToHost(dst, *src, size, stream ? stream->getDeviceIdx() : -1);
        return VK_SUCCESS;
    } catch (...) {
        return VK_ERROR_UNKNOWN;
    }
}

inline vkError_t vkMemcpyHtoDAsync(vkDeviceptr_t dst, const void* src, size_t size, vkStream_t stream) {
    try {
        if (stream) stream->synchronize();
        g_vulkanContext.copyHostToBuffer(*dst, src, size, stream ? stream->getDeviceIdx() : -1);
        return VK_SUCCESS;
    } catch (...) {
        return VK_ERROR_UNKNOWN;
    }
}

inline vkError_t vkMemcpyDtoDAsync(vkDeviceptr_t dst, vkDeviceptr_t src, size_t size, vkStream_t stream) {
    try {
        if (stream) {
            VkCommandBuffer cmd = stream->getCommandBuffer();
            VkBufferCopy copyRegion{};
            copyRegion.size = size;
            vkCmdCopyBuffer(cmd, src->buffer, dst->buffer, 1, &copyRegion);
        } else {
            g_vulkanContext.copyBuffer(*src, *dst, size);
        }
        return VK_SUCCESS;
    } catch (...) {
        return VK_ERROR_UNKNOWN;
    }
}

inline vkError_t vkMallocAsync(vkDeviceptr_t* ptr, size_t size, vkStream_t stream) {
    // Vulkan doesn't have async allocation, just do sync allocation
    return vkMalloc(ptr, size);
}

inline vkError_t vkFreeAsync(vkDeviceptr_t ptr, vkStream_t stream) {
    // Vulkan doesn't have async free, synchronize and free
    if (stream) stream->synchronize();
    return vkFree(ptr);
}

inline vkError_t vkMemsetAsync(vkDeviceptr_t ptr, int value, size_t size, vkStream_t stream) {
    try {
        VkCommandBuffer cmd = stream ? stream->getCommandBuffer() : g_vulkanContext.beginSingleTimeCommands();
        vkCmdFillBuffer(cmd, ptr->buffer, 0, size, static_cast<uint32_t>(value));
        if (!stream) g_vulkanContext.endSingleTimeCommands(cmd);
        return VK_SUCCESS;
    } catch (...) {
        return VK_ERROR_UNKNOWN;
    }
}

} // namespace vk_backend

#endif // VULKAN_STREAM_HPP
