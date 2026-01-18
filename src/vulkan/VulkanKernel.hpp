#ifndef VULKAN_KERNEL_HPP
#define VULKAN_KERNEL_HPP

#include "VulkanContext.hpp"
#include "VulkanStream.hpp"
#include <map>
#include <string>
#include <vector>
#include <fstream>

namespace vk_backend {

// Kernel descriptor for managing compute pipelines
class VulkanKernelDescriptor {
private:
    ComputePipeline pipeline;
    int deviceIdx;
    std::string kernelName;
    std::vector<VkDescriptorSetLayoutBinding> bindings;

public:
    VulkanKernelDescriptor(const std::string& name, const std::vector<uint32_t>& spirvCode,
                           const std::vector<VkDescriptorSetLayoutBinding>& layoutBindings,
                           int devIdx = -1)
        : deviceIdx(devIdx), kernelName(name), bindings(layoutBindings) {
        pipeline = g_vulkanContext.createComputePipeline(spirvCode, bindings, deviceIdx);
    }

    ~VulkanKernelDescriptor() {
        g_vulkanContext.destroyPipeline(pipeline, deviceIdx);
    }

    const ComputePipeline& getPipeline() const { return pipeline; }
    const std::string& getName() const { return kernelName; }
};

// Kernel registry to manage all kernels
class VulkanKernelRegistry {
private:
    std::map<std::string, std::shared_ptr<VulkanKernelDescriptor>> kernels;
    std::mutex mutex;

public:
    static VulkanKernelRegistry& getInstance() {
        static VulkanKernelRegistry instance;
        return instance;
    }

    void registerKernel(const std::string& name, const std::vector<uint32_t>& spirvCode,
                       const std::vector<VkDescriptorSetLayoutBinding>& bindings, int deviceIdx = -1) {
        std::lock_guard<std::mutex> lock(mutex);
        std::string key = name + "_dev" + std::to_string(deviceIdx);
        kernels[key] = std::make_shared<VulkanKernelDescriptor>(name, spirvCode, bindings, deviceIdx);
    }

    std::shared_ptr<VulkanKernelDescriptor> getKernel(const std::string& name, int deviceIdx = -1) {
        std::lock_guard<std::mutex> lock(mutex);
        std::string key = name + "_dev" + std::to_string(deviceIdx);
        auto it = kernels.find(key);
        if (it != kernels.end()) {
            return it->second;
        }
        return nullptr;
    }

    void clear() {
        std::lock_guard<std::mutex> lock(mutex);
        kernels.clear();
    }
};

// Push constants structure for passing simple parameters
struct KernelPushConstants {
    int64_t param0;
    int64_t param1;
    int64_t param2;
    int64_t param3;
};

// Kernel launcher helper
class VulkanKernelLauncher {
private:
    std::shared_ptr<VulkanKernelDescriptor> kernel;
    VulkanStream* stream;
    std::vector<VulkanBuffer*> buffers;
    KernelPushConstants pushConstants;
    uint32_t gridX, gridY, gridZ;
    uint32_t blockX, blockY, blockZ;

public:
    VulkanKernelLauncher(const std::string& kernelName, VulkanStream* str = nullptr)
        : stream(str), gridX(1), gridY(1), gridZ(1), blockX(1), blockY(1), blockZ(1) {
        int deviceIdx = stream ? stream->getDeviceIdx() : -1;
        kernel = VulkanKernelRegistry::getInstance().getKernel(kernelName, deviceIdx);
        if (!kernel) {
            throw VshipError(HIPError, __FILE__, __LINE__, "Kernel not found: " + kernelName);
        }
        memset(&pushConstants, 0, sizeof(pushConstants));
    }

    VulkanKernelLauncher& setGrid(uint32_t x, uint32_t y = 1, uint32_t z = 1) {
        gridX = x;
        gridY = y;
        gridZ = z;
        return *this;
    }

    VulkanKernelLauncher& setBlock(uint32_t x, uint32_t y = 1, uint32_t z = 1) {
        blockX = x;
        blockY = y;
        blockZ = z;
        return *this;
    }

    VulkanKernelLauncher& addBuffer(VulkanBuffer* buffer) {
        buffers.push_back(buffer);
        return *this;
    }

    VulkanKernelLauncher& setPushConstant(int index, int64_t value) {
        switch (index) {
            case 0: pushConstants.param0 = value; break;
            case 1: pushConstants.param1 = value; break;
            case 2: pushConstants.param2 = value; break;
            case 3: pushConstants.param3 = value; break;
        }
        return *this;
    }

    void dispatch() {
        VulkanDevice* dev = g_vulkanContext.getDevice(stream ? stream->getDeviceIdx() : -1);
        const ComputePipeline& pipeline = kernel->getPipeline();

        // Create descriptor pool
        std::vector<VkDescriptorPoolSize> poolSizes;
        VkDescriptorPoolSize poolSize{};
        poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        poolSize.descriptorCount = static_cast<uint32_t>(buffers.size());
        poolSizes.push_back(poolSize);

        VkDescriptorPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
        poolInfo.pPoolSizes = poolSizes.data();
        poolInfo.maxSets = 1;

        VkDescriptorPool descriptorPool;
        VK_CHECK(vkCreateDescriptorPool(dev->device, &poolInfo, nullptr, &descriptorPool));

        // Allocate descriptor set
        VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = descriptorPool;
        allocInfo.descriptorSetCount = 1;
        allocInfo.pSetLayouts = &pipeline.descriptorSetLayout;

        VkDescriptorSet descriptorSet;
        VK_CHECK(vkAllocateDescriptorSets(dev->device, &allocInfo, &descriptorSet));

        // Update descriptor set with buffers
        std::vector<VkDescriptorBufferInfo> bufferInfos;
        std::vector<VkWriteDescriptorSet> descriptorWrites;

        for (size_t i = 0; i < buffers.size(); i++) {
            VkDescriptorBufferInfo bufferInfo{};
            bufferInfo.buffer = buffers[i]->buffer;
            bufferInfo.offset = 0;
            bufferInfo.range = buffers[i]->size;
            bufferInfos.push_back(bufferInfo);

            VkWriteDescriptorSet descriptorWrite{};
            descriptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrite.dstSet = descriptorSet;
            descriptorWrite.dstBinding = static_cast<uint32_t>(i);
            descriptorWrite.dstArrayElement = 0;
            descriptorWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            descriptorWrite.descriptorCount = 1;
            descriptorWrite.pBufferInfo = &bufferInfos[i];
            descriptorWrites.push_back(descriptorWrite);
        }

        vkUpdateDescriptorSets(dev->device, static_cast<uint32_t>(descriptorWrites.size()),
                              descriptorWrites.data(), 0, nullptr);

        // Get or begin command buffer
        VkCommandBuffer cmd = stream ? stream->getCommandBuffer() : g_vulkanContext.beginSingleTimeCommands();

        // Bind pipeline
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.pipeline);

        // Bind descriptor sets
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.layout,
                               0, 1, &descriptorSet, 0, nullptr);

        // Dispatch compute
        vkCmdDispatchBase(cmd, 0, 0, 0, gridX, gridY, gridZ);

        // Submit if not using stream
        if (!stream) {
            g_vulkanContext.endSingleTimeCommands(cmd);
        }

        // Clean up descriptor pool (should be delayed until after execution)
        // For now, we'll leak this - in production, track and clean up later
        // vkDestroyDescriptorPool(dev->device, descriptorPool, nullptr);
    }
};

// Helper to load SPIR-V from file
inline std::vector<uint32_t> loadSPIRV(const std::string& filename) {
    std::ifstream file(filename, std::ios::ate | std::ios::binary);

    if (!file.is_open()) {
        throw VshipError(BadPath, __FILE__, __LINE__, "Failed to open shader file: " + filename);
    }

    size_t fileSize = static_cast<size_t>(file.tellg());
    std::vector<uint32_t> buffer(fileSize / sizeof(uint32_t));

    file.seekg(0);
    file.read(reinterpret_cast<char*>(buffer.data()), fileSize);
    file.close();

    return buffer;
}

// Macro to help with kernel dispatch (similar to CUDA's <<<>>>)
#define LAUNCH_KERNEL(kernelName, grid, block, stream, ...) \
    VulkanKernelLauncher(#kernelName, stream) \
        .setGrid(grid.x, grid.y, grid.z) \
        .setBlock(block.x, block.y, block.z) \
        __VA_ARGS__ \
        .dispatch()

} // namespace vk_backend

#endif // VULKAN_KERNEL_HPP
