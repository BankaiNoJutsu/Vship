#include "VulkanContext.hpp"
#include <stdexcept>
#include <iostream>
#include <cstring>

namespace vk_backend {

VulkanContext g_vulkanContext;

void VulkanContext::initialize() {
    if (initialized) return;

    createInstance();
    pickPhysicalDevices();

    // Create logical device for each physical device
    for (auto& device : devices) {
        createLogicalDevice(device.get());
    }

    initialized = true;
}

void VulkanContext::cleanup() {
    if (!initialized) return;

    // Clean up all device resources
    for (size_t i = 0; i < devices.size(); ++i) {
        setDevice(i);

        // Destroy buffers
        if (deviceBuffers.find(i) != deviceBuffers.end()) {
            for (auto& buffer : deviceBuffers[i]) {
                destroyBuffer(buffer, i);
            }
            deviceBuffers[i].clear();
        }

        // Destroy pipelines
        if (devicePipelines.find(i) != devicePipelines.end()) {
            for (auto& pipeline : devicePipelines[i]) {
                destroyPipeline(pipeline, i);
            }
            devicePipelines[i].clear();
        }

        // Destroy command pools
        if (deviceCommandPools.find(i) != deviceCommandPools.end()) {
            VulkanDevice* dev = getDevice(i);
            for (auto& cmdPool : deviceCommandPools[i]) {
                if (cmdPool.pool != VK_NULL_HANDLE) {
                    vkDestroyCommandPool(dev->device, cmdPool.pool, nullptr);
                }
            }
            deviceCommandPools[i].clear();
        }

        // Destroy fences
        if (deviceFences.find(i) != deviceFences.end()) {
            VulkanDevice* dev = getDevice(i);
            for (auto fence : deviceFences[i]) {
                if (fence != VK_NULL_HANDLE) {
                    vkDestroyFence(dev->device, fence, nullptr);
                }
            }
            deviceFences[i].clear();
        }

        // Destroy semaphores
        if (deviceSemaphores.find(i) != deviceSemaphores.end()) {
            VulkanDevice* dev = getDevice(i);
            for (auto semaphore : deviceSemaphores[i]) {
                if (semaphore != VK_NULL_HANDLE) {
                    vkDestroySemaphore(dev->device, semaphore, nullptr);
                }
            }
            deviceSemaphores[i].clear();
        }
    }

    devices.clear();
    initialized = false;
}

void VulkanContext::createInstance() {
    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "Vship";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "Vship Vulkan Backend";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_2;

    VkInstanceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;

    // No validation layers or extensions for compute-only application
    createInfo.enabledLayerCount = 0;
    createInfo.enabledExtensionCount = 0;

    VkInstance instance;
    VK_CHECK(vkCreateInstance(&createInfo, nullptr, &instance));

    // Store instance in first device (will be used by all)
    if (devices.empty()) {
        devices.push_back(std::make_unique<VulkanDevice>());
    }
    devices[0]->instance = instance;
}

void VulkanContext::pickPhysicalDevices() {
    uint32_t deviceCount = 0;
    VkInstance instance = devices[0]->instance;
    vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);

    if (deviceCount == 0) {
        throw VshipError(NoDeviceDetected, __FILE__, __LINE__);
    }

    std::vector<VkPhysicalDevice> physicalDevices(deviceCount);
    vkEnumeratePhysicalDevices(instance, &deviceCount, physicalDevices.data());

    // Create a VulkanDevice for each physical device
    devices.clear();
    for (auto physDevice : physicalDevices) {
        auto device = std::make_unique<VulkanDevice>();
        device->instance = instance;
        device->physicalDevice = physDevice;
        device->queueFamilyIndices = findQueueFamilies(physDevice);

        // Skip devices without compute queue
        if (!device->queueFamilyIndices.isComplete()) {
            continue;
        }

        vkGetPhysicalDeviceProperties(physDevice, &device->deviceProperties);
        vkGetPhysicalDeviceMemoryProperties(physDevice, &device->memoryProperties);

        devices.push_back(std::move(device));
    }

    if (devices.empty()) {
        throw VshipError(NoDeviceDetected, __FILE__, __LINE__);
    }
}

QueueFamilyIndices VulkanContext::findQueueFamilies(VkPhysicalDevice device) {
    QueueFamilyIndices indices;

    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);

    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

    for (uint32_t i = 0; i < queueFamilyCount; i++) {
        // Look for compute queue
        if (queueFamilies[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            if (indices.computeFamily == UINT32_MAX) {
                indices.computeFamily = i;
            }
        }

        // Look for transfer queue (prefer dedicated, but can use compute queue)
        if (queueFamilies[i].queueFlags & VK_QUEUE_TRANSFER_BIT) {
            if (indices.transferFamily == UINT32_MAX) {
                indices.transferFamily = i;
            }
            // Prefer dedicated transfer queue
            if (!(queueFamilies[i].queueFlags & VK_QUEUE_COMPUTE_BIT) &&
                !(queueFamilies[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) {
                indices.transferFamily = i;
            }
        }
    }

    // Use compute queue for transfer if no dedicated transfer queue
    if (indices.transferFamily == UINT32_MAX) {
        indices.transferFamily = indices.computeFamily;
    }

    return indices;
}

void VulkanContext::createLogicalDevice(VulkanDevice* device) {
    std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
    std::set<uint32_t> uniqueQueueFamilies = {
        device->queueFamilyIndices.computeFamily,
        device->queueFamilyIndices.transferFamily
    };

    float queuePriority = 1.0f;
    for (uint32_t queueFamily : uniqueQueueFamilies) {
        VkDeviceQueueCreateInfo queueCreateInfo{};
        queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queueCreateInfo.queueFamilyIndex = queueFamily;
        queueCreateInfo.queueCount = 1;
        queueCreateInfo.pQueuePriorities = &queuePriority;
        queueCreateInfos.push_back(queueCreateInfo);
    }

    VkPhysicalDeviceFeatures deviceFeatures{};

    // Enable 16-bit storage for half precision support
    VkPhysicalDevice16BitStorageFeatures storage16BitFeatures{};
    storage16BitFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES;
    storage16BitFeatures.storageBuffer16BitAccess = VK_TRUE;

    VkDeviceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    createInfo.pQueueCreateInfos = queueCreateInfos.data();
    createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
    createInfo.pEnabledFeatures = &deviceFeatures;
    createInfo.pNext = &storage16BitFeatures;

    VK_CHECK(vkCreateDevice(device->physicalDevice, &createInfo, nullptr, &device->device));

    // Get queue handles
    vkGetDeviceQueue(device->device, device->queueFamilyIndices.computeFamily, 0, &device->computeQueue);
    vkGetDeviceQueue(device->device, device->queueFamilyIndices.transferFamily, 0, &device->transferQueue);
}

void VulkanContext::setDevice(int deviceIdx) {
    if (deviceIdx < 0 || deviceIdx >= static_cast<int>(devices.size())) {
        throw VshipError(BadDeviceArgument, __FILE__, __LINE__);
    }
    currentDeviceIdx = deviceIdx;
}

VulkanDevice* VulkanContext::getDevice(int idx) {
    int deviceIdx = (idx < 0) ? currentDeviceIdx : idx;
    if (deviceIdx < 0 || deviceIdx >= static_cast<int>(devices.size())) {
        throw VshipError(BadDeviceArgument, __FILE__, __LINE__);
    }
    return devices[deviceIdx].get();
}

uint32_t VulkanContext::findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties, int deviceIdx) {
    VulkanDevice* dev = getDevice(deviceIdx);

    for (uint32_t i = 0; i < dev->memoryProperties.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) &&
            (dev->memoryProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }

    throw VshipError(OutOfVRAM, __FILE__, __LINE__, "Failed to find suitable memory type");
}

VulkanBuffer VulkanContext::createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                                          VkMemoryPropertyFlags properties, int deviceIdx) {
    VulkanDevice* dev = getDevice(deviceIdx);

    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkBuffer buffer;
    VK_CHECK(vkCreateBuffer(dev->device, &bufferInfo, nullptr, &buffer));

    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(dev->device, buffer, &memRequirements);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties, deviceIdx);

    VkDeviceMemory memory;
    VkResult result = vkAllocateMemory(dev->device, &allocInfo, nullptr, &memory);
    if (result != VK_SUCCESS) {
        vkDestroyBuffer(dev->device, buffer, nullptr);
        throw VshipError(OutOfVRAM, __FILE__, __LINE__);
    }

    VK_CHECK(vkBindBufferMemory(dev->device, buffer, memory, 0));

    return VulkanBuffer(buffer, memory, size);
}

void VulkanContext::destroyBuffer(VulkanBuffer& buffer, int deviceIdx) {
    VulkanDevice* dev = getDevice(deviceIdx);

    if (buffer.mapped) {
        vkUnmapMemory(dev->device, buffer.memory);
        buffer.mapped = nullptr;
    }

    if (buffer.buffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(dev->device, buffer.buffer, nullptr);
        buffer.buffer = VK_NULL_HANDLE;
    }

    if (buffer.memory != VK_NULL_HANDLE) {
        vkFreeMemory(dev->device, buffer.memory, nullptr);
        buffer.memory = VK_NULL_HANDLE;
    }

    buffer.size = 0;
}

void* VulkanContext::mapBuffer(VulkanBuffer& buffer, int deviceIdx) {
    if (buffer.mapped) {
        return buffer.mapped;
    }

    VulkanDevice* dev = getDevice(deviceIdx);
    VK_CHECK(vkMapMemory(dev->device, buffer.memory, 0, buffer.size, 0, &buffer.mapped));
    return buffer.mapped;
}

void VulkanContext::unmapBuffer(VulkanBuffer& buffer, int deviceIdx) {
    if (!buffer.mapped) return;

    VulkanDevice* dev = getDevice(deviceIdx);
    vkUnmapMemory(dev->device, buffer.memory);
    buffer.mapped = nullptr;
}

VkCommandBuffer VulkanContext::beginSingleTimeCommands(int deviceIdx) {
    VulkanDevice* dev = getDevice(deviceIdx);

    // Create command pool if it doesn't exist for this device
    if (deviceCommandPools.find(deviceIdx) == deviceCommandPools.end() ||
        deviceCommandPools[deviceIdx].empty()) {

        CommandBufferPool cmdPool;
        VkCommandPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolInfo.queueFamilyIndex = dev->queueFamilyIndices.computeFamily;
        poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

        VK_CHECK(vkCreateCommandPool(dev->device, &poolInfo, nullptr, &cmdPool.pool));
        deviceCommandPools[deviceIdx].push_back(cmdPool);
    }

    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandPool = deviceCommandPools[deviceIdx][0].pool;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer commandBuffer;
    VK_CHECK(vkAllocateCommandBuffers(dev->device, &allocInfo, &commandBuffer));

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    VK_CHECK(vkBeginCommandBuffer(commandBuffer, &beginInfo));

    return commandBuffer;
}

void VulkanContext::endSingleTimeCommands(VkCommandBuffer commandBuffer, int deviceIdx) {
    VulkanDevice* dev = getDevice(deviceIdx);

    VK_CHECK(vkEndCommandBuffer(commandBuffer));

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;

    VK_CHECK(vkQueueSubmit(dev->computeQueue, 1, &submitInfo, VK_NULL_HANDLE));
    VK_CHECK(vkQueueWaitIdle(dev->computeQueue));

    vkFreeCommandBuffers(dev->device, deviceCommandPools[deviceIdx][0].pool, 1, &commandBuffer);
}

void VulkanContext::copyBuffer(VulkanBuffer& src, VulkanBuffer& dst, VkDeviceSize size, int deviceIdx) {
    VkCommandBuffer commandBuffer = beginSingleTimeCommands(deviceIdx);

    VkBufferCopy copyRegion{};
    copyRegion.size = size;
    vkCmdCopyBuffer(commandBuffer, src.buffer, dst.buffer, 1, &copyRegion);

    endSingleTimeCommands(commandBuffer, deviceIdx);
}

void VulkanContext::copyBufferToHost(void* hostPtr, VulkanBuffer& deviceBuffer, VkDeviceSize size, int deviceIdx) {
    // Create staging buffer
    VulkanBuffer stagingBuffer = createBuffer(
        size,
        VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        deviceIdx
    );

    // Copy device buffer to staging buffer
    copyBuffer(deviceBuffer, stagingBuffer, size, deviceIdx);

    // Map and copy to host
    void* data = mapBuffer(stagingBuffer, deviceIdx);
    memcpy(hostPtr, data, size);
    unmapBuffer(stagingBuffer, deviceIdx);

    destroyBuffer(stagingBuffer, deviceIdx);
}

void VulkanContext::copyHostToBuffer(VulkanBuffer& deviceBuffer, const void* hostPtr, VkDeviceSize size, int deviceIdx) {
    // Create staging buffer
    VulkanBuffer stagingBuffer = createBuffer(
        size,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        deviceIdx
    );

    // Map and copy from host
    void* data = mapBuffer(stagingBuffer, deviceIdx);
    memcpy(data, hostPtr, size);
    unmapBuffer(stagingBuffer, deviceIdx);

    // Copy staging buffer to device buffer
    copyBuffer(stagingBuffer, deviceBuffer, size, deviceIdx);

    destroyBuffer(stagingBuffer, deviceIdx);
}

ComputePipeline VulkanContext::createComputePipeline(const std::vector<uint32_t>& spirvCode,
                                                      const std::vector<VkDescriptorSetLayoutBinding>& bindings,
                                                      int deviceIdx) {
    VulkanDevice* dev = getDevice(deviceIdx);
    ComputePipeline pipeline;

    // Create shader module
    VkShaderModuleCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = spirvCode.size() * sizeof(uint32_t);
    createInfo.pCode = spirvCode.data();

    VK_CHECK(vkCreateShaderModule(dev->device, &createInfo, nullptr, &pipeline.shaderModule));

    // Create descriptor set layout
    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();

    VK_CHECK(vkCreateDescriptorSetLayout(dev->device, &layoutInfo, nullptr, &pipeline.descriptorSetLayout));

    // Create pipeline layout
    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &pipeline.descriptorSetLayout;

    VK_CHECK(vkCreatePipelineLayout(dev->device, &pipelineLayoutInfo, nullptr, &pipeline.layout));

    // Create compute pipeline
    VkComputePipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.layout = pipeline.layout;
    pipelineInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipelineInfo.stage.module = pipeline.shaderModule;
    pipelineInfo.stage.pName = "main";

    VK_CHECK(vkCreateComputePipelines(dev->device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline.pipeline));

    return pipeline;
}

void VulkanContext::destroyPipeline(ComputePipeline& pipeline, int deviceIdx) {
    VulkanDevice* dev = getDevice(deviceIdx);

    if (pipeline.pipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(dev->device, pipeline.pipeline, nullptr);
        pipeline.pipeline = VK_NULL_HANDLE;
    }

    if (pipeline.layout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(dev->device, pipeline.layout, nullptr);
        pipeline.layout = VK_NULL_HANDLE;
    }

    if (pipeline.descriptorSetLayout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(dev->device, pipeline.descriptorSetLayout, nullptr);
        pipeline.descriptorSetLayout = VK_NULL_HANDLE;
    }

    if (pipeline.descriptorPool != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(dev->device, pipeline.descriptorPool, nullptr);
        pipeline.descriptorPool = VK_NULL_HANDLE;
    }

    if (pipeline.shaderModule != VK_NULL_HANDLE) {
        vkDestroyShaderModule(dev->device, pipeline.shaderModule, nullptr);
        pipeline.shaderModule = VK_NULL_HANDLE;
    }
}

VkFence VulkanContext::createFence(bool signaled, int deviceIdx) {
    VulkanDevice* dev = getDevice(deviceIdx);

    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    if (signaled) {
        fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    }

    VkFence fence;
    VK_CHECK(vkCreateFence(dev->device, &fenceInfo, nullptr, &fence));

    return fence;
}

void VulkanContext::destroyFence(VkFence fence, int deviceIdx) {
    VulkanDevice* dev = getDevice(deviceIdx);
    if (fence != VK_NULL_HANDLE) {
        vkDestroyFence(dev->device, fence, nullptr);
    }
}

void VulkanContext::waitForFence(VkFence fence, int deviceIdx) {
    VulkanDevice* dev = getDevice(deviceIdx);
    VK_CHECK(vkWaitForFences(dev->device, 1, &fence, VK_TRUE, UINT64_MAX));
}

void VulkanContext::resetFence(VkFence fence, int deviceIdx) {
    VulkanDevice* dev = getDevice(deviceIdx);
    VK_CHECK(vkResetFences(dev->device, 1, &fence));
}

VkSemaphore VulkanContext::createSemaphore(int deviceIdx) {
    VulkanDevice* dev = getDevice(deviceIdx);

    VkSemaphoreCreateInfo semaphoreInfo{};
    semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    VkSemaphore semaphore;
    VK_CHECK(vkCreateSemaphore(dev->device, &semaphoreInfo, nullptr, &semaphore));

    return semaphore;
}

void VulkanContext::destroySemaphore(VkSemaphore semaphore, int deviceIdx) {
    VulkanDevice* dev = getDevice(deviceIdx);
    if (semaphore != VK_NULL_HANDLE) {
        vkDestroySemaphore(dev->device, semaphore, nullptr);
    }
}

std::string VulkanContext::getDeviceName(int deviceIdx) {
    VulkanDevice* dev = getDevice(deviceIdx);
    return std::string(dev->deviceProperties.deviceName);
}

VkPhysicalDeviceProperties VulkanContext::getDeviceProperties(int deviceIdx) {
    VulkanDevice* dev = getDevice(deviceIdx);
    return dev->deviceProperties;
}

} // namespace vk_backend
