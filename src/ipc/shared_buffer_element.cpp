// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#include "klartraum/ipc/shared_buffer_element.hpp"

#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace klartraum::ipc {

SharedBufferElement::SharedBufferElement(size_t byteSize)
    : region(SharedMemoryRegion::create(byteSize)),
      byteSize(byteSize) {
    if (byteSize == 0) {
        throw std::invalid_argument("SharedBufferElement: size must be greater than 0");
    }
}

SharedBufferElement::~SharedBufferElement() {
    if (vulkanContext != nullptr) {
        vkDestroyBuffer(vulkanContext->getDevice(), buffer, nullptr);
        vkFreeMemory(vulkanContext->getDevice(), memory, nullptr);
    }
}

void SharedBufferElement::checkInput(ComputeGraphElementPtr input, int index) {
    throw std::runtime_error("SharedBufferElement has no inputs");
}

void SharedBufferElement::_setup(VulkanContext& vulkanContext, uint32_t numberPaths, VkQueue setupQueue) {
    initialized = true;
    // Shared by all graphs and paths: imported once.
    if (this->vulkanContext != nullptr) {
        return;
    }
    if (!vulkanContext.isExternalMemoryHostSupported()) {
        throw std::runtime_error("SharedBufferElement: running locally needs VK_EXT_external_memory_host");
    }
    const VkDeviceSize alignment = vulkanContext.getMinImportedHostPointerAlignment();
    const VkDeviceSize allocationSize = (byteSize + alignment - 1) / alignment * alignment;
    if (allocationSize > region.size() || reinterpret_cast<uintptr_t>(region.data()) % alignment != 0) {
        throw std::runtime_error("SharedBufferElement: the region does not meet the import alignment");
    }
    VkDevice device = vulkanContext.getDevice();

    VkExternalMemoryBufferCreateInfo externalInfo{};
    externalInfo.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
    externalInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;
    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.pNext = &externalInfo;
    bufferInfo.size = byteSize;
    bufferInfo.usage =
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
        throw std::runtime_error("SharedBufferElement: failed to create the buffer");
    }

    VkMemoryHostPointerPropertiesEXT pointerProperties{};
    pointerProperties.sType = VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT;
    VkMemoryRequirements requirements;
    vkGetBufferMemoryRequirements(device, buffer, &requirements);
    if (vulkanContext.getMemoryHostPointerPropertiesEXT()(device,
                                                          VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT,
                                                          region.data(), &pointerProperties) != VK_SUCCESS) {
        vkDestroyBuffer(device, buffer, nullptr);
        throw std::runtime_error("SharedBufferElement: the region cannot be imported");
    }
    const uint32_t typeBits = pointerProperties.memoryTypeBits & requirements.memoryTypeBits;

    VkImportMemoryHostPointerInfoEXT importInfo{};
    importInfo.sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT;
    importInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;
    importInfo.pHostPointer = region.data();
    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.pNext = &importInfo;
    allocInfo.allocationSize = std::max(allocationSize, requirements.size);
    try {
        allocInfo.memoryTypeIndex = vulkanContext.findMemoryType(typeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                                                               VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    } catch (const std::runtime_error&) {
        vkDestroyBuffer(device, buffer, nullptr);
        throw;
    }
    if (vkAllocateMemory(device, &allocInfo, nullptr, &memory) != VK_SUCCESS) {
        vkDestroyBuffer(device, buffer, nullptr);
        throw std::runtime_error("SharedBufferElement: failed to import the region");
    }
    vkBindBufferMemory(device, buffer, memory, 0);
    this->vulkanContext = &vulkanContext;
}

VkBuffer& SharedBufferElement::getVkBuffer(uint32_t pathId) {
    if (buffer == VK_NULL_HANDLE) {
        throw std::runtime_error("SharedBufferElement: no buffer before a local setup");
    }
    return buffer;
}

} // namespace klartraum::ipc
