#ifndef KLARTRAUM_VULKAN_TENSOR_HPP
#define KLARTRAUM_VULKAN_TENSOR_HPP

#include "klartraum/vulkan_buffer.hpp"


namespace klartraum {

template <typename T>
class VulkanTensor {
public:
    VulkanTensor(VulkanContext& kernel, uint32_t batch, uint32_t depth=1, uint32_t height=1, uint32_t width=1) :
        buffer(kernel, batch * depth * height * width),
        batch(batch),
        depth(depth),
        height(height),
        width(width)
    {

    }

private:
    VulkanBuffer<typename T> buffer;
    uint32_t batch;
    uint32_t depth;
    uint32_t height;
    uint32_t width;
};

} // namespace klartraum

#endif // KLARTRAUM_VULKAN_TENSOR_HPP