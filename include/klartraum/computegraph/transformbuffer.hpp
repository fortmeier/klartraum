// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#ifndef KLARTRAUM_COMPUTEGRAPH_TRANSFORMBUFFER_HPP
#define KLARTRAUM_COMPUTEGRAPH_TRANSFORMBUFFER_HPP

#include <array>
#include <cstdint>
#include <memory>

#include "klartraum/computegraph/bufferelement.hpp"
#include "klartraum/computegraph/generalcomputation.hpp"
#include "klartraum/vulkan_buffer.hpp"
#include "klartraum/vulkan_gaussian_splatting_types.hpp"

namespace klartraum {

/**
 * @brief Builds a transform buffer for GaussianTransform from seven scalars.
 *
 * Reads translation x, y, z, rotation pitch, yaw, roll in degrees (about X,
 * then Y, then Z) and a uniform scale, each from the first float of its
 * buffer, e.g. a HostFloat the CPU animates. Writes translation, scale, the
 * rotation quaternion and the rotation matrices of the three SH bands
 * (shaders/gaussians/make_transform.comp). One invocation per run.
 */
class TransformBuffer : public GeneralComputation<void> {
public:
    // Floats in a transform buffer (91 used).
    static constexpr uint32_t kSize = 96;

    TransformBuffer(VulkanContext& vulkanContext)
        : GeneralComputation<void>(vulkanContext, "shaders/gaussians/make_transform.comp.spv") {}

    const char* getType() const override { return "TransformBuffer"; }
};

/**
 * @brief Result of createTransformBuffer(): the element and the transform buffer it writes.
 */
struct TransformBufferResult {
    std::shared_ptr<TransformBuffer> element;
    BufferRef transform; // input 7 of the element
};

/**
 * @brief Creates a TransformBuffer from seven parameters in the order x, y, z, pitch, yaw, roll, scale.
 */
inline TransformBufferResult createTransformBuffer(VulkanContext& vulkanContext,
                                                   const std::array<BufferRef, 7>& parameters) {
    auto element = std::make_shared<TransformBuffer>(vulkanContext);
    for (int i = 0; i < 7; ++i) {
        parameters[i].connectTo(*element, i);
    }
    auto transform = std::make_shared<BufferElement<VulkanBuffer<float>>>(vulkanContext, TransformBuffer::kSize);
    transform->setName("Transform");
    element->setInput(transform, 7);
    element->setGroupCount(1, 1, 1);
    return {element, BufferRef{element, 7}};
}

} // namespace klartraum

#endif // KLARTRAUM_COMPUTEGRAPH_TRANSFORMBUFFER_HPP
