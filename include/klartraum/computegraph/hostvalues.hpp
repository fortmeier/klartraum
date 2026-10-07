// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#ifndef KLARTRAUM_COMPUTEGRAPH_HOSTVALUES_HPP
#define KLARTRAUM_COMPUTEGRAPH_HOSTVALUES_HPP

#include <cstdint>
#include <stdexcept>
#include <vector>

#include <glm/glm.hpp>

#include "klartraum/computegraph/bufferelement.hpp"
#include "klartraum/vulkan_buffer.hpp"

namespace klartraum {

/**
 * @brief A few values the CPU sets and shaders read, e.g. animation values.
 *
 * set() only stores the values on the CPU. Right before a path of the graph
 * is submitted, _update() copies them into that path's storage buffer, so a
 * value set between frames is seen by the next frame and paths still in
 * flight keep theirs. Shaders read it as an array of T.
 */
template <typename T>
class HostValues : public BufferElement<VulkanBuffer<T>> {
public:
    HostValues(VulkanContext& vulkanContext, std::vector<T> initial)
        : BufferElement<VulkanBuffer<T>>(vulkanContext, static_cast<uint32_t>(initial.size())),
          values_(std::move(initial)) {
        if (values_.empty()) {
            throw std::invalid_argument("HostValues: needs at least one value");
        }
    }

    HostValues(VulkanContext& vulkanContext, uint32_t count = 1)
        : HostValues(vulkanContext, std::vector<T>(count, T{})) {}

    void set(uint32_t index, const T& value) { values_.at(index) = value; }

    void set(const std::vector<T>& values) {
        if (values.size() != values_.size()) {
            throw std::invalid_argument("HostValues: wrong number of values");
        }
        values_ = values;
    }

    const std::vector<T>& values() const { return values_; }

    void _update(uint32_t pathId) override { this->getBuffer(pathId).memcopyFrom(values_); }

    bool isUpdatable() const override { return true; }

    const char* getType() const override { return "HostValues"; }

private:
    std::vector<T> values_;
};

// Values are copied tightly packed; shaders read vec3 arrays with the scalar
// block layout (GL_EXT_scalar_block_layout).
using HostFloat = HostValues<float>;
using HostInt = HostValues<int32_t>;
using HostUint = HostValues<uint32_t>;
using HostVec2 = HostValues<glm::vec2>;
using HostVec3 = HostValues<glm::vec3>;
using HostVec4 = HostValues<glm::vec4>;
using HostMat4 = HostValues<glm::mat4>;

} // namespace klartraum

#endif // KLARTRAUM_COMPUTEGRAPH_HOSTVALUES_HPP
