#ifndef KLARTRAUM_COMPUTEGRAPH_HOSTVALUES_HPP
#define KLARTRAUM_COMPUTEGRAPH_HOSTVALUES_HPP

#include <stdexcept>
#include <vector>

#include "klartraum/computegraph/bufferelement.hpp"
#include "klartraum/vulkan_buffer.hpp"

namespace klartraum {

/**
 * @brief A few floats the CPU sets and shaders read, e.g. animation values.
 *
 * set() only stores the values on the CPU. Right before a path of the graph
 * is submitted, _update() copies them into that path's storage buffer, so a
 * value set between frames is seen by the next frame and paths still in
 * flight keep theirs. Shaders read it as `float values[]`.
 */
class HostValues : public BufferElement<VulkanBuffer<float>> {
public:
    HostValues(VulkanContext& vulkanContext, std::vector<float> initial)
        : BufferElement<VulkanBuffer<float>>(vulkanContext, static_cast<uint32_t>(initial.size())),
          values_(std::move(initial)) {
        if (values_.empty()) {
            throw std::invalid_argument("HostValues: needs at least one value");
        }
    }

    HostValues(VulkanContext& vulkanContext, uint32_t count = 1)
        : HostValues(vulkanContext, std::vector<float>(count, 0.0f)) {}

    void set(uint32_t index, float value) { values_.at(index) = value; }

    void set(const std::vector<float>& values) {
        if (values.size() != values_.size()) {
            throw std::invalid_argument("HostValues: wrong number of values");
        }
        values_ = values;
    }

    const std::vector<float>& values() const { return values_; }

    void _update(uint32_t pathId) override { getBuffer(pathId).memcopyFrom(values_); }

    const char* getType() const override { return "HostValues"; }

private:
    std::vector<float> values_;
};

} // namespace klartraum

#endif // KLARTRAUM_COMPUTEGRAPH_HOSTVALUES_HPP
