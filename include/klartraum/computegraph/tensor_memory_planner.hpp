// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#ifndef KLARTRAUM_COMPUTEGRAPH_TENSOR_MEMORY_PLANNER_HPP
#define KLARTRAUM_COMPUTEGRAPH_TENSOR_MEMORY_PLANNER_HPP

#include <algorithm>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <typeindex>
#include <vector>

namespace klartraum {

struct TensorLifetimeRequest {
    std::string name;
    size_t bytes;
    size_t firstUse;
    size_t lastUse;
    std::type_index elementType;
};

struct TensorMemoryAssignment {
    std::string name;
    size_t slot;
};

struct TensorMemoryPlan {
    std::vector<TensorMemoryAssignment> assignments;
    std::vector<size_t> slotCapacities;
    size_t logicalBytes = 0;
    size_t allocatedBytes = 0;
    size_t peakLiveBytes = 0;
};

/**
 * Assigns tensors with non-overlapping inclusive lifetimes to reusable storage
 * slots. Slots are kept type-compatible and grow to the largest tensor assigned
 * to them.
 */
class TensorMemoryPlanner {
public:
    static TensorMemoryPlan plan(std::vector<TensorLifetimeRequest> requests) {
        for (const auto& request : requests) {
            if (request.name.empty()) {
                throw std::invalid_argument("TensorMemoryPlanner: tensor name must not be empty");
            }
            if (request.lastUse < request.firstUse) {
                throw std::invalid_argument("TensorMemoryPlanner: invalid tensor lifetime");
            }
        }

        std::stable_sort(requests.begin(), requests.end(), [](const auto& lhs, const auto& rhs) {
            if (lhs.firstUse != rhs.firstUse)
                return lhs.firstUse < rhs.firstUse;
            return lhs.bytes > rhs.bytes;
        });

        struct SlotState {
            size_t capacity;
            size_t lastUse;
            std::type_index elementType;
        };

        TensorMemoryPlan result;
        std::vector<SlotState> slots;
        for (const auto& request : requests) {
            result.logicalBytes += request.bytes;

            size_t selected = std::numeric_limits<size_t>::max();
            size_t bestCapacity = std::numeric_limits<size_t>::max();
            for (size_t slot = 0; slot < slots.size(); ++slot) {
                const auto& state = slots[slot];
                if (state.elementType == request.elementType && state.lastUse < request.firstUse &&
                    state.capacity >= request.bytes && state.capacity < bestCapacity) {
                    selected = slot;
                    bestCapacity = state.capacity;
                }
            }

            if (selected == std::numeric_limits<size_t>::max()) {
                size_t largestReusableCapacity = 0;
                for (size_t slot = 0; slot < slots.size(); ++slot) {
                    const auto& state = slots[slot];
                    if (state.elementType == request.elementType && state.lastUse < request.firstUse &&
                        state.capacity > largestReusableCapacity) {
                        selected = slot;
                        largestReusableCapacity = state.capacity;
                    }
                }
            }

            if (selected == std::numeric_limits<size_t>::max()) {
                selected = slots.size();
                slots.push_back({request.bytes, request.lastUse, request.elementType});
            } else {
                slots[selected].capacity = std::max(slots[selected].capacity, request.bytes);
                slots[selected].lastUse = request.lastUse;
            }
            result.assignments.push_back({request.name, selected});
        }

        result.slotCapacities.reserve(slots.size());
        for (const auto& slot : slots) {
            result.slotCapacities.push_back(slot.capacity);
            result.allocatedBytes += slot.capacity;
        }

        for (const auto& point : requests) {
            size_t liveBytes = 0;
            for (const auto& request : requests) {
                if (request.firstUse <= point.firstUse && request.lastUse >= point.firstUse) {
                    liveBytes += request.bytes;
                }
            }
            result.peakLiveBytes = std::max(result.peakLiveBytes, liveBytes);
        }
        return result;
    }
};

} // namespace klartraum

#endif // KLARTRAUM_COMPUTEGRAPH_TENSOR_MEMORY_PLANNER_HPP
