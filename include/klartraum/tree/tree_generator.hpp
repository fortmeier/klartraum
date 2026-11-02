// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#ifndef KLARTRAUM_TREE_TREE_GENERATOR_HPP
#define KLARTRAUM_TREE_TREE_GENERATOR_HPP

#include <array>
#include <memory>
#include <vector>

#include "klartraum/computegraph/bufferelement.hpp"
#include "klartraum/computegraph/computegraphgroup.hpp"
#include "klartraum/computegraph/generalcomputation.hpp"
#include "klartraum/computegraph/hostvalues.hpp"
#include "klartraum/tree/tree_model.hpp"
#include "klartraum/rendering/vulkan_gaussian_splatting_types.hpp"

namespace klartraum {

/// Push constants of the tree generation passes (`TreeGenPush` in tree_generate_common.glsl).
struct TreeGenPush {
    uint32_t level = 0;
    uint32_t instanceCount = 0;
    uint32_t segmentCapacity = 0;
    uint32_t leafCapacity = 0;
    glm::uvec4 stemOffset{0};
    glm::uvec4 stemCapacity{0};
};

/// Number of records the generation can store; records beyond are dropped and flagged.
struct TreeCapacity {
    std::array<uint32_t, TREE_MAX_LEVELS> stems{};
    uint32_t segments = 0;
    uint32_t leaves = 0;

    /// The record counts of the CPU reference for these trees, times `margin`.
    static TreeCapacity fromReference(const std::vector<TreeTypeParams>& types,
                                      const std::vector<TreeInstance>& instances, const TreeFrame& frame,
                                      float margin = 1.25f);
};

/// Indices into the counter buffer of TreeGenerator.
enum TreeCounter : uint32_t {
    kTreeCounterSegments = 4,
    kTreeCounterLeaves = 5,
    kTreeCounterOverflow = 6,
    kTreeCounterCount = 16,
};

/**
 * @brief Generates trees on the GPU every frame: the work graph of Kuth et al. as a fixed chain of compute passes.
 *
 * The passes are `tree_roots.comp` (one trunk record per instance) and, for
 * each stem level, `tree_args.comp` (dispatch arguments from the level's
 * queue length) followed by `tree_level.comp` (one 32-thread workgroup per
 * stem, dispatched indirectly). They append segment and leaf records, which
 * renderers read with the counts in getCounters(). Tree type parameters,
 * instances and the frame values (up, wind, time, season) are HostValues, so
 * the CPU can change them between frames without re-recording.
 *
 * All passes share the bindings of `tree_generate_common.glsl`:
 * 0 params, 1 instances, 2 frame, 3 stems, 4 segments, 5 leaves, 6 counters,
 * 7 dispatch arguments.
 */
class TreeGenerator : public ComputeGraphGroup {
public:
    TreeGenerator(VulkanContext& vulkanContext, std::vector<TreeTypeParams> types, std::vector<TreeInstance> instances,
                  const TreeCapacity& capacity, const TreeFrame& frame = defaultTreeFrame());

    const char* getType() const override { return "TreeGenerator"; }
    void checkInput(ComputeGraphElementPtr input, int index = 0) override {}

    HostValues<TreeTypeParams>& getParams() { return *params; }
    HostValues<TreeInstance>& getInstances() { return *instances; }
    HostValues<TreeFrame>& getFrame() { return *frame; }

    /// The buffers as the last pass leaves them; consumers connected to them run after the generation.
    BufferRef getSegments() const { return {lastPass, 4}; }
    BufferRef getLeaves() const { return {lastPass, 5}; }
    BufferRef getCounters() const { return {lastPass, 6}; }
    BufferRef getStems() const { return {lastPass, 3}; }
    BufferRef getParamsRef() const { return {lastPass, 0}; }
    BufferRef getInstancesRef() const { return {lastPass, 1}; }
    BufferRef getFrameRef() const { return {lastPass, 2}; }

    const TreeCapacity& getCapacity() const { return capacity; }
    const TreeGenPush& getPush() const { return push; }

    /// Reads back the records of path `pathId` after its submission finished.
    TreeRecords readRecords(uint32_t pathId = 0);
    /// Reads back the counter buffer of path `pathId`.
    std::vector<uint32_t> readCounters(uint32_t pathId = 0);

private:
    TreeCapacity capacity;
    TreeGenPush push;
    std::shared_ptr<HostValues<TreeTypeParams>> params;
    std::shared_ptr<HostValues<TreeInstance>> instances;
    std::shared_ptr<HostValues<TreeFrame>> frame;
    std::shared_ptr<BufferElement<VulkanBuffer<StemRecord>>> stems;
    std::shared_ptr<BufferElement<VulkanBuffer<SegmentRecord>>> segments;
    std::shared_ptr<BufferElement<VulkanBuffer<LeafRecord>>> leaves;
    std::shared_ptr<BufferElement<VulkanBuffer<uint32_t>>> counters;
    std::array<std::shared_ptr<DispatchIndirectCommandBufferElement>, TREE_MAX_LEVELS> args;
    std::vector<std::shared_ptr<GeneralComputation<TreeGenPush>>> passes;
    ComputeGraphElementPtr lastPass;
};

} // namespace klartraum

#endif // KLARTRAUM_TREE_TREE_GENERATOR_HPP
