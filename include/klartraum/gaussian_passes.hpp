#ifndef KLARTRAUM_GAUSSIAN_PASSES_HPP
#define KLARTRAUM_GAUSSIAN_PASSES_HPP

#include <array>
#include <cstdint>
#include <memory>

#include "klartraum/computegraph/generalcomputation.hpp"
#include "klartraum/vulkan_gaussian_splatting_types.hpp"

namespace klartraum {

class VulkanContext;

// Compute passes that move and combine Gaussians on the GPU, every time the
// graph runs. Their outputs are per-path buffers, so a frame in flight keeps
// its Gaussians while the next one is written. Wire them to a backend with
// createGaussianSplatting(..., const GaussianSoABuffers&, ...).

// Turns seven scalars into a transform buffer for transformGaussiansPass():
// translation x, y, z, rotation pitch, yaw, roll in degrees (about X, then Y,
// then Z) and a uniform scale, each read from the first float of its buffer,
// e.g. a HostValues the CPU animates. One invocation per run.
class TransformBufferPass : public GeneralComputation<void> {
public:
    // Floats in a transform buffer: translation, scale, rotation quaternion
    // and the rotation matrices of the three SH bands (91 used).
    static constexpr uint32_t kSize = 96;

    TransformBufferPass(VulkanContext& vulkanContext)
        : GeneralComputation<void>(vulkanContext, "shaders/gaussians/make_transform.comp.spv") {}
    const char* getType() const override { return "TransformBufferPass"; }
};

struct TransformBuffer {
    std::shared_ptr<TransformBufferPass> pass;
    BufferRef transform;  // slot 7 of the pass
};

// Parameters in the order x, y, z, pitch, yaw, roll, scale.
TransformBuffer makeTransformBuffer(VulkanContext& vulkanContext, const std::array<BufferRef, 7>& parameters);

struct GaussianPassPushConstants {
    uint32_t countA;
    uint32_t countB;  // merge only
};

class GaussianTransformPass : public GeneralComputation<GaussianPassPushConstants> {
public:
    GaussianTransformPass(VulkanContext& vulkanContext)
        : GeneralComputation<GaussianPassPushConstants>(vulkanContext, "shaders/gaussians/transform_gaussians.comp.spv") {}
    const char* getType() const override { return "GaussianTransformPass"; }
};

class GaussianMergePass : public GeneralComputation<GaussianPassPushConstants> {
public:
    GaussianMergePass(VulkanContext& vulkanContext)
        : GeneralComputation<GaussianPassPushConstants>(vulkanContext, "shaders/gaussians/merge_gaussians.comp.spv") {}
    const char* getType() const override { return "GaussianMergePass"; }
};

template <typename Pass> struct GaussianPass {
    std::shared_ptr<Pass> pass;
    GaussianSoABuffers output;  // buffers written by the pass
};

// Writes transformed copies of `source` (see makeTransformBuffer): positions,
// orientations and sizes move, SH coefficients rotate with the scene.
GaussianPass<GaussianTransformPass> transformGaussiansPass(VulkanContext& vulkanContext,
                                                          const GaussianSoABuffers& source,
                                                          const BufferRef& transform);

// Concatenates two sets of Gaussians, `a`'s first, so that one backend
// renders and depth-sorts them together.
GaussianPass<GaussianMergePass> mergeGaussiansPass(VulkanContext& vulkanContext, const GaussianSoABuffers& a,
                                                   const GaussianSoABuffers& b);

} // namespace klartraum

#endif // KLARTRAUM_GAUSSIAN_PASSES_HPP
