// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#ifndef KLARTRAUM_RENDERING_SKELETONGAUSSIANS_HPP
#define KLARTRAUM_RENDERING_SKELETONGAUSSIANS_HPP

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include <glm/glm.hpp>

#include "klartraum/computegraph/generalcomputation.hpp"
#include "klartraum/rendering/vulkan_gaussian_splatting_types.hpp"

namespace klartraum {

// What a SkeletonGaussians element reads every run (HostValues<SkeletonGaussianParams>).
struct SkeletonGaussianParams {
    glm::mat4 toScene{1.0f};                  // joints (character space) to scene space
    glm::vec4 colourOpacity{1.0f};            // rgb: linear colour, a: opacity (0 hides the skeleton)
    glm::vec4 shape{0.01f, 1.6f, 0.0f, 0.0f}; // x: bone radius (scene units), y: joint radius relative to it
};

struct SkeletonGaussiansPushConstants {
    uint32_t boneCount;
    uint32_t samplesPerBone;
    uint32_t jointCount; // joints drawn as spheres
    uint32_t total;
};

/**
 * @brief Draws a skeleton as Gaussians every time the graph runs.
 *
 * Reads joint positions (vec4, e.g. CharacterController::getJoints()) and
 * writes, per bone, `samplesPerBone` thin Gaussians spaced along it, their long
 * axis along the bone, and one round Gaussian per listed joint
 * (shaders/gaussians/skeleton_gaussians.comp). Merged with a scene
 * (GaussianMerge), the splatting sorts the skeleton with the scene, so the
 * scene hides it where it is in front. The output is per-path.
 */
class SkeletonGaussians : public GeneralComputation<SkeletonGaussiansPushConstants> {
public:
    SkeletonGaussians(VulkanContext& vulkanContext)
        : GeneralComputation<SkeletonGaussiansPushConstants>(vulkanContext,
                                                             "shaders/gaussians/skeleton_gaussians.comp.spv") {}

    const char* getType() const override { return "SkeletonGaussians"; }
};

struct SkeletonGaussiansResult {
    std::shared_ptr<SkeletonGaussians> element;
    GaussianSoABuffers output; // inputs 3 .. 9 of the element
};

/// The bones of CharacterController's joints for a plain skeleton (no face): pairs of joint indices.
const std::vector<std::pair<uint32_t, uint32_t>>& characterSkeletonBones();

/**
 * @brief Creates a SkeletonGaussians drawing @p bones of the joints in @p joints, with the
 * SkeletonGaussianParams buffer @p params. The joints the bones use get a sphere each.
 */
SkeletonGaussiansResult createSkeletonGaussians(VulkanContext& vulkanContext, const BufferRef& joints,
                                                const std::vector<std::pair<uint32_t, uint32_t>>& bones,
                                                const BufferRef& params, uint32_t samplesPerBone = 8);

} // namespace klartraum

#endif // KLARTRAUM_RENDERING_SKELETONGAUSSIANS_HPP
