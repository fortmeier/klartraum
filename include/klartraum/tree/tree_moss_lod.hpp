// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#ifndef KLARTRAUM_TREE_TREE_MOSS_LOD_HPP
#define KLARTRAUM_TREE_TREE_MOSS_LOD_HPP

#include <memory>
#include <vector>

#include "klartraum/computegraph/generalcomputation.hpp"
#include "klartraum/computegraph/hostvalues.hpp"
#include "klartraum/gaussian_lod.hpp"
#include "klartraum/tree/bonsai_scene.hpp"
#include "klartraum/tree/tree_gaussians.hpp"

namespace klartraum {

/// Push constants of `tree_lod_select.comp`.
struct TreeLodSelectPush {
    uint32_t instanceCount = 0;
    uint32_t capacity = 0;
    uint32_t pad0 = 0;
    uint32_t pad1 = 0;
    glm::vec4 boundaries{160.0f, 48.0f, 12.0f, 0.15f};
};

/**
 * @brief Draws many placed copies of captured patches with a level of detail per copy, every frame on the GPU.
 *
 * After klartraum-spacegame's Gaussian level of detail: every patch has four
 * levels (buildGaussianLods()), and every placed copy (TreeLodInstance) shows
 * the level its diameter in pixels calls for, crossfading between levels
 * within a band around the boundaries instead of switching. The copies'
 * Gaussians go into the extra slots of a TreeGaussians (TreeGaussianSources::
 * extraSlots), a pool shared by all copies: `tree_lod_clear.comp` empties it,
 * `tree_lod_select.comp` culls the copies outside the view and reserves room
 * for the others, and `tree_lod_write.comp` writes them, turned, scaled, on
 * their ground planes and coloured like captured Gaussians. Copies that do not
 * fit any more are left out (readCounter()[1]).
 */
class TreeMossLod {
public:
    TreeMossLod(VulkanContext& vulkanContext, TreeGaussians& target, const std::vector<MossPatch>& patches,
                std::vector<TreeLodInstance> instances, std::shared_ptr<CameraUboType> camera,
                std::shared_ptr<HostValues<TreeRenderSettings>> settings, BufferRef light = {},
                glm::vec4 boundaries = glm::vec4(160.0f, 48.0f, 12.0f, 0.15f));

    /// The target's Gaussians, as the write pass leaves them; give these to the splatting backend.
    const GaussianSoABuffers& getGaussians() const { return gaussians; }
    ComputeGraphElementPtr getElement() const { return write; }

    /// After a submission of `pathId`: [0] Gaussians written, [1] copies left out for lack of room.
    std::vector<uint32_t> readCounter(uint32_t pathId = 0);
    /// The Gaussians at full detail of all copies: what the pool would need without level of detail.
    uint32_t getFullCount() const { return fullCount; }

private:
    std::shared_ptr<GeneralComputation<glm::uvec2>> clear;
    std::shared_ptr<GeneralComputation<TreeLodSelectPush>> select;
    std::shared_ptr<GeneralComputation<uint32_t>> write;
    std::shared_ptr<BufferElement<VulkanBuffer<uint32_t>>> counter;
    GaussianSoABuffers gaussians;
    uint32_t fullCount = 0;
};

} // namespace klartraum

#endif // KLARTRAUM_TREE_TREE_MOSS_LOD_HPP
