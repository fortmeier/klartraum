// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#ifndef KLARTRAUM_RASTER_SCENE_TRAINING_HPP
#define KLARTRAUM_RASTER_SCENE_TRAINING_HPP

#include <cstdint>
#include <memory>
#include <vector>

#include <glm/glm.hpp>

#include "klartraum/computegraph/bufferelement.hpp"
#include "klartraum/computegraph/computegraph.hpp"
#include "klartraum/computegraph/tensorelement.hpp"
#include "klartraum/layers/training.hpp"
#include "klartraum/raster/mesh.hpp"
#include "klartraum/splatting/splat_training.hpp"
#include "klartraum/vulkan_context.hpp"

namespace klartraum::raster {

/**
 * @brief Image and texture sizes and the optimizers of a SceneTraining; a learning rate of 0 keeps that parameter
 * group fixed.
 */
struct SceneTrainingOptions {
    splatting::SplatRasterOptions raster;           ///< image size and the limits of the Gaussians' forward chain
    uint32_t textureWidth = 64, textureHeight = 32; ///< size of the mesh's texture in texels
    float maskWeight = 1.0f;                        ///< weight of the coverage term of imageLoss()
    training::AdamParameters texture{0.01f};        ///< optimizer of the texels
    training::AdamParameters means{0.0f};           ///< optimizer of the Gaussians' positions
    training::AdamParameters logScales{0.0f};       ///< optimizer of the Gaussians' log scales
    training::AdamParameters rotations{0.0f};       ///< optimizer of the Gaussians' rotations
    training::AdamParameters colors{0.01f};         ///< optimizer of the Gaussians' colours
    training::AdamParameters opacities{0.05f};      ///< optimizer of the Gaussians' logit opacities
};

/**
 * @brief Renders a textured triangle mesh and Gaussians together from every camera: the Gaussians blended until the
 * mesh's surface depth and composited over the antialiased mesh, as SceneTraining draws them. Four floats per pixel
 * and view (premultiplied rgb, coverage), views one after another.
 */
std::vector<float> renderScene(VulkanContext& vulkanContext, const TriangleMesh& mesh,
                               const std::vector<float>& texture, const splatting::SplatSet& splats,
                               const std::vector<glm::mat4>& cameras, const SceneTrainingOptions& options);

/**
 * @brief Fits the appearance (and optionally the shape of the Gaussians) of a scene of a textured triangle mesh and
 * Gaussians to masked images seen by known cameras.
 *
 * One training step is one pre-recorded command buffer and one submit: advance the view index (on the GPU, cycling
 * through all cameras), the mesh's visibility, surface depth, shading and antialiasing, the Gaussians blended until
 * that depth, their composite over the mesh, imageLoss() against the view's target, the backward elements down to
 * texel and Gaussian gradients, and an Adam step per trained parameter group. The mesh's geometry stays fixed, and
 * which surface is nearest is held fixed within a step.
 */
class SceneTraining {
public:
    /**
     * @brief Builds the training of @p texture (four floats per texel) on @p mesh and of @p splats on @p targets (four
     * floats per pixel and view, views in camera order: rgb premultiplied by the mask, and the mask).
     */
    SceneTraining(VulkanContext& vulkanContext, const TriangleMesh& mesh, const std::vector<float>& texture,
                  const splatting::SplatSet& splats, const std::vector<glm::mat4>& cameras,
                  const std::vector<float>& targets, const SceneTrainingOptions& options);
    ~SceneTraining();

    /** @brief Runs one training step on the next view. */
    void step();

    /** @brief The loss of the last step. */
    float loss() const;

    /** @brief The current texture, four floats per texel. */
    std::vector<float> texture() const;

    /** @brief The current Gaussians. */
    splatting::SplatSet splats() const;

private:
    using Tensor = TensorElement<float>;
    using UintBuffer = BufferElement<VulkanBuffer<uint32_t>>;

    // The context the graph runs on; it outlives this object.
    VulkanContext& vulkanContext;
    // The trained texels and Gaussians.
    std::shared_ptr<Tensor> texels;
    splatting::SplatParameters parameters;
    // The loss of the last step (one float).
    std::shared_ptr<Tensor> lossValue;
    // The forward and backward chains of the Gaussians.
    std::unique_ptr<splatting::SplatForward> forward;
    std::unique_ptr<splatting::SplatBackward> backward;
    // The training graph; it owns no data that outlives the members above.
    std::unique_ptr<ComputeGraph> graph;
};

} // namespace klartraum::raster

#endif // KLARTRAUM_RASTER_SCENE_TRAINING_HPP
