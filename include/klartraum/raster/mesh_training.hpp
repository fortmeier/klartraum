// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#ifndef KLARTRAUM_RASTER_MESH_TRAINING_HPP
#define KLARTRAUM_RASTER_MESH_TRAINING_HPP

#include <cstdint>
#include <memory>
#include <vector>

#include <glm/glm.hpp>

#include "klartraum/computegraph/bufferelement.hpp"
#include "klartraum/computegraph/computegraph.hpp"
#include "klartraum/computegraph/tensorelement.hpp"
#include "klartraum/layers/training.hpp"
#include "klartraum/raster/mesh.hpp"
#include "klartraum/vulkan_context.hpp"

namespace klartraum::raster {

/** @brief Image and texture sizes, loss weights and optimizers of a MeshTraining. */
struct MeshTrainingOptions {
    uint32_t width = 128, height = 128;             ///< size of the target images in pixels
    uint32_t textureWidth = 64, textureHeight = 32; ///< size of the trained texture in texels
    float maskWeight = 1.0f;                        ///< weight of the coverage term of imageLoss()
    float laplacianWeight = 100.0f;                 ///< weight of laplacianRegularizer(), 0 for none
    training::AdamParameters geometry{0.002f};      ///< optimizer of the vertex positions
    training::AdamParameters texture{0.01f};        ///< optimizer of the texels
    uint32_t seed = 1;                              ///< seed of the noise the texture starts from
};

/**
 * @brief Fits the vertex positions and the texture of a triangle mesh to masked images seen by known cameras.
 *
 * One training step is one pre-recorded command buffer and one submit: advance the view index (on the GPU, cycling
 * through all cameras), the visibility pass, shading, antialiasing, the image loss, the backward elements down to
 * vertex and texel gradients, the Laplacian regularizer and an Adam step for positions and texels. The topology and
 * the corner texture coordinates of the start mesh stay fixed.
 */
class MeshTraining {
public:
    /** @brief Builds the training graph for @p start seen by @p cameras (clip from world, one per target image). */
    MeshTraining(VulkanContext& vulkanContext, const TriangleMesh& start, const std::vector<glm::mat4>& cameras,
                 const MeshTrainingOptions& options);

    /**
     * @brief Sets the target images: four floats per pixel and view, views in camera order, rgb premultiplied by
     * the mask and the mask (1 on the object, 0 off it, fractional at its outline).
     */
    void setTargets(const std::vector<float>& targets);

    /** @brief Runs one training step on the next view. */
    void step();

    /** @brief The loss of the last step: image loss plus regularizer. */
    float loss() const;

    /** @brief The start mesh with the current vertex positions. */
    TriangleMesh mesh() const;

    /** @brief The current texture, four floats per texel (rgb used), row-major. */
    std::vector<float> texture() const;

    /** @brief Replaces the texture (four floats per texel). */
    void setTexture(const std::vector<float>& texels);

private:
    using Tensor = TensorElement<float>;
    using UintBuffer = BufferElement<VulkanBuffer<uint32_t>>;

    // The context the graph runs on; it outlives this object.
    VulkanContext& vulkanContext;
    MeshTrainingOptions options;
    // The start mesh: topology and corner texture coordinates of the trained one.
    TriangleMesh startMesh;
    // The trained parameters and their Adam moments.
    std::shared_ptr<Tensor> positions, texels, positionMoments[2], texelMoments[2];
    // The fixed scene inputs: corner texture coordinates, cameras, targets and the adjacency of the regularizer.
    std::shared_ptr<Tensor> cornerUvs, cameraBuffer, targets;
    std::shared_ptr<UintBuffer> triangles, viewIndex, adjacencyOffsets, neighbours, stepCount;
    // The intermediate images and the gradients, cleared at the start of every step.
    std::shared_ptr<Tensor> colors, antialiased, imageGradients, colorGradients, vertexGradients, textureGradients,
        lossValue;
    std::unique_ptr<ComputeGraph> trainingGraph;
};

/**
 * @brief Renders @p mesh with @p texture from every camera with the shading and antialiasing of the training:
 * four floats per pixel and view (premultiplied rgb, coverage), views one after another.
 */
std::vector<float> renderImages(VulkanContext& vulkanContext, const TriangleMesh& mesh,
                                const std::vector<float>& texture, uint32_t textureWidth, uint32_t textureHeight,
                                const std::vector<glm::mat4>& cameras, uint32_t width, uint32_t height);

} // namespace klartraum::raster

#endif // KLARTRAUM_RASTER_MESH_TRAINING_HPP
