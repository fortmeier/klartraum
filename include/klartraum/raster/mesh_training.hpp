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
    bool viewColors = false;                    ///< learn a colour gain and offset per view (imageLossWithViewColors())
    float viewColorWeight = 0.01f;              ///< pull of the per-view corrections towards identity
    training::AdamParameters viewColor{0.002f}; ///< optimizer of the per-view corrections
    uint32_t seed = 1;                          ///< seed of the noise the texture starts from
    uint32_t stepsPerSubmit = 1;                ///< training steps recorded one after another into one submission
    /// Whether the training compiles a graph of its own (step() submits it); otherwise its elements join the
    /// caller's graph (compile from getOutput(), then call initialize()).
    bool ownGraph = true;
};

/**
 * @brief Fits the vertex positions and the texture of a triangle mesh to masked images seen by known cameras.
 *
 * One training step is one pre-recorded command buffer and one submit: advance the view index (on the GPU, cycling
 * through all cameras), the visibility pass, shading, antialiasing, the image loss (optionally with a colour correction
 * per view), the backward elements down to vertex and texel gradients, the Laplacian regularizer and an Adam step for
 * positions and texels. The topology and the corner texture coordinates of the start mesh stay fixed.
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

    /** @brief Runs one submission of stepsPerSubmit training steps, each on the next view (with ownGraph). */
    void step();

    /** @brief The last element of a submission's steps: compile the caller's graph from it (without ownGraph). */
    ComputeGraphElementPtr getOutput() const { return lastElement; }

    /** @brief Makes the steps wait for @p element, e.g. the one writing the targets (getTargets()). */
    void addDependency(const ComputeGraphElementPtr& element);

    /** @brief Uploads the start mesh, cameras and state; after compiling the caller's graph (without ownGraph). */
    void initialize();

    /** @brief The target images (four floats per pixel and view, see setTargets()), e.g. for an element writing
     * them. */
    std::shared_ptr<TensorElement<float>> getTargets() const { return targets; }
    /// The trained vertex positions (four floats per vertex) and texels (four floats each), e.g. for drawing them.
    std::shared_ptr<TensorElement<float>> getPositions() const { return positions; }
    std::shared_ptr<TensorElement<float>> getTexels() const { return texels; }
    /// The fixed triangles (three uint32 each) and corner texture coordinates (two floats per corner).
    std::shared_ptr<BufferElement<VulkanBuffer<uint32_t>>> getTriangles() const { return triangles; }
    std::shared_ptr<TensorElement<float>> getCornerUvs() const { return cornerUvs; }

    /** @brief The loss of the last step: image loss plus regularizer. */
    float loss() const;

    /** @brief The start mesh with the current vertex positions. */
    TriangleMesh mesh() const;

    /** @brief The current texture, four floats per texel (rgb used), row-major. */
    std::vector<float> texture() const;

    /** @brief The per-view colour corrections: eight floats per view (gain rgb, 0, offset rgb, 0); empty without. */
    std::vector<float> viewColors() const;

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
    // The cameras of the target images.
    std::vector<glm::mat4> cameras;
    // The first elements of a submission (view advance, clearing) and the last (an Adam step).
    std::vector<ComputeGraphElementPtr> firstElements;
    ComputeGraphElementPtr lastElement;
    // The trained parameters and their Adam moments.
    std::shared_ptr<Tensor> positions, texels, positionMoments[2], texelMoments[2];
    // The per-view colour corrections (with MeshTrainingOptions::viewColors), their gradients and Adam moments.
    std::shared_ptr<Tensor> viewColorValues, viewColorGradients, viewColorMoments[2];
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
