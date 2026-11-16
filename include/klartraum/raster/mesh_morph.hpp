// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#ifndef KLARTRAUM_RASTER_MESH_MORPH_HPP
#define KLARTRAUM_RASTER_MESH_MORPH_HPP

#include <cstdint>
#include <memory>
#include <vector>

#include <glm/glm.hpp>

#include "klartraum/computegraph/bufferelement.hpp"
#include "klartraum/computegraph/computegraphelement.hpp"
#include "klartraum/computegraph/hostvalues.hpp"
#include "klartraum/computegraph/imageviewsrc.hpp"
#include "klartraum/draw_component.hpp"
#include "klartraum/offscreen_target.hpp"
#include "klartraum/raster/mesh.hpp"
#include "klartraum/raster/mesh_training.hpp"
#include "klartraum/rendering/vulkan_gaussian_splatting_types.hpp"
#include "klartraum/vulkan_context.hpp"

namespace klartraum::raster {

/** @brief The target views, sizes and optimization of a MeshMorph. */
struct MeshMorphOptions {
    std::vector<CameraMVP> views; ///< cameras of the target pictures (Vulkan projection, y down)
    MeshTrainingOptions training; ///< width/height: of the target pictures; stepsPerSubmit: steps per frame
    VkExtent2D frame{1280, 720};  ///< size of the frame the mesh is drawn into
};

/**
 * @brief Morphs a mesh into the shape and colours of Gaussians, live, within a frame's compute graph.
 *
 * Every run of the graph splats the target Gaussians into an offscreen target per view (cleared to transparent
 * black), turns the pictures into training targets (imageTarget()), runs MeshTrainingOptions::stepsPerSubmit steps
 * of differentiable mesh optimization towards them (MeshTraining) and draws the current mesh, placed by
 * setModel() and seen by setCamera(), over the frame (visibility pass, shading and antialiasing, composited). The
 * gradients thus flow from the rendered pictures back into the mesh's shape and texture in one submission.
 */
class MeshMorph {
public:
    /**
     * @brief Builds the morph of @p start into @p targets, drawn over @p frame after @p frameReady (the element that
     * leaves the frame in @p frameLayout, e.g. a renderer drawing the rest of the scene).
     */
    MeshMorph(VulkanContext& vulkanContext, const TriangleMesh& start, const GaussianSoABuffers& targets,
              const std::shared_ptr<ImageViewSrc>& frame, VkImageLayout frameLayout,
              const ComputeGraphElementPtr& frameReady, const MeshMorphOptions& options);

    /** @brief The last element of a frame: compile the graph from it, then call initialize(). */
    ComputeGraphElementPtr getOutput() const { return output; }

    /** @brief Uploads the start mesh and the cameras; after compiling the graph. */
    void initialize();

    /** @brief The camera the mesh is drawn with in the next frames (clip from world, Vulkan, y down). */
    void setCamera(const glm::mat4& clipFromWorld);
    /** @brief Where the mesh stands in the world in the next frames (world from the morph's frame). */
    void setModel(const glm::mat4& worldFromMorph);

    /** @brief The training, e.g. for its loss, mesh and texture. */
    MeshTraining& getTraining() { return *training; }

private:
    using Tensor = TensorElement<float>;

    // Recomputes the drawing camera from `camera` and `model`.
    void updateDrawCamera();

    MeshMorphOptions options;
    // The offscreen pictures of the target Gaussians and their cameras, one per view.
    std::vector<std::shared_ptr<OffscreenTarget>> pictures;
    std::vector<std::shared_ptr<CameraUboType>> pictureCameras;
    std::unique_ptr<MeshTraining> training;
    // The camera the mesh is drawn with (clip from morph frame) and the drawn view's index (always 0).
    std::shared_ptr<HostValues<glm::mat4>> drawCamera;
    std::shared_ptr<BufferElement<VulkanBuffer<uint32_t>>> drawView;
    glm::mat4 camera{1.0f}, model{1.0f};
    ComputeGraphElementPtr output;
};

} // namespace klartraum::raster

#endif // KLARTRAUM_RASTER_MESH_MORPH_HPP
