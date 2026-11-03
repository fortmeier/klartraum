// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#include "klartraum/raster/mesh_morph.hpp"

#include <stdexcept>

#include "klartraum/computegraph/generalcomputation.hpp"
#include "klartraum/computegraph/tensorelement.hpp"
#include "klartraum/raster/differentiable_shading.hpp"
#include "klartraum/raster/image_target.hpp"
#include "klartraum/raster/visibility_pass.hpp"
#include "klartraum/rendering/gaussian_splatting_factory.hpp"

namespace klartraum::raster {

namespace {

struct CompositePushConstants {
    uint32_t width, height;
};

// Moves its image (input 0) from GENERAL back to `layout`, where the frame's renderer left it.
class ToLayout : public ComputeGraphElement {
public:
    explicit ToLayout(VkImageLayout layout)
        : layout(layout) {}
    const char* getType() const override { return "ToLayout"; }
    void checkInput(ComputeGraphElementPtr input, int index) override {}
    void _record(VkCommandBuffer commandBuffer, uint32_t pathId) override {
        auto image = std::dynamic_pointer_cast<ImageViewSrc>(getInputElement(0));
        VkImageMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        barrier.newLayout = layout;
        barrier.srcQueueFamilyIndex = barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = image->getImage(pathId);
        barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
        vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0,
                             0, nullptr, 0, nullptr, 1, &barrier);
    }

private:
    // The layout the frame is returned to.
    VkImageLayout layout;
};

} // namespace

MeshMorph::MeshMorph(VulkanContext& vulkanContext, const TriangleMesh& start, const GaussianSoABuffers& targets,
                     const std::shared_ptr<ImageViewSrc>& frame, VkImageLayout frameLayout,
                     const ComputeGraphElementPtr& frameReady, const MeshMorphOptions& options)
    : options(options) {
    if (options.views.empty())
        throw std::invalid_argument("MeshMorph: at least one target view is needed");
    MeshTrainingOptions trainingOptions = options.training;
    trainingOptions.ownGraph = false;
    std::vector<glm::mat4> clipFromWorld;
    for (const CameraMVP& view : options.views)
        clipFromWorld.push_back(view.proj * view.view * view.model);
    training = std::make_unique<MeshTraining>(vulkanContext, start, clipFromWorld, trainingOptions);

    // The target pictures: the Gaussians splatted over transparent black, turned into the training's targets.
    const VkExtent2D pictureExtent{trainingOptions.width, trainingOptions.height};
    for (uint32_t v = 0; v < uint32_t(options.views.size()); ++v) {
        auto picture = std::make_shared<OffscreenTarget>(vulkanContext, pictureExtent, 1);
        picture->setClearColor({{0.0f, 0.0f, 0.0f, 0.0f}});
        auto camera = std::make_shared<CameraUboType>();
        camera->ubo = options.views[v];
        auto splatting = createGaussianSplatting(vulkanContext, GsplatBackend::Raster, picture, camera, targets);
        training->addDependency(
            imageTarget(vulkanContext, picture, splatting->getOutputLayout(), splatting, training->getTargets(), v));
        pictures.push_back(picture);
        pictureCameras.push_back(camera);
    }

    // The current mesh over the frame: visibility, shading and antialiasing after the training's steps.
    const uint32_t width = options.frame.width, height = options.frame.height, pixels = width * height;
    const uint32_t tw = trainingOptions.textureWidth, th = trainingOptions.textureHeight;
    drawCamera = std::make_shared<HostValues<glm::mat4>>(vulkanContext, std::vector<glm::mat4>{glm::mat4(1.0f)});
    drawView = std::make_shared<BufferElement<VulkanBuffer<uint32_t>>>(vulkanContext, 1);
    auto visibility = std::make_shared<BufferElement<VulkanBuffer<uint32_t>>>(vulkanContext, pixels);
    auto pass = std::make_shared<VisibilityPass>(vulkanContext, width, height, uint32_t(start.triangles.size()));
    pass->setInput(training->getPositions(), 0);
    pass->setInput(training->getTriangles(), 1);
    pass->setInput(drawCamera, 2);
    pass->setInput(drawView, 3);
    pass->setInput(visibility, 4);
    pass->addDependency(training->getOutput());
    const auto connect = [&](const ComputeGraphElementPtr& element) {
        element->setInput(training->getPositions(), 0);
        element->setInput(training->getTriangles(), 1);
        element->setInput(training->getCornerUvs(), 2);
        element->setInput(training->getTexels(), 3);
        element->setInput(drawCamera, 4);
        element->setInput(drawView, 5);
        element->setInput(pass, 6, 4);
    };
    auto shading = shade(vulkanContext, width, height, tw, th);
    connect(shading);
    shading->setInput(vulkanContext.create<Tensor>(std::vector<uint32_t>{pixels, 4}), 7);
    auto smoothing = antialias(vulkanContext, width, height, tw, th);
    connect(smoothing);
    smoothing->setInput(shading, 7, 7);
    smoothing->setInput(vulkanContext.create<Tensor>(std::vector<uint32_t>{pixels, 4}), 8);

    drawModel = std::make_shared<HostValues<glm::mat4>>(vulkanContext, std::vector<glm::mat4>{glm::mat4(1.0f)});
    const bool lit = options.lightSettings.element && options.light.element;
    auto composite = vulkanContext.create<GeneralComputation<CompositePushConstants>>(
        lit ? "shaders/tree/tree_captured_layer.comp.spv" : "shaders/raster/composite_layer.comp.spv");
    composite->setName("MeshMorphComposite");
    composite->setPushConstants({{width, height}});
    composite->setGroupCount((width + 15) / 16, (height + 15) / 16, 1);
    composite->setInput(frame, 0);
    composite->setInput(smoothing, 1, 8);
    if (lit) {
        composite->setInput(pass, 2, 4);
        composite->setInput(training->getPositions(), 3);
        composite->setInput(training->getTriangles(), 4);
        composite->setInput(drawModel, 5);
        options.lightSettings.connectTo(*composite, 6);
        options.light.connectTo(*composite, 7);
    }
    composite->setImageLayoutTransition(0, frameLayout, VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_MEMORY_WRITE_BIT,
                                        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
    if (frameReady)
        composite->addDependency(frameReady);
    output = composite;
    if (frameLayout != VK_IMAGE_LAYOUT_GENERAL) {
        auto back = std::make_shared<ToLayout>(frameLayout);
        back->setInput(frame, 0);
        back->addDependency(composite);
        output = back;
    }
}

void MeshMorph::initialize() {
    training->initialize();
    for (uint32_t v = 0; v < uint32_t(pictureCameras.size()); ++v)
        pictureCameras[v]->update(0);
    drawView->getBuffer(0).memcopyFrom(std::vector<uint32_t>{0});
    updateDrawCamera();
}

void MeshMorph::setCamera(const glm::mat4& clipFromWorld) {
    camera = clipFromWorld;
    updateDrawCamera();
}

void MeshMorph::setModel(const glm::mat4& worldFromMorph) {
    model = worldFromMorph;
    updateDrawCamera();
}

void MeshMorph::updateDrawCamera() {
    drawCamera->set(0, camera * model);
    drawModel->set(0, model);
}

} // namespace klartraum::raster
