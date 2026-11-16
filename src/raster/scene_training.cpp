// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#include "klartraum/raster/scene_training.hpp"

#include <stdexcept>

#include <glm/gtc/type_ptr.hpp>

#include "klartraum/raster/differentiable_shading.hpp"
#include "klartraum/raster/visibility_pass.hpp"

namespace klartraum::raster {

namespace {

using Tensor = TensorElement<float>;
using UintBuffer = BufferElement<VulkanBuffer<uint32_t>>;

std::vector<float> read(const std::shared_ptr<Tensor>& tensor) {
    std::vector<float> values(tensor->getDataElementCount());
    tensor->getDataBuffer(0).memcopyTo(values);
    return values;
}

// The fixed mesh inputs and the trained texels, uploaded after compiling.
struct MeshBuffers {
    std::shared_ptr<Tensor> positions, cornerUvs, texels;
    std::shared_ptr<UintBuffer> triangles;

    MeshBuffers(VulkanContext& vulkanContext, const TriangleMesh& mesh, const SceneTrainingOptions& options) {
        const uint32_t triangleCount = uint32_t(mesh.triangles.size());
        positions = vulkanContext.create<Tensor>(std::vector<uint32_t>{uint32_t(mesh.positions.size()), 4});
        cornerUvs = vulkanContext.create<Tensor>(std::vector<uint32_t>{3 * triangleCount, 2});
        texels = vulkanContext.create<Tensor>(std::vector<uint32_t>{options.textureWidth * options.textureHeight, 4});
        triangles = std::make_shared<UintBuffer>(vulkanContext, 3 * triangleCount);
    }

    void upload(const TriangleMesh& mesh, const std::vector<float>& texture) const {
        std::vector<float> p, uv;
        std::vector<uint32_t> t;
        for (const auto& v : mesh.positions)
            p.insert(p.end(), {v.x, v.y, v.z, 1.0f});
        for (const auto& c : mesh.cornerUvs)
            uv.insert(uv.end(), {c.x, c.y});
        for (const auto& tri : mesh.triangles)
            t.insert(t.end(), {tri.x, tri.y, tri.z});
        positions->getDataBuffer(0).memcopyFrom(p);
        cornerUvs->getDataBuffer(0).memcopyFrom(uv);
        triangles->getBuffer(0).memcopyFrom(t);
        texels->getDataBuffer(0).memcopyFrom(texture);
    }
};

// The forward chain of one view: the mesh's visibility, surface depth, shading and antialiasing, the Gaussians
// blended until that depth and their composite over the mesh (slot 2 of composite).
struct SceneForward {
    ComputeGraphElementPtr pass, depth, shading, smoothing, composite;
    std::shared_ptr<Tensor> depths, colors, antialiased, combined;
    std::unique_ptr<splatting::SplatForward> splats;

    SceneForward(VulkanContext& vulkanContext, const TriangleMesh& mesh, const MeshBuffers& buffers,
                 const splatting::SplatParameters& parameters, const std::shared_ptr<Tensor>& cameras,
                 const ComputeGraphElementPtr& viewSource, int viewSlot, const SceneTrainingOptions& options) {
        const uint32_t w = options.raster.width, h = options.raster.height, tw = options.textureWidth,
                       th = options.textureHeight, pixels = w * h;
        const auto tensor = [&](uint32_t columns) {
            return vulkanContext.create<Tensor>(std::vector<uint32_t>{pixels, columns});
        };
        depths = tensor(1);
        colors = tensor(4);
        antialiased = tensor(4);
        combined = tensor(4);
        auto visibility = std::make_shared<VisibilityPass>(vulkanContext, w, h, uint32_t(mesh.triangles.size()));
        visibility->setInput(buffers.positions, 0);
        visibility->setInput(buffers.triangles, 1);
        visibility->setInput(cameras, 2);
        visibility->setInput(viewSource, 3, viewSlot);
        visibility->setInput(std::make_shared<UintBuffer>(vulkanContext, pixels), 4);
        pass = visibility;
        depth = surfaceDepth(vulkanContext, w, h, tw, th);
        connect(buffers, cameras, viewSource, viewSlot, depth);
        depth->setInput(depths, 7);
        shading = shade(vulkanContext, w, h, tw, th);
        connect(buffers, cameras, viewSource, viewSlot, shading);
        shading->setInput(colors, 7);
        smoothing = antialias(vulkanContext, w, h, tw, th);
        connect(buffers, cameras, viewSource, viewSlot, smoothing);
        smoothing->setInput(shading, 7, 7);
        smoothing->setInput(antialiased, 8);
        splats = std::make_unique<splatting::SplatForward>(vulkanContext, parameters, cameras, viewSource, viewSlot,
                                                           options.raster, depth, 7);
        composite = compositeOver(vulkanContext, w, h);
        composite->setInput(splats->render, 0, 3);
        composite->setInput(smoothing, 1, 8);
        composite->setInput(combined, 2);
    }

    // Connects the shared shading slots 0-6 of `element`.
    void connect(const MeshBuffers& buffers, const std::shared_ptr<Tensor>& cameras,
                 const ComputeGraphElementPtr& viewSource, int viewSlot, const ComputeGraphElementPtr& element) const {
        element->setInput(buffers.positions, 0);
        element->setInput(buffers.triangles, 1);
        element->setInput(buffers.cornerUvs, 2);
        element->setInput(buffers.texels, 3);
        element->setInput(cameras, 4);
        element->setInput(viewSource, 5, viewSlot);
        element->setInput(pass, 6, 4);
    }
};

void uploadSplats(const splatting::SplatParameters& p, const splatting::SplatSet& s) {
    p.means->getDataBuffer(0).memcopyFrom(s.means);
    p.logScales->getDataBuffer(0).memcopyFrom(s.logScales);
    p.rotations->getDataBuffer(0).memcopyFrom(s.rotations);
    p.colors->getDataBuffer(0).memcopyFrom(s.colors);
    p.opacities->getDataBuffer(0).memcopyFrom(s.opacities);
}

std::vector<float> packCameras(const std::vector<glm::mat4>& cameras) {
    std::vector<float> values;
    for (const auto& m : cameras)
        values.insert(values.end(), glm::value_ptr(m), glm::value_ptr(m) + 16);
    return values;
}

void checkSizes(const TriangleMesh& mesh, const std::vector<float>& texture, const splatting::SplatSet& splats,
                const std::vector<glm::mat4>& cameras, const SceneTrainingOptions& options) {
    if (cameras.empty() || mesh.triangles.empty() || splats.size() == 0)
        throw std::invalid_argument("SceneTraining: a camera, a triangle and a Gaussian are needed");
    if (texture.size() != size_t(4) * options.textureWidth * options.textureHeight)
        throw std::invalid_argument("SceneTraining: the texture does not have the size of the options");
}

} // namespace

std::vector<float> renderScene(VulkanContext& vulkanContext, const TriangleMesh& mesh,
                               const std::vector<float>& texture, const splatting::SplatSet& splats,
                               const std::vector<glm::mat4>& cameras, const SceneTrainingOptions& options) {
    checkSizes(mesh, texture, splats, cameras, options);
    const MeshBuffers buffers(vulkanContext, mesh, options);
    const splatting::SplatParameters parameters = splatting::createSplatParameters(vulkanContext, splats.size());
    auto cameraBuffer = vulkanContext.create<Tensor>(std::vector<uint32_t>{uint32_t(cameras.size()), 16});
    auto viewIndex = std::make_shared<UintBuffer>(vulkanContext, 1);
    SceneForward forward(vulkanContext, mesh, buffers, parameters, cameraBuffer, viewIndex, -1, options);
    ComputeGraph graph(vulkanContext, 1);
    graph.compileFrom(forward.composite);
    buffers.upload(mesh, texture);
    uploadSplats(parameters, splats);
    cameraBuffer->getDataBuffer(0).memcopyFrom(packCameras(cameras));
    std::vector<float> images;
    for (uint32_t view = 0; view < uint32_t(cameras.size()); ++view) {
        viewIndex->getBuffer(0).memcopyFrom(std::vector<uint32_t>{view});
        graph.submitAndWait(vulkanContext.getGraphicsQueue(), 0);
        const std::vector<float> image = read(forward.combined);
        images.insert(images.end(), image.begin(), image.end());
    }
    return images;
}

SceneTraining::SceneTraining(VulkanContext& vulkanContext, const TriangleMesh& mesh, const std::vector<float>& texture,
                             const splatting::SplatSet& splats, const std::vector<glm::mat4>& cameras,
                             const std::vector<float>& targetValues, const SceneTrainingOptions& options)
    : vulkanContext(vulkanContext) {
    checkSizes(mesh, texture, splats, cameras, options);
    const uint32_t w = options.raster.width, h = options.raster.height, tw = options.textureWidth,
                   th = options.textureHeight, pixels = w * h, views = uint32_t(cameras.size());
    if (targetValues.size() != size_t(4) * pixels * views)
        throw std::invalid_argument("SceneTraining: the targets do not match the cameras and the image size");
    const auto tensor = [&](uint32_t rows, uint32_t columns) {
        return vulkanContext.create<Tensor>(std::vector<uint32_t>{rows, columns});
    };
    const MeshBuffers buffers(vulkanContext, mesh, options);
    texels = buffers.texels;
    parameters = splatting::createSplatParameters(vulkanContext, splats.size());
    auto cameraBuffer = tensor(views, 16);
    auto targets = tensor(views * pixels, 4);
    auto viewIndex = std::make_shared<UintBuffer>(vulkanContext, 1);
    auto stepBuffer = std::make_shared<UintBuffer>(vulkanContext, 1);
    lossValue = tensor(1, 1);
    auto imageGradients = tensor(pixels, 4), splatImageGradients = tensor(pixels, 4),
         meshImageGradients = tensor(pixels, 4), colorGradients = tensor(pixels, 4),
         vertexGradients = tensor(uint32_t(mesh.positions.size()), 4), texelGradients = tensor(tw * th, 4);

    auto nextView = cycleView(vulkanContext, views);
    nextView->setInput(viewIndex, 0);
    SceneForward scene(vulkanContext, mesh, buffers, parameters, cameraBuffer, nextView, 0, options);
    auto clear = clearBuffers(vulkanContext);
    clear->setInput(colorGradients, 0);
    clear->setInput(vertexGradients, 1);
    clear->setInput(texelGradients, 2);
    clear->setInput(lossValue, 3);
    auto loss = imageLoss(vulkanContext, w, h, options.maskWeight);
    loss->setInput(scene.composite, 0, 2);
    loss->setInput(targets, 1);
    loss->setInput(nextView, 2, 0);
    loss->setInput(imageGradients, 3);
    loss->setInput(clear, 4, 3);
    auto compositeBackward = compositeOverBackward(vulkanContext, w, h);
    compositeBackward->setInput(scene.splats->render, 0, 3);
    compositeBackward->setInput(scene.smoothing, 1, 8);
    compositeBackward->setInput(loss, 2, 3);
    compositeBackward->setInput(splatImageGradients, 3);
    compositeBackward->setInput(meshImageGradients, 4);
    backward = std::make_unique<splatting::SplatBackward>(vulkanContext, parameters, cameraBuffer, nextView, 0,
                                                          *scene.splats, compositeBackward, 3);
    auto smoothingBackward = antialiasBackward(vulkanContext, w, h, tw, th);
    scene.connect(buffers, cameraBuffer, nextView, 0, smoothingBackward);
    smoothingBackward->setInput(scene.shading, 7, 7);
    smoothingBackward->setInput(compositeBackward, 8, 4);
    smoothingBackward->setInput(clear, 9, 0);
    smoothingBackward->setInput(clear, 10, 1);
    auto shadingBackward = shadeColorBackward(vulkanContext, w, h, tw, th);
    scene.connect(buffers, cameraBuffer, nextView, 0, shadingBackward);
    shadingBackward->setInput(smoothingBackward, 7, 9);
    shadingBackward->setInput(smoothingBackward, 8, 10);
    shadingBackward->setInput(clear, 9, 2);

    // Adam per trained group, after the step counter and every gradient (all reads of the parameters are upstream).
    auto counter = training::stepCounter(vulkanContext);
    counter->setInput(stepBuffer, 0);
    counter->addDependency(shadingBackward);
    counter->addDependency(backward->projectBackward);
    ComputeGraphElementPtr last = counter;
    const auto optimize = [&](const std::shared_ptr<Tensor>& values, const ComputeGraphElementPtr& gradientSource,
                              int gradientSlot, const training::AdamParameters& adam) {
        if (adam.learningRate == 0.0f)
            return;
        const uint32_t elements = uint32_t(values->getDataElementCount());
        auto step = training::adamStep(vulkanContext, elements, adam);
        step->setInput(values, 0);
        step->setInput(gradientSource, 1, gradientSlot);
        step->setInput(tensor(elements, 1), 2);
        step->setInput(tensor(elements, 1), 3);
        step->setInput(stepBuffer, 4);
        step->addDependency(last);
        last = step;
    };
    optimize(texels, shadingBackward, 9, options.texture);
    optimize(parameters.means, backward->projectBackward, 8, options.means);
    optimize(parameters.logScales, backward->projectBackward, 9, options.logScales);
    optimize(parameters.rotations, backward->projectBackward, 10, options.rotations);
    optimize(parameters.colors, backward->projectBackward, 11, options.colors);
    optimize(parameters.opacities, backward->projectBackward, 12, options.opacities);
    forward = std::move(scene.splats);
    graph = std::make_unique<ComputeGraph>(vulkanContext, 1);
    graph->compileFrom(last);

    buffers.upload(mesh, texture);
    uploadSplats(parameters, splats);
    cameraBuffer->getDataBuffer(0).memcopyFrom(packCameras(cameras));
    targets->getDataBuffer(0).memcopyFrom(targetValues);
    // The first step advances to view 0.
    viewIndex->getBuffer(0).memcopyFrom(std::vector<uint32_t>{views - 1});
    stepBuffer->getBuffer(0).memcopyFrom(std::vector<uint32_t>{0});
    backward->statistics->getDataBuffer(0).memcopyFrom(std::vector<float>(2 * splats.size(), 0.0f));
}

SceneTraining::~SceneTraining() = default;

void SceneTraining::step() { graph->submitAndWait(vulkanContext.getGraphicsQueue(), 0); }

float SceneTraining::loss() const { return read(lossValue)[0]; }

std::vector<float> SceneTraining::texture() const { return read(texels); }

splatting::SplatSet SceneTraining::splats() const {
    return {read(parameters.means), read(parameters.logScales), read(parameters.rotations), read(parameters.colors),
            read(parameters.opacities)};
}

} // namespace klartraum::raster
