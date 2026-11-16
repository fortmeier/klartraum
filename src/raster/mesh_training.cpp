// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#include "klartraum/raster/mesh_training.hpp"

#include <stdexcept>

#include <glm/gtc/type_ptr.hpp>

#include "klartraum/raster/differentiable_shading.hpp"
#include "klartraum/raster/visibility_pass.hpp"

namespace klartraum::raster {

namespace {

std::vector<float> packPositions(const TriangleMesh& mesh) {
    std::vector<float> values;
    for (const auto& p : mesh.positions)
        values.insert(values.end(), {p.x, p.y, p.z, 1.0f});
    return values;
}

std::vector<uint32_t> packTriangles(const TriangleMesh& mesh) {
    std::vector<uint32_t> values;
    for (const auto& t : mesh.triangles)
        values.insert(values.end(), {t.x, t.y, t.z});
    return values;
}

std::vector<float> packCornerUvs(const TriangleMesh& mesh) {
    std::vector<float> values;
    for (const auto& uv : mesh.cornerUvs)
        values.insert(values.end(), {uv.x, uv.y});
    return values;
}

std::vector<float> packCameras(const std::vector<glm::mat4>& cameras) {
    std::vector<float> values;
    for (const auto& m : cameras)
        values.insert(values.end(), glm::value_ptr(m), glm::value_ptr(m) + 16);
    return values;
}

} // namespace

MeshTraining::MeshTraining(VulkanContext& vulkanContext, const TriangleMesh& start,
                           const std::vector<glm::mat4>& cameras, const MeshTrainingOptions& options)
    : vulkanContext(vulkanContext),
      options(options),
      startMesh(start) {
    if (cameras.empty())
        throw std::invalid_argument("MeshTraining: at least one camera is needed");
    const uint32_t w = options.width, h = options.height, tw = options.textureWidth, th = options.textureHeight;
    const uint32_t pixels = w * h, texelCount = tw * th, vertexCount = uint32_t(start.positions.size()),
                   triangleCount = uint32_t(start.triangles.size()), views = uint32_t(cameras.size());
    const auto tensor = [&](uint32_t rows, uint32_t columns) {
        return vulkanContext.create<Tensor>(std::vector<uint32_t>{rows, columns});
    };
    const VertexAdjacency adjacency = vertexAdjacency(start);
    positions = tensor(vertexCount, 4);
    texels = tensor(texelCount, 4);
    for (auto& moments : positionMoments)
        moments = tensor(4 * vertexCount, 1);
    for (auto& moments : texelMoments)
        moments = tensor(4 * texelCount, 1);
    cornerUvs = tensor(3 * triangleCount, 2);
    cameraBuffer = tensor(views, 16);
    targets = tensor(views * pixels, 4);
    triangles = std::make_shared<UintBuffer>(vulkanContext, 3 * triangleCount);
    viewIndex = std::make_shared<UintBuffer>(vulkanContext, 1);
    adjacencyOffsets = std::make_shared<UintBuffer>(vulkanContext, adjacency.offsets.size());
    neighbours = std::make_shared<UintBuffer>(vulkanContext, adjacency.neighbours.size());
    stepCount = std::make_shared<UintBuffer>(vulkanContext, 1);
    colors = tensor(pixels, 4);
    antialiased = tensor(pixels, 4);
    imageGradients = tensor(pixels, 4);
    colorGradients = tensor(pixels, 4);
    vertexGradients = tensor(vertexCount, 4);
    textureGradients = tensor(texelCount, 4);
    lossValue = tensor(1, 1);

    auto nextView = cycleView(vulkanContext, views);
    nextView->setInput(viewIndex, 0);
    auto visibility = std::make_shared<UintBuffer>(vulkanContext, pixels);
    auto pass = std::make_shared<VisibilityPass>(vulkanContext, w, h, triangleCount);
    pass->setInput(positions, 0);
    pass->setInput(triangles, 1);
    pass->setInput(cameraBuffer, 2);
    pass->setInput(nextView, 3, 0);
    pass->setInput(visibility, 4);
    const auto connect = [&](const ComputeGraphElementPtr& element) {
        element->setInput(positions, 0);
        element->setInput(triangles, 1);
        element->setInput(cornerUvs, 2);
        element->setInput(texels, 3);
        element->setInput(cameraBuffer, 4);
        element->setInput(nextView, 5, 0);
        element->setInput(pass, 6, 4);
    };
    auto clear = clearBuffers(vulkanContext);
    clear->setInput(colorGradients, 0);
    clear->setInput(vertexGradients, 1);
    clear->setInput(textureGradients, 2);
    clear->setInput(lossValue, 3);
    auto shading = shade(vulkanContext, w, h, tw, th);
    connect(shading);
    shading->setInput(colors, 7);
    auto smoothing = antialias(vulkanContext, w, h, tw, th);
    connect(smoothing);
    smoothing->setInput(shading, 7, 7);
    smoothing->setInput(antialiased, 8);
    auto loss = imageLoss(vulkanContext, w, h, options.maskWeight);
    loss->setInput(smoothing, 0, 8);
    loss->setInput(targets, 1);
    loss->setInput(nextView, 2, 0);
    loss->setInput(imageGradients, 3);
    loss->setInput(clear, 4, 3);
    auto smoothingBackward = antialiasBackward(vulkanContext, w, h, tw, th);
    connect(smoothingBackward);
    smoothingBackward->setInput(shading, 7, 7);
    smoothingBackward->setInput(loss, 8, 3);
    smoothingBackward->setInput(clear, 9, 0);
    smoothingBackward->setInput(clear, 10, 1);
    auto shadingBackward = shadeColorBackward(vulkanContext, w, h, tw, th);
    connect(shadingBackward);
    shadingBackward->setInput(smoothingBackward, 7, 9);
    shadingBackward->setInput(smoothingBackward, 8, 10);
    shadingBackward->setInput(clear, 9, 2);
    ComputeGraphElementPtr geometryGradient = shadingBackward;
    int geometrySlot = 8;
    if (options.laplacianWeight > 0.0f) {
        auto regularizer = laplacianRegularizer(vulkanContext, vertexCount, options.laplacianWeight);
        regularizer->setInput(positions, 0);
        regularizer->setInput(adjacencyOffsets, 1);
        regularizer->setInput(neighbours, 2);
        regularizer->setInput(shadingBackward, 3, 8);
        regularizer->setInput(loss, 4, 4);
        geometryGradient = regularizer;
        geometrySlot = 3;
    }

    // Adam after the step counter and every read of the parameters (all upstream of the gradients).
    auto counter = training::stepCounter(vulkanContext);
    counter->setInput(stepCount, 0);
    auto geometryStep = training::adamStep(vulkanContext, 4 * vertexCount, options.geometry);
    geometryStep->setInput(positions, 0);
    geometryStep->setInput(geometryGradient, 1, geometrySlot);
    geometryStep->setInput(positionMoments[0], 2);
    geometryStep->setInput(positionMoments[1], 3);
    geometryStep->setInput(stepCount, 4);
    geometryStep->addDependency(counter);
    auto textureStep = training::adamStep(vulkanContext, 4 * texelCount, options.texture);
    textureStep->setInput(texels, 0);
    textureStep->setInput(shadingBackward, 1, 9);
    textureStep->setInput(texelMoments[0], 2);
    textureStep->setInput(texelMoments[1], 3);
    textureStep->setInput(stepCount, 4);
    textureStep->addDependency(geometryStep);
    trainingGraph = std::make_unique<ComputeGraph>(vulkanContext, 1);
    trainingGraph->compileFrom(textureStep);

    positions->getDataBuffer(0).memcopyFrom(packPositions(start));
    triangles->getBuffer(0).memcopyFrom(packTriangles(start));
    cornerUvs->getDataBuffer(0).memcopyFrom(packCornerUvs(start));
    cameraBuffer->getDataBuffer(0).memcopyFrom(packCameras(cameras));
    if (options.laplacianWeight > 0.0f) { // the adjacency is allocated only as an input of the regularizer
        adjacencyOffsets->getBuffer(0).memcopyFrom(adjacency.offsets);
        neighbours->getBuffer(0).memcopyFrom(adjacency.neighbours);
    }
    viewIndex->getBuffer(0).memcopyFrom(std::vector<uint32_t>{views - 1}); // the first step advances to view 0
    stepCount->getBuffer(0).memcopyFrom(std::vector<uint32_t>{0});
    for (auto* moments : {&positionMoments, &texelMoments})
        for (auto& moment : *moments)
            moment->getDataBuffer(0).memcopyFrom(std::vector<float>(moment->getDataElementCount(), 0.0f));
    targets->getDataBuffer(0).memcopyFrom(std::vector<float>(targets->getDataElementCount(), 0.0f));
    // A noise texture from a linear congruential generator.
    std::vector<float> noise(4 * size_t(texelCount), 1.0f);
    uint32_t state = options.seed * 747796405u + 2891336453u;
    for (size_t i = 0; i < noise.size(); ++i) {
        if (i % 4 == 3)
            continue;
        state = state * 1664525u + 1013904223u;
        noise[i] = float(state >> 8) / float(1u << 24);
    }
    texels->getDataBuffer(0).memcopyFrom(noise);
}

void MeshTraining::setTargets(const std::vector<float>& values) {
    if (values.size() != targets->getDataElementCount())
        throw std::invalid_argument("MeshTraining: wrong number of target values");
    targets->getDataBuffer(0).memcopyFrom(values);
}

void MeshTraining::step() { trainingGraph->submitAndWait(vulkanContext.getGraphicsQueue(), 0); }

float MeshTraining::loss() const {
    std::vector<float> value(1);
    lossValue->getDataBuffer(0).memcopyTo(value);
    return value[0];
}

TriangleMesh MeshTraining::mesh() const {
    std::vector<float> values(positions->getDataElementCount());
    positions->getDataBuffer(0).memcopyTo(values);
    TriangleMesh result = startMesh;
    for (size_t v = 0; v < result.positions.size(); ++v)
        result.positions[v] = glm::vec3(values[4 * v], values[4 * v + 1], values[4 * v + 2]);
    return result;
}

std::vector<float> MeshTraining::texture() const {
    std::vector<float> values(texels->getDataElementCount());
    texels->getDataBuffer(0).memcopyTo(values);
    return values;
}

void MeshTraining::setTexture(const std::vector<float>& values) {
    if (values.size() != texels->getDataElementCount())
        throw std::invalid_argument("MeshTraining: wrong number of texel values");
    texels->getDataBuffer(0).memcopyFrom(values);
}

std::vector<float> renderImages(VulkanContext& vulkanContext, const TriangleMesh& mesh,
                                const std::vector<float>& texture, uint32_t textureWidth, uint32_t textureHeight,
                                const std::vector<glm::mat4>& cameras, uint32_t width, uint32_t height) {
    if (texture.size() != 4 * size_t(textureWidth) * textureHeight)
        throw std::invalid_argument("renderImages: the texture needs four floats per texel");
    using Tensor = TensorElement<float>;
    using UintBuffer = BufferElement<VulkanBuffer<uint32_t>>;
    const uint32_t pixels = width * height, triangleCount = uint32_t(mesh.triangles.size());
    auto positions = vulkanContext.create<Tensor>(std::vector<uint32_t>{uint32_t(mesh.positions.size()), 4});
    auto cornerUvs = vulkanContext.create<Tensor>(std::vector<uint32_t>{3 * triangleCount, 2});
    auto texels = vulkanContext.create<Tensor>(std::vector<uint32_t>{textureWidth * textureHeight, 4});
    auto cameraBuffer = vulkanContext.create<Tensor>(std::vector<uint32_t>{uint32_t(cameras.size()), 16});
    auto colors = vulkanContext.create<Tensor>(std::vector<uint32_t>{pixels, 4});
    auto antialiased = vulkanContext.create<Tensor>(std::vector<uint32_t>{pixels, 4});
    auto triangles = std::make_shared<UintBuffer>(vulkanContext, 3 * triangleCount);
    auto viewIndex = std::make_shared<UintBuffer>(vulkanContext, 1);
    auto visibility = std::make_shared<UintBuffer>(vulkanContext, pixels);
    auto pass = std::make_shared<VisibilityPass>(vulkanContext, width, height, triangleCount);
    pass->setInput(positions, 0);
    pass->setInput(triangles, 1);
    pass->setInput(cameraBuffer, 2);
    pass->setInput(viewIndex, 3);
    pass->setInput(visibility, 4);
    const auto connect = [&](const ComputeGraphElementPtr& element) {
        element->setInput(positions, 0);
        element->setInput(triangles, 1);
        element->setInput(cornerUvs, 2);
        element->setInput(texels, 3);
        element->setInput(cameraBuffer, 4);
        element->setInput(viewIndex, 5);
        element->setInput(pass, 6, 4);
    };
    auto shading = shade(vulkanContext, width, height, textureWidth, textureHeight);
    connect(shading);
    shading->setInput(colors, 7);
    auto smoothing = antialias(vulkanContext, width, height, textureWidth, textureHeight);
    connect(smoothing);
    smoothing->setInput(shading, 7, 7);
    smoothing->setInput(antialiased, 8);
    ComputeGraph graph(vulkanContext, 1);
    graph.compileFrom(smoothing);
    positions->getDataBuffer(0).memcopyFrom(packPositions(mesh));
    triangles->getBuffer(0).memcopyFrom(packTriangles(mesh));
    cornerUvs->getDataBuffer(0).memcopyFrom(packCornerUvs(mesh));
    texels->getDataBuffer(0).memcopyFrom(texture);
    cameraBuffer->getDataBuffer(0).memcopyFrom(packCameras(cameras));
    std::vector<float> images;
    images.reserve(4 * size_t(pixels) * cameras.size());
    std::vector<float> image(4 * size_t(pixels));
    for (uint32_t view = 0; view < uint32_t(cameras.size()); ++view) {
        viewIndex->getBuffer(0).memcopyFrom(std::vector<uint32_t>{view});
        graph.submitAndWait(vulkanContext.getGraphicsQueue(), 0);
        antialiased->getDataBuffer(0).memcopyTo(image);
        images.insert(images.end(), image.begin(), image.end());
    }
    return images;
}

} // namespace klartraum::raster
