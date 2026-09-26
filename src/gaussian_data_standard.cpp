#include <algorithm>
#include <cmath>
#include <iostream>

#include <glm/glm.hpp>

#include "load-spz.h"

#include "klartraum/gaussian_data_standard.hpp"

namespace klartraum {

static float sigmoidStandard(float x) { return 1.0f / (1.0f + std::exp(-x)); }

GaussianDataStandard::GaussianDataStandard(VulkanContext& vulkanContext, const std::string& path) {
    loadSPZModel(path);
    uploadSoA(vulkanContext);
}

GaussianDataStandard::GaussianDataStandard(VulkanContext& vulkanContext, std::vector<Gaussian3D> gaussians) {
    gaussians3DData = std::move(gaussians);
    buffers_.count  = static_cast<uint32_t>(gaussians3DData.size());
    uploadSoA(vulkanContext);
}

void GaussianDataStandard::uploadSoA(VulkanContext& vulkanContext) {
    const uint32_t N = buffers_.count;

    // Convert AoS -> SoA and upload to GPU (single-path, static).
    std::vector<glm::vec3> pos3d(N), scale3d(N);
    std::vector<glm::vec4> rot3d(N), colAlpha3d(N);
    std::vector<float>     shR(15*N), shG(15*N), shB(15*N);

    for (uint32_t i = 0; i < N; i++) {
        const auto& g = gaussians3DData[i];
        pos3d[i]      = {g.position[0], g.position[1], g.position[2]};
        rot3d[i]      = {g.rotation[0], g.rotation[1], g.rotation[2], g.rotation[3]};
        scale3d[i]    = {g.scale[0],    g.scale[1],    g.scale[2]};
        colAlpha3d[i] = {g.color[0],    g.color[1],    g.color[2],    g.alpha};
        for (int b = 0; b < 15; b++) {
            shR[b * N + i] = g.shR[b];
            shG[b * N + i] = g.shG[b];
            shB[b * N + i] = g.shB[b];
        }
    }

    buffers_.pos      = std::make_shared<BufferElementSinglePath<VulkanBuffer<glm::vec3>>>(vulkanContext, N);
    buffers_.rot      = std::make_shared<BufferElementSinglePath<VulkanBuffer<glm::vec4>>>(vulkanContext, N);
    buffers_.scale    = std::make_shared<BufferElementSinglePath<VulkanBuffer<glm::vec3>>>(vulkanContext, N);
    buffers_.colAlpha = std::make_shared<BufferElementSinglePath<VulkanBuffer<glm::vec4>>>(vulkanContext, N);
    buffers_.shR      = std::make_shared<BufferElementSinglePath<VulkanBuffer<float>>>(vulkanContext, 15*N);
    buffers_.shG      = std::make_shared<BufferElementSinglePath<VulkanBuffer<float>>>(vulkanContext, 15*N);
    buffers_.shB      = std::make_shared<BufferElementSinglePath<VulkanBuffer<float>>>(vulkanContext, 15*N);

    buffers_.pos->setName("Pos3D");
    buffers_.rot->setName("Rot3D");
    buffers_.scale->setName("Scale3D");
    buffers_.colAlpha->setName("ColAlpha3D");
    buffers_.shR->setName("ShR");
    buffers_.shG->setName("ShG");
    buffers_.shB->setName("ShB");

    buffers_.pos->getBuffer().memcopyFrom(pos3d);
    buffers_.rot->getBuffer().memcopyFrom(rot3d);
    buffers_.scale->getBuffer().memcopyFrom(scale3d);
    buffers_.colAlpha->getBuffer().memcopyFrom(colAlpha3d);
    buffers_.shR->getBuffer().memcopyFrom(shR);
    buffers_.shG->getBuffer().memcopyFrom(shG);
    buffers_.shB->getBuffer().memcopyFrom(shB);
}

void GaussianDataStandard::loadSPZModel(const std::string& path) {
    spz::PackedGaussians packed = spz::loadSpzPacked(path);
    gaussians3DData.clear();
    gaussians3DData.reserve(packed.numPoints);
    spz::CoordinateConverter conv;

    for (int i = 0; i < packed.numPoints; i++) {
        spz::UnpackedGaussian ug = packed.unpack(i, conv);
        Gaussian3D g;
        g.position = ug.position;
        g.rotation = ug.rotation;
        g.scale = ug.scale;
        g.color = ug.color;
        g.alpha = ug.alpha;
        std::copy_n(ug.shR.begin(), g.shR.size(), g.shR.begin());
        std::copy_n(ug.shG.begin(), g.shG.size(), g.shG.begin());
        std::copy_n(ug.shB.begin(), g.shB.size(), g.shB.begin());
        g.alpha    = sigmoidStandard(ug.alpha);
        g.scale[0] = std::exp(ug.scale[0]);
        g.scale[1] = std::exp(ug.scale[1]);
        g.scale[2] = std::exp(ug.scale[2]);
        gaussians3DData.push_back(g);
    }
    buffers_.count = static_cast<uint32_t>(gaussians3DData.size());
    std::cout << "Loaded " << buffers_.count << " gaussians from " << path << "\n";
}

} // namespace klartraum
