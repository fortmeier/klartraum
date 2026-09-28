#include <algorithm>
#include <cmath>
#include <iostream>

#include <glm/glm.hpp>

#include "load-spz.h"

#include "klartraum/gaussian_data_standard.hpp"

namespace klartraum {

static float sigmoidStandard(float x) { return 1.0f / (1.0f + std::exp(-x)); }

GaussianDataStandard::GaussianDataStandard(
    VulkanContext& vulkanContext, const std::string& path, bool flipY) {
    uploadSoA(vulkanContext, loadGaussiansSpz(path, flipY));
}

GaussianDataStandard::GaussianDataStandard(VulkanContext& vulkanContext, std::vector<Gaussian3D> gaussians) {
    uploadSoA(vulkanContext, gaussians);
}

void GaussianDataStandard::uploadSoA(VulkanContext& vulkanContext, const std::vector<Gaussian3D>& gaussians) {
    const uint32_t N = static_cast<uint32_t>(gaussians.size());
    buffers_.count = N;

    // Convert AoS -> SoA and upload to GPU (single-path, static).
    std::vector<glm::vec3> pos3d(N), scale3d(N);
    std::vector<glm::vec4> rot3d(N), colAlpha3d(N);
    std::vector<float>     shR(15*N), shG(15*N), shB(15*N);

    for (uint32_t i = 0; i < N; i++) {
        const auto& g = gaussians[i];
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

    auto upload = [&](const auto& data, uint32_t count, const char* name) {
        using T = typename std::decay_t<decltype(data)>::value_type;
        auto element = std::make_shared<BufferElementSinglePath<VulkanBuffer<T>>>(vulkanContext, count);
        element->setName(name);
        element->getBuffer().memcopyFrom(data);
        return BufferRef{element};
    };
    buffers_.pos      = upload(pos3d, N, "Pos3D");
    buffers_.rot      = upload(rot3d, N, "Rot3D");
    buffers_.scale    = upload(scale3d, N, "Scale3D");
    buffers_.colAlpha = upload(colAlpha3d, N, "ColAlpha3D");
    buffers_.shR      = upload(shR, 15 * N, "ShR");
    buffers_.shG      = upload(shG, 15 * N, "ShG");
    buffers_.shB      = upload(shB, 15 * N, "ShB");
}

std::vector<Gaussian3D> loadGaussiansSpz(const std::string& path, bool flipY) {
    spz::PackedGaussians packed = spz::loadSpzPacked(path);
    std::vector<Gaussian3D> gaussians;
    gaussians.reserve(packed.numPoints);
    spz::CoordinateConverter conv = flipY
        ? spz::coordinateConverter(spz::CoordinateSystem::RUB, spz::CoordinateSystem::RDB)
        : spz::CoordinateConverter{};

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
        gaussians.push_back(g);
    }
    std::cout << "Loaded " << gaussians.size() << " gaussians from " << path;
    if (flipY) std::cout << " (Y axis flipped)";
    std::cout << "\n";
    return gaussians;
}

} // namespace klartraum
