// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#include "klartraum/rendering/gaussian_lighting.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include <glm/gtc/quaternion.hpp>

#include "klartraum/computegraph/bufferelement.hpp"

namespace klartraum {

std::array<float, 9> visibilityBasis(const glm::vec3& d) {
    return {0.282094792f,
            0.488602512f * d.y,
            0.488602512f * d.z,
            0.488602512f * d.x,
            1.092548431f * d.x * d.y,
            1.092548431f * d.y * d.z,
            0.315391565f * (3.0f * d.z * d.z - 1.0f),
            1.092548431f * d.x * d.z,
            0.546274215f * (d.x * d.x - d.y * d.y)};
}

float evaluateVisibility(const SplatMaterial& material, const glm::vec3& direction) {
    const auto b = visibilityBasis(glm::normalize(direction));
    float value = material.parameters.x * b[8];
    for (int k = 0; k < 4; ++k) {
        value += material.visibility0[k] * b[k] + material.visibility1[k] * b[k + 4];
    }
    return std::clamp(value, 0.0f, 1.0f);
}

namespace {

float percentile(std::vector<float> values, float fraction) {
    const size_t k = std::min(values.size() - 1, size_t(fraction * float(values.size() - 1) + 0.5f));
    std::nth_element(values.begin(), values.begin() + std::ptrdiff_t(k), values.end());
    return values[k];
}

} // namespace

std::vector<SplatMaterial> bakeSplatMaterials(const std::vector<Gaussian3D>& gaussians, float roughness,
                                              float metalness, float specularScale, int gridSide) {
    if (gaussians.empty()) {
        return {};
    }
    if (gridSide < 3) {
        throw std::invalid_argument("bakeSplatMaterials: the grid needs at least 3 cells per side");
    }
    const int side = gridSide;
    const size_t cells = size_t(side) * side * side;

    // The grid spans the 1st to 99th percentile of the positions, with one
    // empty cell around it; splats outside it count for the edge cells.
    glm::vec3 low, high;
    for (int axis = 0; axis < 3; ++axis) {
        std::vector<float> values(gaussians.size());
        for (size_t i = 0; i < gaussians.size(); ++i) {
            values[i] = gaussians[i].position[axis];
        }
        low[axis] = percentile(values, 0.01f);
        high[axis] = percentile(values, 0.99f);
    }
    const glm::vec3 extent = high - low;
    const float cell = std::max(std::max({extent.x, extent.y, extent.z}) / float(side - 2), 1e-5f);
    low -= glm::vec3(cell);
    auto cellOf = [&](const Gaussian3D& g) {
        const glm::vec3 p(g.position[0], g.position[1], g.position[2]);
        return glm::clamp(glm::ivec3(glm::floor((p - low) / cell)), glm::ivec3(0), glm::ivec3(side - 1));
    };
    auto index = [&](const glm::ivec3& c) { return (size_t(c.z) * side + c.y) * side + c.x; };

    // Each splat adds its largest cross-section, weighted by its opacity,
    // relative to the cell's face.
    std::vector<float> opacity(cells, 0.0f);
    for (const Gaussian3D& g : gaussians) {
        const glm::vec3 s(g.scale[0], g.scale[1], g.scale[2]);
        const float area = std::max({s.x * s.y, s.x * s.z, s.y * s.z});
        opacity[index(cellOf(g))] += g.alpha * 6.2831853f * area / (cell * cell);
    }
    std::vector<bool> occupied(cells);
    for (size_t c = 0; c < cells; ++c) {
        occupied[c] = opacity[c] > 0.0f;
        opacity[c] = 1.0f - std::exp(-opacity[c]);
    }

    // Transmittance from each occupied cell along 64 directions (a Fibonacci
    // sphere), projected onto the SH basis; the cell itself does not count.
    constexpr int kRays = 64;
    std::vector<glm::vec3> rays(kRays);
    for (int r = 0; r < kRays; ++r) {
        const float y = 1.0f - 2.0f * (float(r) + 0.5f) / float(kRays);
        const float radius = std::sqrt(1.0f - y * y);
        const float angle = float(r) * 2.39996323f;
        rays[r] = {radius * std::cos(angle), y, radius * std::sin(angle)};
    }
    std::vector<std::array<float, 9>> coefficients(cells);
    for (int z = 0; z < side; ++z) {
        for (int y = 0; y < side; ++y) {
            for (int x = 0; x < side; ++x) {
                const glm::ivec3 c(x, y, z);
                const size_t id = index(c);
                if (!occupied[id]) {
                    continue;
                }
                const glm::vec3 origin = glm::vec3(c) + glm::vec3(0.5f);
                auto& sh = coefficients[id];
                sh.fill(0.0f);
                for (const glm::vec3& d : rays) {
                    float transmission = 1.0f;
                    size_t previous = id;
                    for (float t = 0.75f; t < float(side) * 1.74f && transmission >= 0.001f; t += 0.5f) {
                        const glm::vec3 p = origin + d * t;
                        if (glm::any(glm::lessThan(p, glm::vec3(0.0f))) ||
                            glm::any(glm::greaterThanEqual(p, glm::vec3(float(side))))) {
                            break;
                        }
                        const size_t j = index(glm::ivec3(p));
                        if (j != previous && j != id) {
                            transmission *= 1.0f - opacity[j];
                        }
                        previous = j;
                    }
                    const auto b = visibilityBasis(d);
                    for (int k = 0; k < 9; ++k) {
                        sh[k] += transmission * b[k] * (12.5663706f / float(kRays));
                    }
                }
            }
        }
    }

    std::vector<SplatMaterial> materials(gaussians.size());
    for (size_t i = 0; i < gaussians.size(); ++i) {
        const Gaussian3D& g = gaussians[i];
        SplatMaterial& m = materials[i];
        const auto& sh = coefficients[index(cellOf(g))];
        m.visibility0 = {sh[0], sh[1], sh[2], sh[3]};
        m.visibility1 = {sh[4], sh[5], sh[6], sh[7]};
        m.parameters = {sh[8], metalness, specularScale, 0.0f};

        // The shortest axis is the normal of a flat splat; it faces the side
        // the splat sees more of.
        const glm::vec3 s(g.scale[0], g.scale[1], g.scale[2]);
        const int axis = s.x < s.y ? (s.x < s.z ? 0 : 2) : (s.y < s.z ? 1 : 2);
        glm::vec3 basis(0.0f);
        basis[axis] = 1.0f;
        const glm::quat q(g.rotation[3], g.rotation[0], g.rotation[1], g.rotation[2]);
        glm::vec3 normal = glm::normalize(glm::normalize(q) * basis);
        if (evaluateVisibility(m, -normal) > evaluateVisibility(m, normal)) {
            normal = -normal;
        }
        m.normalRoughness = glm::vec4(normal, roughness);
    }
    return materials;
}

GaussianLightingResult createGaussianLighting(VulkanContext& vulkanContext, const GaussianSoABuffers& input,
                                              const std::vector<SplatMaterial>& materials, const BufferRef& frame) {
    if (materials.size() != input.count) {
        throw std::invalid_argument("createGaussianLighting: need one material per Gaussian");
    }
    auto element = std::make_shared<GaussianLighting>(vulkanContext);
    input.pos.connectTo(*element, 0);
    input.colAlpha.connectTo(*element, 1);
    // Materials are uploaded once and shared by all paths.
    auto surface = std::make_shared<BufferElementSinglePath<VulkanBuffer<SplatMaterial>>>(
        vulkanContext, uint32_t(materials.size()), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    surface->getBuffer().memcopyFrom(materials);
    surface->setName("SplatMaterials");
    element->setInput(surface, 2);
    frame.connectTo(*element, 3);
    auto colour = std::make_shared<BufferElement<VulkanBuffer<glm::vec4>>>(vulkanContext, input.count);
    colour->setName("LitColAlpha");
    element->setInput(colour, 4);
    GaussianLightingResult result{element, input};
    result.output.colAlpha = BufferRef{element, 4};
    element->setPushConstants({glm::uvec4(input.count, 0u, 0u, 0u)});
    element->setGroupCount((input.count + 255) / 256, 1, 1);
    return result;
}

} // namespace klartraum
