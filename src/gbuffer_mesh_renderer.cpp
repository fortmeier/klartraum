// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#include "klartraum/gbuffer_mesh_renderer.hpp"

#include <stdexcept>

#include "klartraum/computegraph/bufferelement.hpp"
#include "klartraum/storage_draw_component.hpp"

namespace klartraum {

GBufferMeshRenderer::GBufferMeshRenderer(VulkanContext& vulkanContext, const std::vector<GBufferVertex>& vertices,
                                         std::shared_ptr<ImageViewSrc> target, VkExtent2D extent) {
    if (vertices.empty() || vertices.size() % 3 != 0)
        throw std::invalid_argument("GBufferMeshRenderer: needs a non-empty triangle list");
    auto buffer = std::make_shared<BufferElementSinglePath<VulkanBuffer<GBufferVertex>>>(vulkanContext,
                                                                                         uint32_t(vertices.size()));
    buffer->setName("GBufferMeshVertices");
    buffer->getBuffer().memcopyFrom(vertices);
    camera = std::make_shared<CameraUboType>();
    current.output = glm::vec4(float(GBufferOutput::Colour), 0.0f, 1.0f, 0.0f);
    current.light = glm::vec4(glm::normalize(glm::vec3(0.45f, 0.8f, 0.35f)), 0.25f);
    current.lights[0] = glm::vec4(glm::vec3(current.light), 0.0f);
    current.lights[1] = glm::vec4(0.0f, 1.0f, 0.0f, 0.0f);
    current.lighting = glm::vec4(1.0f, 0.0f, 0.25f, float(vertices.size() / 3));
    settings =
        std::make_shared<HostValues<GBufferMeshSettings>>(vulkanContext, std::vector<GBufferMeshSettings>{current});

    StorageDrawConfig config;
    config.vertexShader = "shaders/gbuffer/gbuffer_mesh.vert.spv";
    config.fragmentShader = "shaders/gbuffer/gbuffer_mesh.frag.spv";
    config.extent = extent;
    auto draw = std::make_shared<StorageDrawComponent>(config, std::vector<BufferRef>{{buffer, -1}, {settings, -1}});
    StorageDrawComponent::Draw d;
    d.vertexCount = uint32_t(vertices.size());
    draw->addDraw(std::move(d));

    pass = std::make_shared<RenderPass>(vulkanContext.getSwapChainImageFormat(), extent);
    pass->setInput(target, 0);
    pass->setInput(camera, 1);
    pass->setDepthAttachment(true);
    pass->addComputeDependency(settings);
    pass->addDrawComponent(draw);
}

void GBufferMeshRenderer::setOutput(GBufferOutput output, float nearDistance, float farDistance) {
    current.output = glm::vec4(float(output), nearDistance, farDistance, 0.0f);
    store();
}

void GBufferMeshRenderer::setLight(glm::vec3 towardsLight, float ambient) {
    current.light = glm::vec4(glm::normalize(towardsLight), ambient);
    store();
}

void GBufferMeshRenderer::setLights(const GBufferLight& first, const GBufferLight& second, float ambient) {
    current.lights[0] = glm::vec4(glm::normalize(first.towardsLight), first.angularRadius);
    current.lights[1] = glm::vec4(glm::normalize(second.towardsLight), second.angularRadius);
    current.lighting = glm::vec4(first.intensity, second.intensity, ambient, current.lighting.w);
    store();
}

void GBufferMeshRenderer::store() { settings->set(0, current); }

std::vector<GBufferVertex> gbufferBox(glm::vec3 low, glm::vec3 high, uint32_t surfaceClass, glm::vec3 albedo,
                                      const std::array<uint32_t, 6>& faceIds) {
    std::vector<GBufferVertex> vertices;
    const glm::vec3 size = high - low;
    for (int face = 0; face < 6; ++face) {
        const int axis = face / 2;
        const float sign = face % 2 == 0 ? 1.0f : -1.0f;
        glm::vec3 normal(0.0f);
        normal[axis] = sign;
        // The face's right and up directions as seen from outside: up is +y on the sides, -z on top, +z below.
        const glm::vec3 up = axis == 1 ? glm::vec3(0.0f, 0.0f, -sign) : glm::vec3(0.0f, 1.0f, 0.0f);
        const glm::vec3 right = glm::cross(up, normal);
        const glm::vec3 centre = 0.5f * (low + high) + 0.5f * size * normal;
        const glm::vec3 halfRight = 0.5f * size * right, halfUp = 0.5f * size * up;
        const auto corner = [&](float u, float v) {
            GBufferVertex vertex{};
            vertex.pos =
                glm::vec4(centre + (2.0f * u - 1.0f) * halfRight + (2.0f * v - 1.0f) * halfUp, float(surfaceClass));
            vertex.normal = glm::vec4(normal, float(faceIds[size_t(face)]));
            vertex.albedo = glm::vec4(albedo, 0.0f);
            vertex.uv = glm::vec4(u, v, 0.0f, 0.0f);
            return vertex;
        };
        const GBufferVertex a = corner(0.0f, 0.0f), b = corner(1.0f, 0.0f), c = corner(1.0f, 1.0f),
                            d = corner(0.0f, 1.0f);
        vertices.insert(vertices.end(), {a, b, c, a, c, d});
    }
    return vertices;
}

std::vector<GBufferVertex> gbufferFloor(float halfSize, float y, float tile, uint32_t surfaceClass, glm::vec3 albedo,
                                        uint32_t objectId) {
    const auto corner = [&](float x, float z) {
        GBufferVertex vertex{};
        vertex.pos = glm::vec4(x, y, z, float(surfaceClass));
        vertex.normal = glm::vec4(0.0f, 1.0f, 0.0f, float(objectId));
        vertex.albedo = glm::vec4(albedo, 0.0f);
        vertex.uv = glm::vec4(x / tile, z / tile, 1.0f, 0.0f);
        return vertex;
    };
    const GBufferVertex a = corner(-halfSize, -halfSize), b = corner(halfSize, -halfSize),
                        c = corner(halfSize, halfSize), d = corner(-halfSize, halfSize);
    return {a, b, c, a, c, d};
}

} // namespace klartraum
