// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#include "klartraum/tree/tree_impostors.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

#include "klartraum/computegraph/computegraph.hpp"
#include "klartraum/computegraph/renderpass.hpp"
#include "klartraum/offscreen_target.hpp"
#include "klartraum/tree/tree_generator.hpp"
#include "klartraum/tree/tree_leaf_renderer.hpp"
#include "klartraum/tree/tree_stem_renderer.hpp"

namespace klartraum {

namespace {

// The target's pixels as BGRA bytes, after the graph left it ready to be read.
std::vector<uint8_t> readTarget(VulkanContext& vc, VkImage image, VkExtent2D extent) {
    const VkDeviceSize bytes = VkDeviceSize{extent.width} * extent.height * 4;
    VkBuffer buffer;
    VkDeviceMemory memory;
    vc.createBuffer(bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, buffer, memory);
    vc.submitImmediate([&](VkCommandBuffer commandBuffer) {
        VkBufferImageCopy region{};
        region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        region.imageExtent = {extent.width, extent.height, 1};
        vkCmdCopyImageToBuffer(commandBuffer, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, buffer, 1, &region);
    });
    void* data;
    vkMapMemory(vc.getDevice(), memory, 0, bytes, 0, &data);
    std::vector<uint8_t> pixels(static_cast<uint8_t*>(data), static_cast<uint8_t*>(data) + bytes);
    vkUnmapMemory(vc.getDevice(), memory);
    vkDestroyBuffer(vc.getDevice(), buffer, nullptr);
    vkFreeMemory(vc.getDevice(), memory, nullptr);
    return pixels;
}

uint32_t pack(glm::vec4 v) {
    const glm::uvec4 b = glm::uvec4(glm::round(glm::clamp(v, 0.0f, 1.0f) * 255.0f));
    return b.x | (b.y << 8) | (b.z << 16) | (b.w << 24);
}

} // namespace

TreeImpostorAtlas::TreeImpostorAtlas(VulkanContext& vulkanContext, const std::vector<TreeTypeParams>& types,
                                     TreeImpostorConfig config) {
    if (types.empty() || config.seasons.empty() || config.seasons.size() > 4) {
        throw std::invalid_argument("TreeImpostorAtlas: needs tree types and 1 to 4 season samples");
    }
    const uint32_t views = config.views, pixels = config.pixels;
    const uint32_t seasonCount = uint32_t(config.seasons.size());
    const uint32_t typeCount = std::min<uint32_t>(uint32_t(types.size()), 2u);
    params.grid = glm::uvec4(views, pixels, seasonCount, typeCount);
    for (uint32_t k = 0; k < 4; ++k) {
        params.seasons[k] = config.seasons[std::min(k, seasonCount - 1)];
    }
    const uint32_t side = views * pixels;
    const uint32_t big = 2 * side;
    albedo.assign(size_t(typeCount) * seasonCount * side * side, 0u);
    normals.assign(albedo.size(), 0u);

    const glm::vec4 upRotation = treeUpRotation(glm::vec3(0.0f, 1.0f, 0.0f));
    const glm::quat upright(upRotation.w, upRotation.x, upRotation.y, upRotation.z);
    for (uint32_t type = 0; type < typeCount; ++type) {
        // The tree's bounds at scale 1, upright with its foot at the origin, in leaf (summer).
        TreeFrame frame = defaultTreeFrame();
        frame.wind.w = 0.0f;
        frame.timeSeason.y = 2.0f;
        TreeInstance single{};
        single.posScale = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
        single.rot = upRotation;
        single.ids = glm::uvec4(type, 1u, 0u, 0u);
        const TreeRecords records = generateTreesReference(types, {single}, frame);
        glm::vec3 lo(1e30f), hi(-1e30f);
        for (const SegmentRecord& s : records.segments) {
            for (const glm::vec4& p : {s.fromPosRadius, s.toPosRadius}) {
                lo = glm::min(lo, glm::vec3(p) - glm::vec3(p.w));
                hi = glm::max(hi, glm::vec3(p) + glm::vec3(p.w));
            }
        }
        for (const LeafRecord& l : records.leaves) {
            lo = glm::min(lo, glm::vec3(l.posScale) - glm::vec3(l.posScale.w));
            hi = glm::max(hi, glm::vec3(l.posScale) + glm::vec3(l.posScale.w));
        }
        const glm::vec3 center = 0.5f * (lo + hi);
        const float radius = 0.5f * glm::length(hi - lo) * 1.02f;
        params.bounds[type] = glm::vec4(center, radius);
        params.heights[type] = hi.y;

        // A copy per view, turned so that the camera looking along -z sees it from that view.
        const float cell = 2.0f * radius;
        std::vector<TreeInstance> instances;
        for (uint32_t j = 0; j < views; ++j) {
            for (uint32_t i = 0; i < views; ++i) {
                const glm::vec2 uv(-1.0f + (2.0f * i + 1.0f) / views, -1.0f + (2.0f * j + 1.0f) / views);
                const glm::vec3 d = tree::treeHemiOctDecode(uv);
                const glm::mat3 basis =
                    glm::transpose(glm::mat3(tree::treeImpostorRight(d), tree::treeImpostorUp(d), d));
                const glm::quat turn = glm::quat_cast(basis);
                const glm::quat rotation = turn * upright;
                const glm::vec3 middle((i + 0.5f) * cell, (views - j - 0.5f) * cell, 0.0f);
                TreeInstance inst{};
                inst.posScale = glm::vec4(middle - turn * center, 1.0f);
                inst.rot = glm::vec4(rotation.x, rotation.y, rotation.z, rotation.w);
                inst.ids = glm::uvec4(type, 1u, 0u, 0u);
                instances.push_back(inst);
            }
        }
        frame.reserved.x = 1.0f;
        auto generator = std::make_shared<TreeGenerator>(vulkanContext, types, instances,
                                                         TreeCapacity::fromReference(types, instances, frame), frame);

        // Orthographic over the grid, from in front of the copies.
        const VkExtent2D extent{big, big};
        const float width = float(views) * cell;
        auto camera = std::make_shared<CameraUboType>();
        camera->ubo.model = glm::mat4(1.0f);
        camera->ubo.view =
            glm::lookAt(glm::vec3(0.0f, 0.0f, 2.0f * radius), glm::vec3(0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
        camera->ubo.proj = glm::orthoRH_ZO(0.0f, width, 0.0f, width, 0.1f * radius, 3.5f * radius);
        // Vulkan's y points down: flip y, and with it the orthographic offset.
        camera->ubo.proj[1][1] *= -1.0f;
        camera->ubo.proj[3][1] *= -1.0f;
        // Far along +z, so that every surface sees the camera in front (for the leaves' undersides).
        camera->ubo.cameraWorldPos = glm::vec4(0.5f * width, 0.5f * width, 1e4f * radius, 1.0f);
        TreeRenderSettings values = defaultTreeRenderSettings(extent);
        // The renderers' sizes in pixels are a focal length (from viewport.y) divided by the view depth, about 2
        // radii, which an orthographic view does not divide by; and the copies are drawn at twice the size they are
        // stored at. Scaling the focal length by both makes those sizes the stored ones, so every threshold (leaf
        // simplification and thinning, thinnest stems, triangle edges) acts as in a view of the stored size.
        values.viewport.y *= radius;
        values.occlusion = glm::vec4(0.0f);
        auto settings =
            std::make_shared<HostValues<TreeRenderSettings>>(vulkanContext, std::vector<TreeRenderSettings>{values});
        TreeStemRenderer stems(vulkanContext, *generator, camera, settings, extent);
        TreeLeafRenderer leaves(vulkanContext, *generator, types, camera, settings, extent);
        auto target = std::make_shared<OffscreenTarget>(vulkanContext, extent, 1);
        auto pass = std::make_shared<RenderPass>(vulkanContext.getSwapChainImageFormat(), extent);
        pass->setInput(target, 0);
        pass->setInput(camera, 1);
        pass->setDepthAttachment(true);
        pass->addComputeDependency(stems.getClassify());
        pass->addComputeDependency(leaves.getClassify());
        pass->addComputeDependency(settings);
        pass->addDrawComponent(stems.getDrawComponent());
        pass->addDrawComponent(leaves.getDrawComponent());
        ComputeGraph graph(vulkanContext, 1);
        graph.compileFrom(pass);
        camera->update(0);

        for (uint32_t season = 0; season < seasonCount; ++season) {
            const float s = config.seasons[season];
            frame.timeSeason.y = s;
            generator->getFrame().set(0, frame);
            std::vector<uint8_t> images[2];
            for (int mode = 1; mode <= 2; ++mode) {
                values.season = glm::vec4(s, tree::treeSnowCover(s), 0.0f, 0.0f);
                values.occlusion.y = float(mode);
                settings->set(0, values);
                graph.submitAndWait(vulkanContext.getGraphicsQueue(), 0);
                images[mode - 1] = readTarget(vulkanContext, target->getImage(0), extent);
            }
            // 2 x 2 pixels to a texel: coverage, the mean albedo and normal of the covered ones.
            const size_t base = layer(type, season);
            for (uint32_t y = 0; y < side; ++y) {
                for (uint32_t x = 0; x < side; ++x) {
                    glm::vec3 a(0.0f), n(0.0f);
                    int covered = 0;
                    for (uint32_t k = 0; k < 4; ++k) {
                        const size_t p = (size_t(2 * y + k / 2) * big + 2 * x + k % 2) * 4;
                        const glm::vec3 c = glm::vec3(images[0][p + 2], images[0][p + 1], images[0][p]) / 255.0f;
                        if (c.x + c.y + c.z <= 0.0f) {
                            continue;
                        }
                        ++covered;
                        a += c * c;
                        n += glm::vec3(images[1][p + 2], images[1][p + 1], images[1][p]) / 255.0f * 2.0f -
                             glm::vec3(1.0f);
                    }
                    if (covered == 0) {
                        continue;
                    }
                    a /= float(covered);
                    n = glm::length(n) > 1e-6f ? glm::normalize(n) : glm::vec3(0.0f, 0.0f, 1.0f);
                    albedo[base + size_t(y) * side + x] = pack(glm::vec4(glm::sqrt(a), float(covered) / 4.0f));
                    normals[base + size_t(y) * side + x] = pack(glm::vec4(0.5f * n + glm::vec3(0.5f), 1.0f));
                }
            }
        }
    }

    albedoBuffer =
        std::make_shared<BufferElementSinglePath<VulkanBuffer<uint32_t>>>(vulkanContext, uint32_t(albedo.size()));
    albedoBuffer->setName("TreeImpostorAlbedo");
    albedoBuffer->getBuffer().memcopyFrom(albedo);
    normalBuffer =
        std::make_shared<BufferElementSinglePath<VulkanBuffer<uint32_t>>>(vulkanContext, uint32_t(normals.size()));
    normalBuffer->setName("TreeImpostorNormals");
    normalBuffer->getBuffer().memcopyFrom(normals);
    paramsBuffer = std::make_shared<BufferElementSinglePath<VulkanBuffer<TreeImpostorParams>>>(vulkanContext, 1u);
    paramsBuffer->setName("TreeImpostorParams");
    paramsBuffer->getBuffer().memcopyFrom(std::vector<TreeImpostorParams>{params});
}

TreeImpostorRenderer::TreeImpostorRenderer(VulkanContext& vulkanContext, BufferRef sites, BufferRef drawArgs,
                                           VkDeviceSize offset, const TreeImpostorAtlas& atlas,
                                           std::shared_ptr<HostValues<TreeRenderSettings>> settings, VkExtent2D extent,
                                           TreeRenderInputs inputs) {
    StorageDrawConfig config;
    config.vertexShader = "shaders/tree/tree_impostor.vert.spv";
    config.fragmentShader = "shaders/tree/tree_impostor.frag.spv";
    config.extent = extent;
    draw = std::make_shared<StorageDrawComponent>(
        config, std::vector<BufferRef>{sites,
                                       {settings, -1},
                                       bufferOrPlaceholder(vulkanContext, inputs.sceneDepth),
                                       bufferOrPlaceholder(vulkanContext, inputs.light),
                                       {atlas.getAlbedoBuffer(), -1},
                                       {atlas.getNormalBuffer(), -1},
                                       {atlas.getParamsBuffer(), -1}});
    StorageDrawComponent::Draw d;
    d.indirect = drawArgs;
    d.offset = offset;
    draw->addDraw(std::move(d));
}

} // namespace klartraum
