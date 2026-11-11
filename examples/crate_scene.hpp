// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#ifndef KLARTRAUM_EXAMPLES_CRATE_SCENE_HPP
#define KLARTRAUM_EXAMPLES_CRATE_SCENE_HPP

// The crate toy scene of crate_gbuffer_example and crate_training_example: a crate (a box whose six faces have their
// own surface coordinates and object ids) standing on a floor, its cameras and its G-buffers.

#include <array>
#include <cmath>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

#include <glm/gtc/matrix_transform.hpp>

#include "klartraum/computegraph/computegraph.hpp"
#include "klartraum/gbuffer_mesh_renderer.hpp"
#include "klartraum/neural/hash_texture.hpp"
#include "klartraum/neural/photo_enhancer.hpp"
#include "klartraum/neural/photo_enhancer_training.hpp"
#include "klartraum/offscreen_target.hpp"
#include "klartraum/tree/tree_model.hpp"

namespace crate {

constexpr float kFovDegrees = 40.0f;
constexpr float kDepthFar = 60.0f; // the depth window, beyond the floor's far corners

// The crate and (unless not wanted) the floor. The crate's faces get ids spread over the 16 bits, (face + 0.5) / 6 of
// the range, so that the texture's id axis holds six well separated slices; the floor sits in between at 1/2.
inline std::vector<klartraum::GBufferVertex> vertices(bool floor = true) {
    std::array<uint32_t, 6> faceIds{};
    for (uint32_t face = 0; face < 6; ++face)
        faceIds[face] = uint32_t(std::lround((float(face) + 0.5f) / 6.0f * 65535.0f));
    auto mesh = klartraum::gbufferBox(glm::vec3(-0.5f, 0.0f, -0.5f), glm::vec3(0.5f, 1.0f, 0.5f),
                                      klartraum::tree::TREE_CLASS_STEM, glm::vec3(0.62f, 0.45f, 0.28f), faceIds);
    if (!floor)
        return mesh;
    const auto plane = klartraum::gbufferFloor(40.0f, 0.0f, 4.0f, klartraum::tree::TREE_CLASS_SOIL,
                                               glm::vec3(0.5f, 0.5f, 0.48f), 32768u);
    mesh.insert(mesh.end(), plane.begin(), plane.end());
    return mesh;
}

inline void setCamera(klartraum::CameraMVP& ubo, glm::vec3 eye, glm::vec3 look) {
    ubo.model = glm::mat4(1.0f);
    ubo.view = glm::lookAt(eye, look, glm::vec3(0.0f, 1.0f, 0.0f));
    ubo.proj = glm::perspective(glm::radians(kFovDegrees), 1.0f, 0.05f, 100.0f);
    ubo.proj[1][1] *= -1.0f;
    ubo.cameraWorldPos = glm::vec4(eye, 1.0f);
}

// The surface texture of train_photo_enhancer.py: hash_texture.SurfaceTexture's defaults.
inline klartraum::neural::HashTextureShape textureShape() {
    klartraum::neural::HashTextureShape shape;
    shape.levels = 12;
    shape.features = 2;
    shape.tableSize = 1u << 17;
    shape.classes = {2, 3, 4};
    const double growth = std::exp((std::log(2048.0) - std::log(16.0)) / 11.0);
    for (uint32_t level = 0; level < shape.levels; ++level)
        shape.resolutions.push_back(uint32_t(std::floor(16.0 * std::pow(growth, double(level)))));
    return shape;
}

// The initial value of a photo enhancer parameter as train_photo_enhancer.py starts it: nn.Linear and nn.Conv2d's
// default initialisation (uniform within 1 / sqrt(fan in)), ChannelNorm's ones and zeros, the texture's +-1e-4 (times
// textureScale), the codes' N(0, 0.1); FiLM's heads and the output layer start at zero.
inline std::vector<float> initialValue(const klartraum::neural::PhotoEnhancerTraining& trainer, const std::string& name,
                                       std::mt19937& rng, float textureScale = 1.0f) {
    const auto dims = trainer.parameter(name)->getDimensions();
    size_t count = 1;
    for (uint32_t d : dims)
        count *= d;
    std::vector<float> values(count, 0.0f);
    const bool bias = name.size() > 5 && name.substr(name.size() - 5) == ".bias";
    // A block's layers 1 and 4 are its ChannelNorms ("generator.down.1.4.weight": block 1, layer 4).
    const std::string stem = name.substr(0, name.find_last_of('.'));
    const std::string layer = stem.substr(stem.find_last_of('.') + 1);
    const bool inBlock = name.rfind("generator.down.", 0) == 0 || name.rfind("generator.up.", 0) == 0;
    const bool norm = inBlock && (layer == "1" || layer == "4");
    const bool zeroStart = name.find("film.gammas") != std::string::npos ||
                           name.find("film.betas") != std::string::npos || name.find(".out.") != std::string::npos;
    if (name == "texture.table") {
        std::uniform_real_distribution<float> tiny(-1e-4f * textureScale, 1e-4f * textureScale);
        for (auto& v : values)
            v = tiny(rng);
    } else if (name == "codes.weight") {
        std::normal_distribution<float> normal(0.0f, 0.1f);
        for (auto& v : values)
            v = normal(rng);
    } else if (zeroStart) {
    } else if (norm) {
        if (!bias)
            std::fill(values.begin(), values.end(), 1.0f);
    } else {
        // The fan in of the layer the parameter belongs to (the weight's dimensions after the first).
        const auto weightDims =
            trainer.parameter(bias ? name.substr(0, name.size() - 5) + ".weight" : name)->getDimensions();
        uint32_t fanIn = 1;
        for (size_t k = 1; k < weightDims.size(); ++k)
            fanIn *= weightDims[k];
        std::uniform_real_distribution<float> uniform(-1.0f / std::sqrt(float(fanIn)), 1.0f / std::sqrt(float(fanIn)));
        for (auto& v : values)
            v = uniform(rng);
    }
    return values;
}

// Reads an OffscreenTarget image back as tightly packed BGRA bytes; the render pass leaves it in TRANSFER_SRC_OPTIMAL.
inline std::vector<uint8_t> readImage(klartraum::VulkanContext& vc, VkImage image, VkExtent2D extent) {
    const VkDeviceSize bytes = VkDeviceSize(extent.width) * extent.height * 4;
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

// The G-buffers the photo enhancer takes, of the view the renderer's camera shows.
inline klartraum::neural::GBufferFrame renderFrame(klartraum::VulkanContext& vc,
                                                   klartraum::GBufferMeshRenderer& renderer,
                                                   klartraum::ComputeGraph& graph, klartraum::OffscreenTarget& target,
                                                   uint32_t size) {
    using klartraum::GBufferOutput;
    klartraum::neural::GBufferFrame frame;
    frame.width = frame.height = size;
    frame.depthFar = kDepthFar;
    const std::pair<std::vector<uint8_t>*, GBufferOutput> outputs[] = {{&frame.colour, GBufferOutput::Colour},
                                                                       {&frame.classes, GBufferOutput::Classes},
                                                                       {&frame.depth16, GBufferOutput::Depth16},
                                                                       {&frame.u, GBufferOutput::CoordinatesU},
                                                                       {&frame.v, GBufferOutput::CoordinatesV}};
    for (const auto& [buffer, output] : outputs) {
        renderer.setOutput(output, 0.0f, kDepthFar);
        graph.submitAndWait(vc.getGraphicsQueue(), 0);
        *buffer = readImage(vc, target.getImage(0), VkExtent2D{size, size});
    }
    return frame;
}

} // namespace crate

#endif // KLARTRAUM_EXAMPLES_CRATE_SCENE_HPP
