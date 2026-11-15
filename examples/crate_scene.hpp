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
#include <stdexcept>
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

// A surface texture as hash_texture.SurfaceTexture makes it: `levels` resolutions growing geometrically from `base` to
// `finest` cells per axis (the defaults are its own: 12 levels, 16 to 2048).
inline klartraum::neural::HashTextureShape textureShape(uint32_t levels = 12, uint32_t base = 16,
                                                        uint32_t finest = 2048) {
    klartraum::neural::HashTextureShape shape;
    shape.levels = levels;
    shape.features = 2;
    shape.tableSize = 1u << 17;
    shape.classes = {2, 3, 4};
    const double growth = std::exp((std::log(double(finest)) - std::log(double(base))) / std::max(1.0, levels - 1.0));
    for (uint32_t level = 0; level < shape.levels; ++level)
        shape.resolutions.push_back(uint32_t(std::floor(double(base) * std::pow(growth, double(level)))));
    return shape;
}

// A plain dense texture of `components` numbers per point instead of a multiresolution hash texture: every level at
// the same `resolution`, each holding up to 4 of the components, its table large enough for the (resolution + 1)^3
// corners to be indexed directly (no hash collisions); a grid for the crate's class only. 0 components: no texture.
inline klartraum::neural::HashTextureShape denseTextureShape(uint32_t components, uint32_t resolution) {
    klartraum::neural::HashTextureShape shape;
    shape.classes = {klartraum::tree::TREE_CLASS_STEM};
    if (components == 0) {
        shape.levels = 0;
        return shape;
    }
    if (components > 64 || (components > 4 && components % 4 != 0))
        throw std::invalid_argument("denseTextureShape: 1 to 4 components, or a multiple of 4 up to 64");
    shape.features = std::min(components, 4u);
    shape.levels = components <= 4 ? 1u : components / 4u;
    const uint64_t corners = uint64_t(resolution + 1) * (resolution + 1) * (resolution + 1);
    shape.tableSize = 1;
    while (shape.tableSize < corners)
        shape.tableSize <<= 1;
    shape.resolutions.assign(shape.levels, resolution);
    return shape;
}

// The initial value of a photo enhancer parameter as train_photo_enhancer.py starts it: nn.Linear and nn.Conv2d's
// default initialisation (uniform within 1 / sqrt(fan in)), ChannelNorm's ones and zeros, the texture's +-1e-4 (times
// textureScale), the codes' and the latent table's N(0, 0.1); FiLM's heads and the output layer start at zero.
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
    // The encoder's ChannelNorms are its layers 1, 4, 7 ("encoder.body.4.weight").
    const bool encoderNorm = name.rfind("encoder.body.", 0) == 0 && (std::stoi(layer) % 3) == 1;
    const bool norm = (inBlock && (layer == "1" || layer == "4")) || encoderNorm;
    const bool zeroStart = name.find("film.gammas") != std::string::npos ||
                           name.find("film.betas") != std::string::npos || name.find(".out.") != std::string::npos;
    if (name == "texture.table") {
        std::uniform_real_distribution<float> tiny(-1e-4f * textureScale, 1e-4f * textureScale);
        for (auto& v : values)
            v = tiny(rng);
    } else if (name == "codes.weight" || name == "latents.weight") {
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
    const std::pair<std::vector<uint8_t>*, GBufferOutput> outputs[] = {
        {&frame.colour, GBufferOutput::Colour},   {&frame.classes, GBufferOutput::Classes},
        {&frame.depth16, GBufferOutput::Depth16}, {&frame.u, GBufferOutput::CoordinatesU},
        {&frame.v, GBufferOutput::CoordinatesV},  {&frame.normals, GBufferOutput::Normals}};
    for (const auto& [buffer, output] : outputs) {
        renderer.setOutput(output, 0.0f, kDepthFar);
        graph.submitAndWait(vc.getGraphicsQueue(), 0);
        *buffer = readImage(vc, target.getImage(0), VkExtent2D{size, size});
    }
    return frame;
}

// The flow of view b onto view a, both rendered at size x size: for every pixel of b, where a shows the same surface
// point. The pixel's ray through b's camera at the distance of b's Depth16 G-buffer (over kDepthFar; 0 where nothing
// was drawn) gives the point, a's camera projects it; it counts as seen by a (valid) when it falls inside a and a's
// own distance at the nearest pixel agrees within 2 % (otherwise something in front of it occludes it).
struct Flow {
    uint32_t size = 0;
    std::vector<float> x, y;    ///< b's pixels' positions in a, in a's pixel coordinates (pixel centres at i + 0.5)
    std::vector<uint8_t> valid; ///< 1 where a sees b's point
};

inline float distance16(const std::vector<uint8_t>& bgra, size_t p, float depthFar = kDepthFar) {
    return float(bgra[4 * p + 2] * 256u + bgra[4 * p + 1]) / 65535.0f * depthFar;
}

// The depth windows' far ends default to the crate scene's (kDepthFar); flights pass each frame's own.
inline Flow flow(const klartraum::CameraMVP& a, const std::vector<uint8_t>& depthA, const klartraum::CameraMVP& b,
                 const std::vector<uint8_t>& depthB, uint32_t size, float farA = kDepthFar, float farB = kDepthFar) {
    Flow result;
    result.size = size;
    const size_t pixels = size_t(size) * size;
    result.x.assign(pixels, 0.0f);
    result.y.assign(pixels, 0.0f);
    result.valid.assign(pixels, 0);
    const glm::mat4 toWorldB = glm::inverse(b.proj * b.view), toClipA = a.proj * a.view;
    const glm::vec3 eyeA(a.cameraWorldPos), eyeB(b.cameraWorldPos);
    for (uint32_t row = 0; row < size; ++row)
        for (uint32_t column = 0; column < size; ++column) {
            const size_t p = size_t(row) * size + column;
            const float distanceB = distance16(depthB, p, farB);
            if (distanceB <= 0.0f)
                continue;
            const glm::vec4 far = toWorldB * glm::vec4((column + 0.5f) / size * 2.0f - 1.0f,
                                                       (row + 0.5f) / size * 2.0f - 1.0f, 1.0f, 1.0f);
            const glm::vec3 point = eyeB + glm::normalize(glm::vec3(far) / far.w - eyeB) * distanceB;
            const glm::vec4 clip = toClipA * glm::vec4(point, 1.0f);
            if (clip.w <= 0.0f)
                continue;
            const float x = (clip.x / clip.w + 1.0f) * 0.5f * size, y = (clip.y / clip.w + 1.0f) * 0.5f * size;
            result.x[p] = x;
            result.y[p] = y;
            if (x < 0.0f || y < 0.0f || x >= float(size) || y >= float(size))
                continue;
            const float distanceA = distance16(depthA, size_t(y) * size + size_t(x), farA);
            const float expected = glm::length(point - eyeA);
            result.valid[p] = distanceA > 0.0f && std::abs(distanceA - expected) < 0.02f * expected;
        }
    return result;
}

} // namespace crate

#endif // KLARTRAUM_EXAMPLES_CRATE_SCENE_HPP
