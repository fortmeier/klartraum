// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

/**
 * TESTS:
 * - boxHasSixFacesWithOwnIds: gbufferBox() gives 12 triangles, each face's normal points away from the box centre and
 *   its corners carry the face's id and coordinates in [0, 1]
 * - classesShowBoxAndFloor: a box on a floor seen from the front: the image centre has the box's class colour, the
 *   bottom the floor's and the top (above the horizon of a finite floor seen level) the black background
 * - depth16IsTheDistance: the 16-bit depth at the image centre decodes to the distance from the camera to the box's
 *   front face
 * - coordinatesFollowTheFace: at the image centre the coordinates are those of the middle of the front face (u and v
 *   near 0.5) with its object id, and they grow to the right and upwards
 * - normalsShowTheFaces: the normals G-buffer of the box's front face (seen from +z) decodes to (0, 0, 1), of the
 *   floor to (0, 1, 0)
 * - colourIsLitFromTheLight: the face turned towards the light is brighter than the one turned away
 **/

#include <gtest/gtest.h>

#include <cmath>
#include <memory>

#include "klartraum/computegraph/computegraph.hpp"
#include "klartraum/gbuffer_mesh_renderer.hpp"
#include "klartraum/headless_frontend.hpp"
#include "klartraum/offscreen_target.hpp"
#include "klartraum/tree/tree_model.hpp"
#include "tree_test_helpers.hpp"

using namespace klartraum;
using namespace klartraum::tree_test;

namespace {

constexpr VkExtent2D kExtent{64, 64};
const std::array<uint32_t, 6> kFaceIds{100, 200, 300, 400, 500, 600};

// A unit box standing on a floor, the camera in front of it (+z) at the height of its centre.
struct BoxScene {
    BoxScene() {
        vc = &frontend.getKlartraumEngine().getVulkanContext();
        auto vertices = gbufferBox(glm::vec3(-0.5f, 0.0f, -0.5f), glm::vec3(0.5f, 1.0f, 0.5f), tree::TREE_CLASS_STEM,
                                   glm::vec3(0.6f, 0.45f, 0.3f), kFaceIds);
        const auto floor = gbufferFloor(6.0f, 0.0f, 4.0f, tree::TREE_CLASS_SOIL, glm::vec3(0.5f), 7);
        vertices.insert(vertices.end(), floor.begin(), floor.end());
        target = std::make_shared<OffscreenTarget>(*vc, kExtent, 1);
        renderer = std::make_unique<GBufferMeshRenderer>(*vc, vertices, target, kExtent);
        graph = std::make_unique<ComputeGraph>(*vc, 1);
        graph->compileFrom(renderer->getOutput());
    }

    // Draws `output` seen from `eye` towards the box centre; BGRA bytes.
    std::vector<uint8_t> draw(GBufferOutput output, glm::vec3 eye, float nearDistance = 0.0f,
                              float farDistance = 1.0f) {
        setCamera(renderer->getCamera()->ubo, eye, glm::vec3(0.0f, 0.5f, 0.0f), 40.0f, 1.0f, 0.05f, 50.0f);
        renderer->getCamera()->update(0);
        renderer->setOutput(output, nearDistance, farDistance);
        graph->submitAndWait(vc->getGraphicsQueue(), 0);
        return readImage(*vc, target->getImage(0), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, kExtent);
    }

    HeadlessFrontend frontend;
    VulkanContext* vc = nullptr;
    std::shared_ptr<OffscreenTarget> target;
    std::unique_ptr<GBufferMeshRenderer> renderer;
    std::unique_ptr<ComputeGraph> graph;
};

size_t at(uint32_t x, uint32_t y) { return (size_t(y) * kExtent.width + x) * 4; }

glm::vec3 rgb(const std::vector<uint8_t>& bgra, uint32_t x, uint32_t y) {
    const size_t k = at(x, y);
    return glm::vec3(bgra[k + 2], bgra[k + 1], bgra[k]) / 255.0f;
}

float value16(const std::vector<uint8_t>& bgra, uint32_t x, uint32_t y) {
    const size_t k = at(x, y);
    return float(bgra[k + 2] * 256u + bgra[k + 1]) / 65535.0f;
}

} // namespace

TEST(GBufferMeshRenderer, boxHasSixFacesWithOwnIds) {
    const auto box =
        gbufferBox(glm::vec3(-1.0f, 0.0f, -2.0f), glm::vec3(1.0f, 3.0f, 2.0f), 3, glm::vec3(1.0f), kFaceIds);
    ASSERT_EQ(box.size(), 36u);
    const glm::vec3 centre(0.0f, 1.5f, 0.0f);
    for (size_t face = 0; face < 6; ++face)
        for (size_t k = 0; k < 6; ++k) {
            const GBufferVertex& v = box[face * 6 + k];
            EXPECT_GT(glm::dot(glm::vec3(v.normal), glm::vec3(v.pos) - centre), 0.0f) << face;
            EXPECT_EQ(uint32_t(v.normal.w), kFaceIds[face]);
            EXPECT_EQ(uint32_t(v.pos.w), 3u);
            EXPECT_GE(v.uv.x, 0.0f);
            EXPECT_LE(v.uv.x, 1.0f);
            EXPECT_GE(v.uv.y, 0.0f);
            EXPECT_LE(v.uv.y, 1.0f);
            EXPECT_EQ(v.uv.z, 0.0f);
        }
}

TEST(GBufferMeshRenderer, classesShowBoxAndFloor) {
    BoxScene scene;
    const auto classes = scene.draw(GBufferOutput::Classes, glm::vec3(0.0f, 0.5f, 3.0f));
    EXPECT_LT(glm::length(rgb(classes, 32, 32) - tree::treeClassColor(tree::TREE_CLASS_STEM)), 0.01f);
    EXPECT_LT(glm::length(rgb(classes, 32, 62) - tree::treeClassColor(tree::TREE_CLASS_SOIL)), 0.01f);
    EXPECT_LT(glm::length(rgb(classes, 32, 1)), 0.01f);
}

TEST(GBufferMeshRenderer, depth16IsTheDistance) {
    BoxScene scene;
    const auto depth = scene.draw(GBufferOutput::Depth16, glm::vec3(0.0f, 0.5f, 3.0f), 0.0f, 10.0f);
    // The ray through the centre of pixel (32, 32) is a little off the axis; the front face is at z = 0.5.
    const float tan = std::tan(glm::radians(20.0f)) / 32.0f * 0.5f;
    const float expected = 2.5f * std::sqrt(1.0f + 2.0f * tan * tan);
    EXPECT_NEAR(value16(depth, 32, 32) * 10.0f, expected, 2e-3f);
}

TEST(GBufferMeshRenderer, coordinatesFollowTheFace) {
    BoxScene scene;
    const glm::vec3 eye(0.0f, 0.5f, 3.0f);
    const auto us = scene.draw(GBufferOutput::CoordinatesU, eye);
    const auto vs = scene.draw(GBufferOutput::CoordinatesV, eye);
    EXPECT_NEAR(value16(us, 32, 32), 0.5f, 0.03f);
    EXPECT_NEAR(value16(vs, 32, 32), 0.5f, 0.03f);
    EXPECT_GT(value16(us, 40, 32), value16(us, 32, 32));
    EXPECT_GT(value16(vs, 32, 24), value16(vs, 32, 32));
    // The front face is +z, the fifth face; its id is split into the blue channels.
    const uint32_t id = uint32_t(us[at(32, 32)]) * 256u + vs[at(32, 32)];
    EXPECT_EQ(id, kFaceIds[4]);
    // The floor repeats every 4 units: its coordinates stay in [0, 1) with the floor's id.
    const uint32_t floorId = uint32_t(us[at(32, 62)]) * 256u + vs[at(32, 62)];
    EXPECT_EQ(floorId, 7u);
}

TEST(GBufferMeshRenderer, colourIsLitFromTheLight) {
    BoxScene scene;
    scene.renderer->setLight(glm::vec3(1.0f, 0.0f, 0.0f), 0.2f);
    // Seen from the front right corner: the +x face on the right, the +z face on the left.
    const auto lit = scene.draw(GBufferOutput::Colour, glm::vec3(2.2f, 0.5f, 2.2f));
    scene.renderer->setLight(glm::vec3(-1.0f, 0.0f, 0.0f), 0.2f);
    const auto unlit = scene.draw(GBufferOutput::Colour, glm::vec3(2.2f, 0.5f, 2.2f));
    EXPECT_GT(glm::length(rgb(lit, 40, 32)), 2.0f * glm::length(rgb(unlit, 40, 32)));
}

TEST(GBufferMeshRenderer, normalsShowTheFaces) {
    BoxScene scene;
    const auto normals = scene.draw(GBufferOutput::Normals, glm::vec3(0.0f, 0.5f, 3.0f));
    const glm::vec3 front = rgb(normals, 32, 32) * 2.0f - 1.0f, floor = rgb(normals, 32, 62) * 2.0f - 1.0f;
    EXPECT_LT(glm::length(front - glm::vec3(0.0f, 0.0f, 1.0f)), 0.02f);
    EXPECT_LT(glm::length(floor - glm::vec3(0.0f, 1.0f, 0.0f)), 0.02f);
}
