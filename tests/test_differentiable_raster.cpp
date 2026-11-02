// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

/**
 * TESTS:
 * - floatAtomicAddSumsContributions: the device supports atomic float addition on storage buffers, and many
 *   invocations adding into a few elements produce the exact sums
 * - icosphereIsClosedAndOutward: the icosphere is a closed surface (Euler characteristic 2, every edge shared by two
 *   triangles), its triangles face outwards, and its corner texture coordinates lie in [0, 2) x [0, 1] with no
 *   triangle spanning more than a quarter turn in u
 * - visibilityMatchesCpuRasterization: the visibility pass of a pumpkin mesh from two ring cameras, the second
 *   selected through the view-index buffer, equals a CPU rasterization (nearest triangle containing the pixel centre)
 *   at every pixel whose answer is unambiguous, background included
 **/

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <memory>
#include <vector>

#include <glm/gtc/type_ptr.hpp>

#include "klartraum/computegraph/bufferelement.hpp"
#include "klartraum/computegraph/computegraph.hpp"
#include "klartraum/computegraph/generalcomputation.hpp"
#include "klartraum/computegraph/tensorelement.hpp"
#include "klartraum/headless_frontend.hpp"
#include "klartraum/raster/mesh.hpp"
#include "klartraum/raster/visibility_pass.hpp"

using namespace klartraum;

namespace {

using UintBuffer = BufferElement<VulkanBuffer<uint32_t>>;
using FloatBuffer = BufferElement<VulkanBuffer<float>>;

class DifferentiableRasterTest : public ::testing::Test {
protected:
    void SetUp() override {
        frontend = std::make_unique<HeadlessFrontend>();
        vc = &frontend->getKlartraumEngine().getVulkanContext();
    }

    std::unique_ptr<HeadlessFrontend> frontend;
    VulkanContext* vc = nullptr;
};

// The mesh's positions as four floats per vertex and its triangles as three indices each.
std::vector<float> packPositions(const raster::TriangleMesh& mesh) {
    std::vector<float> values;
    for (const auto& p : mesh.positions)
        values.insert(values.end(), {p.x, p.y, p.z, 1.0f});
    return values;
}

std::vector<uint32_t> packTriangles(const raster::TriangleMesh& mesh) {
    std::vector<uint32_t> values;
    for (const auto& t : mesh.triangles)
        values.insert(values.end(), {t.x, t.y, t.z});
    return values;
}

std::vector<float> packCameras(const std::vector<glm::mat4>& cameras) {
    std::vector<float> values;
    for (const auto& m : cameras)
        values.insert(values.end(), glm::value_ptr(m), glm::value_ptr(m) + 16);
    return values;
}

// The CPU rasterization of one pixel centre: the nearest triangle containing it plus one (0: none), or -1 when the
// answer is ambiguous (the centre is close to an edge of a candidate, or two candidates are nearly equally near).
int cpuVisibility(const raster::TriangleMesh& mesh, const glm::mat4& camera, uint32_t width, uint32_t height,
                  uint32_t x, uint32_t y) {
    const glm::vec2 pixel(float(x) + 0.5f, float(y) + 0.5f);
    float nearest = std::numeric_limits<float>::max(), second = nearest;
    int best = 0;
    bool ambiguous = false;
    for (size_t t = 0; t < mesh.triangles.size(); ++t) {
        glm::vec3 screen[3];
        bool behind = false;
        for (int c = 0; c < 3; ++c) {
            const glm::vec4 clip = camera * glm::vec4(mesh.positions[mesh.triangles[t][c]], 1.0f);
            behind |= clip.w <= 0.0f;
            const glm::vec3 ndc = glm::vec3(clip) / clip.w;
            screen[c] = glm::vec3((ndc.x * 0.5f + 0.5f) * float(width), (ndc.y * 0.5f + 0.5f) * float(height), ndc.z);
        }
        if (behind)
            continue;
        const glm::vec2 a(screen[0]), b(screen[1]), c(screen[2]);
        const float area = (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
        if (std::abs(area) < 1e-6f)
            continue;
        const float w0 = ((b.x - pixel.x) * (c.y - pixel.y) - (b.y - pixel.y) * (c.x - pixel.x)) / area;
        const float w1 = ((c.x - pixel.x) * (a.y - pixel.y) - (c.y - pixel.y) * (a.x - pixel.x)) / area;
        const float w2 = 1.0f - w0 - w1;
        const float margin = 2e-3f;
        if (w0 < -margin || w1 < -margin || w2 < -margin)
            continue;
        if (w0 < margin || w1 < margin || w2 < margin) {
            ambiguous = true; // too close to an edge to tell
            continue;
        }
        const float depth = w0 * screen[0].z + w1 * screen[1].z + w2 * screen[2].z; // NDC depth is affine on screen
        if (depth < nearest) {
            second = nearest;
            nearest = depth;
            best = int(t) + 1;
        } else {
            second = std::min(second, depth);
        }
    }
    if (ambiguous || (best != 0 && second - nearest < 1e-5f))
        return -1;
    return best;
}

} // namespace

TEST_F(DifferentiableRasterTest, floatAtomicAddSumsContributions) {
    ASSERT_TRUE(vc->isBufferFloatAtomicAddSupported());
    const uint32_t count = 100000;
    struct PushConstants {
        uint32_t count;
    };
    auto sums = std::make_shared<FloatBuffer>(*vc, 7);
    auto add = std::make_shared<GeneralComputation<PushConstants>>(*vc, "shaders/test/float_atomic_add.comp.spv");
    add->setPushConstants({{count}});
    add->setGroupCountX((count + 255) / 256);
    add->setInput(sums, 0);
    ComputeGraph graph(*vc, 1);
    graph.compileFrom(add);
    sums->getBuffer(0).memcopyFrom(std::vector<float>(7, 0.0f));
    graph.submitAndWait(vc->getGraphicsQueue(), 0);
    std::vector<float> result(7);
    sums->getBuffer(0).memcopyTo(result);
    for (uint32_t bucket = 0; bucket < 7; ++bucket) {
        double expected = 0.0;
        for (uint32_t i = bucket; i < count; i += 7)
            expected += double(i % 5) + 0.25;
        EXPECT_FLOAT_EQ(result[bucket], float(expected)) << bucket; // quarters sum exactly in float
    }
}

TEST_F(DifferentiableRasterTest, icosphereIsClosedAndOutward) {
    const auto mesh = raster::icosphere(3);
    std::map<std::pair<uint32_t, uint32_t>, int> edges;
    for (const auto& t : mesh.triangles) {
        for (int c = 0; c < 3; ++c)
            ++edges[std::minmax(t[c], t[(c + 1) % 3])];
        const glm::vec3 a = mesh.positions[t.x], b = mesh.positions[t.y], c = mesh.positions[t.z];
        EXPECT_GT(glm::dot(glm::cross(b - a, c - a), a + b + c), 0.0f);
    }
    for (const auto& [edge, uses] : edges)
        EXPECT_EQ(uses, 2);
    EXPECT_EQ(int(mesh.positions.size()) - int(edges.size()) + int(mesh.triangles.size()), 2);
    ASSERT_EQ(mesh.cornerUvs.size(), 3 * mesh.triangles.size());
    for (size_t t = 0; t < mesh.triangles.size(); ++t) {
        float lo = 2.0f, hi = -1.0f;
        for (int c = 0; c < 3; ++c) {
            const glm::vec2 uv = mesh.cornerUvs[3 * t + c];
            EXPECT_GE(uv.x, 0.0f);
            EXPECT_LT(uv.x, 2.0f);
            EXPECT_GE(uv.y, 0.0f);
            EXPECT_LE(uv.y, 1.0f);
            lo = std::min(lo, uv.x);
            hi = std::max(hi, uv.x);
        }
        EXPECT_LT(hi - lo, 0.25f) << "triangle " << t;
    }
}

TEST_F(DifferentiableRasterTest, visibilityMatchesCpuRasterization) {
    const uint32_t width = 96, height = 80;
    const auto mesh = raster::syntheticPumpkin(3);
    const auto cameras = raster::cameraRing(3, 3.0f, 0.4f, 0.8f, float(width) / float(height));
    const uint32_t vertices = uint32_t(mesh.positions.size()), triangleCount = uint32_t(mesh.triangles.size());

    auto positions = vc->create<TensorElement<float>>(std::vector<uint32_t>{vertices, 4});
    auto triangles = std::make_shared<UintBuffer>(*vc, 3 * triangleCount);
    auto cameraBuffer = vc->create<TensorElement<float>>(std::vector<uint32_t>{uint32_t(cameras.size()), 16});
    auto viewIndex = std::make_shared<UintBuffer>(*vc, 1);
    auto visibility = std::make_shared<UintBuffer>(*vc, width * height);
    auto pass = std::make_shared<raster::VisibilityPass>(*vc, width, height, triangleCount);
    pass->setInput(positions, 0);
    pass->setInput(triangles, 1);
    pass->setInput(cameraBuffer, 2);
    pass->setInput(viewIndex, 3);
    pass->setInput(visibility, 4);
    ComputeGraph graph(*vc, 1);
    graph.compileFrom(pass);
    positions->getDataBuffer(0).memcopyFrom(packPositions(mesh));
    triangles->getBuffer(0).memcopyFrom(packTriangles(mesh));
    cameraBuffer->getDataBuffer(0).memcopyFrom(packCameras(cameras));

    for (const uint32_t view : {0u, 2u}) {
        viewIndex->getBuffer(0).memcopyFrom(std::vector<uint32_t>{view});
        graph.submitAndWait(vc->getGraphicsQueue(), 0);
        std::vector<uint32_t> ids(width * height);
        visibility->getBuffer(0).memcopyTo(ids);
        int compared = 0, covered = 0;
        for (uint32_t y = 0; y < height; ++y) {
            for (uint32_t x = 0; x < width; ++x) {
                const int expected = cpuVisibility(mesh, cameras[view], width, height, x, y);
                if (expected < 0)
                    continue;
                ++compared;
                covered += expected > 0;
                ASSERT_EQ(int(ids[y * width + x]), expected) << "view " << view << " pixel " << x << ", " << y;
            }
        }
        EXPECT_GT(compared, int(width * height) * 8 / 10) << "view " << view; // most pixels are unambiguous
        EXPECT_GT(covered, int(width * height) / 10) << "view " << view;      // the pumpkin is in the picture
    }
}
