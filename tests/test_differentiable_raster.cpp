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
 * - laplacianRegularizerMatchesFiniteDifferences: the vertex adjacency of the icosphere is symmetric with five or
 *   six neighbours per vertex, and the Laplacian smoothness loss of a perturbed icosphere and its gradients equal a
 *   CPU evaluation and its central differences
 * - visibilityMatchesCpuRasterization: the visibility pass of a pumpkin mesh from two ring cameras, the second
 *   selected through the view-index buffer, equals a CPU rasterization (nearest triangle containing the pixel centre)
 *   at every pixel whose answer is unambiguous, background included
 * - shadingMatchesCpu: the shaded colour of every covered pixel equals a CPU evaluation of the projection,
 *   perspective-correct barycentrics, texture coordinate and bilinear lookup, and background pixels are zero
 * - shadingGradientsMatchFiniteDifferences: the masked colour loss and its gradients with respect to sampled texels
 *   and vertex positions equal the CPU loss and its central differences, with the visibility of the GPU held fixed
 * - textureRecoveryOnFixedGeometry: training only the texture of the pumpkin from a noise start on eight rendered
 *   views, one submit per step with the view advanced on the GPU, cuts the loss below 2 % of its first value and
 *   reproduces a held-out view with a PSNR above 30 dB inside the mesh
 * - antialiasMatchesCpu: the antialiased colours equal a CPU evaluation of the silhouette crossings and blends on the
 *   GPU visibility, change pixels along the outline only, and leave the image interior untouched
 * - antialiasGradientsMatchFiniteDifferences: the image loss of the antialiased image (colour inside the mask,
 *   coverage everywhere) and its gradients with respect to sampled texels and vertex positions, silhouette vertices
 *   included, equal the CPU loss and its central differences, with the visibility of the GPU held fixed
 * - silhouetteFittingFromMasks: training only the vertex positions of a sphere on the antialiased masks of eight
 *   views of an ellipsoid, one submit per step, reaches a silhouette IoU above 0.95 on a held-out view
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
#include "klartraum/layers/training.hpp"
#include "klartraum/raster/differentiable_shading.hpp"
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

// A textured mesh seen by a set of cameras, with the GPU buffers of the visibility pass and the shading elements.
struct Scene {
    raster::TriangleMesh mesh;
    std::vector<glm::mat4> cameras;
    std::vector<float> texture; // four floats per texel
    uint32_t width = 0, height = 0, textureWidth = 0, textureHeight = 0;
    std::shared_ptr<TensorElement<float>> positions, cornerUvs, texels, cameraBuffer;
    std::shared_ptr<UintBuffer> triangles, viewIndex, visibility;
    std::shared_ptr<raster::VisibilityPass> pass;

    // Elements and tensors belong to one graph each: a scene serves one graph.
    Scene(VulkanContext& vc, uint32_t width, uint32_t height, uint32_t textureWidth, uint32_t textureHeight,
          std::vector<glm::mat4> sceneCameras = {}, raster::TriangleMesh sceneMesh = raster::syntheticPumpkin(2))
        : mesh(std::move(sceneMesh)),
          width(width),
          height(height),
          textureWidth(textureWidth),
          textureHeight(textureHeight) {
        cameras = sceneCameras.empty() ? raster::cameraRing(3, 3.0f, 0.4f, 0.8f, float(width) / float(height))
                                       : std::move(sceneCameras);
        for (uint32_t j = 0; j < textureHeight; ++j)
            for (uint32_t i = 0; i < textureWidth; ++i)
                texture.insert(texture.end(), {0.5f + 0.4f * std::sin(0.9f * float(i)), 0.3f + 0.02f * float(j),
                                               float((i * 7 + j * 3) % 11) / 11.0f, 1.0f});
        const uint32_t vertices = uint32_t(mesh.positions.size()), triangleCount = uint32_t(mesh.triangles.size());
        positions = vc.create<TensorElement<float>>(std::vector<uint32_t>{vertices, 4});
        cornerUvs = vc.create<TensorElement<float>>(std::vector<uint32_t>{3 * triangleCount, 2});
        texels = vc.create<TensorElement<float>>(std::vector<uint32_t>{textureWidth * textureHeight, 4});
        cameraBuffer = vc.create<TensorElement<float>>(std::vector<uint32_t>{uint32_t(cameras.size()), 16});
        triangles = std::make_shared<UintBuffer>(vc, 3 * triangleCount);
        viewIndex = std::make_shared<UintBuffer>(vc, 1);
        visibility = std::make_shared<UintBuffer>(vc, width * height);
        pass = std::make_shared<raster::VisibilityPass>(vc, width, height, triangleCount);
        pass->setInput(positions, 0);
        pass->setInput(triangles, 1);
        pass->setInput(cameraBuffer, 2);
        pass->setInput(viewIndex, 3);
        pass->setInput(visibility, 4);
    }

    // Connects the shared shading slots 0-6 of `element`, the visibility from the pass.
    void connectShading(const ComputeGraphElementPtr& element) const {
        element->setInput(positions, 0);
        element->setInput(triangles, 1);
        element->setInput(cornerUvs, 2);
        element->setInput(texels, 3);
        element->setInput(cameraBuffer, 4);
        element->setInput(viewIndex, 5);
        element->setInput(pass, 6, 4);
    }

    // Uploads the mesh, texture and cameras and selects `view`; call after compiling.
    void upload(uint32_t view) const {
        positions->getDataBuffer(0).memcopyFrom(packPositions(mesh));
        triangles->getBuffer(0).memcopyFrom(packTriangles(mesh));
        std::vector<float> uvs;
        for (const auto& uv : mesh.cornerUvs)
            uvs.insert(uvs.end(), {uv.x, uv.y});
        cornerUvs->getDataBuffer(0).memcopyFrom(uvs);
        texels->getDataBuffer(0).memcopyFrom(texture);
        cameraBuffer->getDataBuffer(0).memcopyFrom(packCameras(cameras));
        viewIndex->getBuffer(0).memcopyFrom(std::vector<uint32_t>{view});
    }
};

// The CPU shading of pixel (x, y) showing `triangle`, in double precision (as differentiable_shading.slang).
glm::dvec3 cpuShade(const raster::TriangleMesh& mesh, const std::vector<double>& texture, uint32_t textureWidth,
                    uint32_t textureHeight, const glm::mat4& camera, uint32_t width, uint32_t height, uint32_t triangle,
                    uint32_t x, uint32_t y, const std::vector<glm::dvec3>* positions = nullptr) {
    const glm::dvec2 pixel(x + 0.5, y + 0.5), size(width, height);
    glm::dvec2 s[3], uv[3];
    double w[3];
    for (int c = 0; c < 3; ++c) {
        const uint32_t vertex = mesh.triangles[triangle][c];
        const glm::dvec3 p = positions ? (*positions)[vertex] : glm::dvec3(mesh.positions[vertex]);
        const glm::dvec4 clip = glm::dmat4(camera) * glm::dvec4(p, 1.0);
        s[c] = (glm::dvec2(clip) / clip.w * 0.5 + 0.5) * size;
        w[c] = clip.w;
        uv[c] = glm::dvec2(mesh.cornerUvs[3 * triangle + c]);
    }
    const double area = (s[1].x - s[0].x) * (s[2].y - s[0].y) - (s[1].y - s[0].y) * (s[2].x - s[0].x);
    const double l0 = ((s[1].x - pixel.x) * (s[2].y - pixel.y) - (s[1].y - pixel.y) * (s[2].x - pixel.x)) / area;
    const double l1 = ((s[2].x - pixel.x) * (s[0].y - pixel.y) - (s[2].y - pixel.y) * (s[0].x - pixel.x)) / area;
    const double l2 = 1.0 - l0 - l1;
    const double q0 = l0 / w[0], q1 = l1 / w[1], q2 = l2 / w[2];
    const glm::dvec2 coordinate = (q0 * uv[0] + q1 * uv[1] + q2 * uv[2]) / (q0 + q1 + q2);
    const double tx = coordinate.x * textureWidth - 0.5, ty = coordinate.y * textureHeight - 0.5;
    const double fx = std::floor(tx), fy = std::floor(ty);
    const int tw = int(textureWidth), th = int(textureHeight);
    const int i0 = ((int(fx) % tw) + tw) % tw, i1 = (i0 + 1) % tw;
    const int j0 = std::clamp(int(fy), 0, th - 1), j1 = std::clamp(int(fy) + 1, 0, th - 1);
    auto texel = [&](int i, int j) {
        const size_t k = 4 * size_t(j * tw + i);
        return glm::dvec3(texture[k], texture[k + 1], texture[k + 2]);
    };
    const double ax = tx - fx, ay = ty - fy;
    return glm::mix(glm::mix(texel(i0, j0), texel(i1, j0), ax), glm::mix(texel(i0, j1), texel(i1, j1), ax), ay);
}

// The CPU reference of antialias.slang and image_loss.comp.slang on a fixed visibility buffer, in double precision.
struct CpuImage {
    const Scene& scene;
    const std::vector<uint32_t>& ids;
    const std::vector<glm::dvec3>& positions;
    const std::vector<double>& texture;
    uint32_t view;

    glm::dvec3 screen(uint32_t vertex) const {
        const glm::dvec4 clip = glm::dmat4(scene.cameras[view]) * glm::dvec4(positions[vertex], 1.0);
        return glm::dvec3((glm::dvec2(clip) / clip.w * 0.5 + 0.5) * glm::dvec2(scene.width, scene.height),
                          clip.z / clip.w);
    }

    // The shaded, premultiplied colour of pixel (x, y): rgb and coverage 1, zero for background.
    glm::dvec4 color(int x, int y) const {
        const uint32_t id = ids[uint32_t(y) * scene.width + uint32_t(x)];
        if (id == 0)
            return glm::dvec4(0.0);
        return glm::dvec4(cpuShade(scene.mesh, texture, scene.textureWidth, scene.textureHeight, scene.cameras[view],
                                   scene.width, scene.height, id - 1, uint32_t(x), uint32_t(y), &positions),
                          1.0);
    }

    double depth(uint32_t triangle, glm::dvec2 pixel) const {
        glm::dvec3 s[3];
        for (int c = 0; c < 3; ++c)
            s[c] = screen(scene.mesh.triangles[triangle][c]);
        const double area = (s[1].x - s[0].x) * (s[2].y - s[0].y) - (s[1].y - s[0].y) * (s[2].x - s[0].x);
        const double l0 = ((s[1].x - pixel.x) * (s[2].y - pixel.y) - (s[1].y - pixel.y) * (s[2].x - pixel.x)) / area;
        const double l1 = ((s[2].x - pixel.x) * (s[0].y - pixel.y) - (s[2].y - pixel.y) * (s[0].x - pixel.x)) / area;
        return l0 * s[0].z + l1 * s[1].z + (1.0 - l0 - l1) * s[2].z;
    }

    // How much pixel `self` blends towards neighbour `other` (as blendWeight(findSilhouette()) in antialias.slang).
    double blendWeight(glm::ivec2 self, glm::ivec2 other) const {
        const uint32_t selfId = ids[uint32_t(self.y) * scene.width + uint32_t(self.x)];
        const uint32_t otherId = ids[uint32_t(other.y) * scene.width + uint32_t(other.x)];
        if (selfId == otherId)
            return 0.0;
        const glm::dvec2 selfPixel = glm::dvec2(self) + 0.5, otherPixel = glm::dvec2(other) + 0.5;
        const bool frontIsSelf =
            selfId != 0 && (otherId == 0 || depth(selfId - 1, selfPixel) < depth(otherId - 1, otherPixel));
        const uint32_t frontId = frontIsSelf ? selfId : otherId, backId = frontIsSelf ? otherId : selfId;
        const glm::dvec2 from = frontIsSelf ? selfPixel : otherPixel, to = frontIsSelf ? otherPixel : selfPixel;
        const glm::uvec3 front = scene.mesh.triangles[frontId - 1];
        double best = 2.0;
        for (int e = 0; e < 3; ++e) {
            const uint32_t i = front[e], j = front[(e + 1) % 3];
            if (backId != 0) {
                const glm::uvec3 back = scene.mesh.triangles[backId - 1];
                const auto has = [&](uint32_t v) { return back.x == v || back.y == v || back.z == v; };
                if (has(i) && has(j))
                    continue;
            }
            const glm::dvec2 a(screen(i)), b(screen(j)), edge = b - a, segment = to - from;
            const double t =
                (edge.x * (a.y - from.y) - edge.y * (a.x - from.x)) / (edge.x * segment.y - edge.y * segment.x);
            if (t > 0.0 && t < 1.0 && t < best)
                best = t;
        }
        if (best > 1.0)
            return 0.0;
        return frontIsSelf ? std::max(0.5 - best, 0.0) : std::max(best - 0.5, 0.0);
    }

    glm::dvec4 antialiased(int x, int y) const {
        const glm::dvec4 own = color(x, y);
        glm::dvec4 result = own;
        for (const glm::ivec2 offset : {glm::ivec2(1, 0), glm::ivec2(-1, 0), glm::ivec2(0, 1), glm::ivec2(0, -1)}) {
            const glm::ivec2 other = glm::ivec2(x, y) + offset;
            if (other.x < 0 || other.y < 0 || other.x >= int(scene.width) || other.y >= int(scene.height))
                continue;
            const double w = blendWeight(glm::ivec2(x, y), other);
            if (w > 0.0)
                result += w * (color(other.x, other.y) - own);
        }
        return result;
    }

    // The image loss of the antialiased image against `targets` (rgb and mask per pixel and view).
    double loss(const std::vector<float>& targets, double maskWeight) const {
        const uint32_t pixels = scene.width * scene.height;
        double sum = 0.0;
        for (uint32_t pixel = 0; pixel < pixels; ++pixel) {
            const float* t = &targets[4 * (size_t(view) * pixels + pixel)];
            const glm::dvec4 c = antialiased(int(pixel % scene.width), int(pixel / scene.width));
            const glm::dvec3 d = glm::dvec3(c) - glm::dvec3(t[0], t[1], t[2]);
            sum += t[3] * glm::dot(d, d) + maskWeight * (c.a - t[3]) * (c.a - t[3]);
        }
        return sum / double(pixels);
    }
};

// The training-time image chain on a scene: shade, antialias, image loss, and the backward elements down to vertex
// and texel gradients (added to `vertexGradients` and `textureGradients`, which `clear` zeroes with the loss).
struct ImageChain {
    std::shared_ptr<TensorElement<float>> colors, antialiased, imageGradients, colorGradients, vertexGradients,
        textureGradients, loss;
    ComputeGraphElementPtr clear, shading, antialias, imageLoss, antialiasBackward, shadeBackward;

    ImageChain(VulkanContext& vc, const Scene& scene, const std::shared_ptr<TensorElement<float>>& targets,
               float maskWeight, const ComputeGraphElementPtr& viewSource) {
        const uint32_t pixels = scene.width * scene.height;
        const auto tensor = [&](uint32_t rows) {
            return vc.create<TensorElement<float>>(std::vector<uint32_t>{rows, 4});
        };
        colors = tensor(pixels);
        antialiased = tensor(pixels);
        imageGradients = tensor(pixels);
        colorGradients = tensor(pixels);
        vertexGradients = tensor(uint32_t(scene.mesh.positions.size()));
        textureGradients = tensor(scene.textureWidth * scene.textureHeight);
        loss = vc.create<TensorElement<float>>(std::vector<uint32_t>{1});
        const uint32_t w = scene.width, h = scene.height, tw = scene.textureWidth, th = scene.textureHeight;
        clear = raster::clearBuffers(vc);
        clear->setInput(colorGradients, 0);
        clear->setInput(vertexGradients, 1);
        clear->setInput(textureGradients, 2);
        clear->setInput(loss, 3);
        const auto connect = [&](const ComputeGraphElementPtr& element) {
            scene.connectShading(element);
            if (viewSource)
                element->setInput(viewSource, 5, 0);
        };
        shading = raster::shade(vc, w, h, tw, th);
        connect(shading);
        shading->setInput(colors, 7);
        antialias = raster::antialias(vc, w, h, tw, th);
        connect(antialias);
        antialias->setInput(shading, 7, 7);
        antialias->setInput(antialiased, 8);
        imageLoss = raster::imageLoss(vc, w, h, maskWeight);
        imageLoss->setInput(antialias, 0, 8);
        imageLoss->setInput(targets, 1);
        if (viewSource)
            imageLoss->setInput(viewSource, 2, 0);
        else
            imageLoss->setInput(scene.viewIndex, 2);
        imageLoss->setInput(imageGradients, 3);
        imageLoss->setInput(clear, 4, 3);
        antialiasBackward = raster::antialiasBackward(vc, w, h, tw, th);
        connect(antialiasBackward);
        antialiasBackward->setInput(shading, 7, 7);
        antialiasBackward->setInput(imageLoss, 8, 3);
        antialiasBackward->setInput(clear, 9, 0);
        antialiasBackward->setInput(clear, 10, 1);
        shadeBackward = raster::shadeColorBackward(vc, w, h, tw, th);
        connect(shadeBackward);
        shadeBackward->setInput(antialiasBackward, 7, 9);
        shadeBackward->setInput(antialiasBackward, 8, 10);
        shadeBackward->setInput(clear, 9, 2);
    }
};

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

TEST_F(DifferentiableRasterTest, laplacianRegularizerMatchesFiniteDifferences) {
    raster::TriangleMesh mesh = raster::icosphere(2);
    for (size_t v = 0; v < mesh.positions.size(); ++v)
        mesh.positions[v] *= 1.0f + 0.1f * std::sin(3.7f * float(v)); // bumps to smooth
    const auto adjacency = raster::vertexAdjacency(mesh);
    const uint32_t vertexCount = uint32_t(mesh.positions.size());
    ASSERT_EQ(adjacency.offsets.size(), vertexCount + 1);
    for (uint32_t v = 0; v < vertexCount; ++v) {
        const uint32_t degree = adjacency.offsets[v + 1] - adjacency.offsets[v];
        EXPECT_TRUE(degree == 5 || degree == 6) << "vertex " << v;
        for (uint32_t k = adjacency.offsets[v]; k < adjacency.offsets[v + 1]; ++k) {
            const uint32_t n = adjacency.neighbours[k];
            const auto begin = adjacency.neighbours.begin();
            EXPECT_TRUE(std::binary_search(begin + adjacency.offsets[n], begin + adjacency.offsets[n + 1], v));
        }
    }

    const float weight = 3.0f;
    auto positions = vc->create<TensorElement<float>>(std::vector<uint32_t>{vertexCount, 4});
    auto offsets = std::make_shared<UintBuffer>(*vc, adjacency.offsets.size());
    auto neighbours = std::make_shared<UintBuffer>(*vc, adjacency.neighbours.size());
    auto gradients = vc->create<TensorElement<float>>(std::vector<uint32_t>{vertexCount, 4});
    auto loss = vc->create<TensorElement<float>>(std::vector<uint32_t>{1});
    auto clear = raster::clearBuffers(*vc);
    clear->setInput(gradients, 0);
    clear->setInput(loss, 1);
    auto regularizer = raster::laplacianRegularizer(*vc, vertexCount, weight);
    regularizer->setInput(positions, 0);
    regularizer->setInput(offsets, 1);
    regularizer->setInput(neighbours, 2);
    regularizer->setInput(clear, 3, 0);
    regularizer->setInput(clear, 4, 1);
    ComputeGraph graph(*vc, 1);
    graph.compileFrom(regularizer);
    positions->getDataBuffer(0).memcopyFrom(packPositions(mesh));
    offsets->getBuffer(0).memcopyFrom(adjacency.offsets);
    neighbours->getBuffer(0).memcopyFrom(adjacency.neighbours);
    graph.submitAndWait(vc->getGraphicsQueue(), 0);
    std::vector<float> gpuLoss(1), gpuGradients(4 * vertexCount);
    loss->getDataBuffer(0).memcopyTo(gpuLoss);
    gradients->getDataBuffer(0).memcopyTo(gpuGradients);

    std::vector<glm::dvec3> p(mesh.positions.begin(), mesh.positions.end());
    auto cpuLoss = [&]() {
        double sum = 0.0;
        for (uint32_t v = 0; v < vertexCount; ++v) {
            glm::dvec3 mean(0.0);
            for (uint32_t k = adjacency.offsets[v]; k < adjacency.offsets[v + 1]; ++k)
                mean += p[adjacency.neighbours[k]];
            const glm::dvec3 l = p[v] - mean / double(adjacency.offsets[v + 1] - adjacency.offsets[v]);
            sum += glm::dot(l, l);
        }
        return weight * sum / double(vertexCount);
    };
    const double reference = cpuLoss();
    ASSERT_GT(reference, 1e-4);
    EXPECT_NEAR(gpuLoss[0], reference, 1e-5 * reference);
    for (uint32_t v = 0; v < vertexCount; v += 3) {
        for (int c = 0; c < 3; ++c) {
            const double original = p[v][c], h = 1e-5;
            p[v][c] = original + h;
            const double up = cpuLoss();
            p[v][c] = original - h;
            const double down = cpuLoss();
            p[v][c] = original;
            const double expected = (up - down) / (2 * h);
            EXPECT_NEAR(gpuGradients[4 * v + c], expected, 1e-6 + 1e-3 * std::abs(expected)) << "vertex " << v;
        }
        EXPECT_EQ(gpuGradients[4 * v + 3], 0.0f);
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

TEST_F(DifferentiableRasterTest, shadingMatchesCpu) {
    Scene scene(*vc, 80, 64, 32, 16);
    auto shading = raster::shade(*vc, scene.width, scene.height, scene.textureWidth, scene.textureHeight);
    scene.connectShading(shading);
    auto colors = vc->create<TensorElement<float>>(std::vector<uint32_t>{scene.width * scene.height, 4});
    shading->setInput(colors, 7);
    ComputeGraph graph(*vc, 1);
    graph.compileFrom(shading);
    scene.upload(1);
    graph.submitAndWait(vc->getGraphicsQueue(), 0);
    std::vector<uint32_t> ids(scene.width * scene.height);
    scene.visibility->getBuffer(0).memcopyTo(ids);
    std::vector<float> rgba(4 * ids.size());
    colors->getDataBuffer(0).memcopyTo(rgba);
    const std::vector<double> texture(scene.texture.begin(), scene.texture.end());
    int covered = 0;
    for (uint32_t y = 0; y < scene.height; ++y) {
        for (uint32_t x = 0; x < scene.width; ++x) {
            const uint32_t pixel = y * scene.width + x;
            if (ids[pixel] == 0) {
                for (int c = 0; c < 4; ++c)
                    EXPECT_EQ(rgba[4 * pixel + c], 0.0f);
                continue;
            }
            ++covered;
            const glm::dvec3 expected = cpuShade(scene.mesh, texture, scene.textureWidth, scene.textureHeight,
                                                 scene.cameras[1], scene.width, scene.height, ids[pixel] - 1, x, y);
            for (int c = 0; c < 3; ++c)
                ASSERT_NEAR(rgba[4 * pixel + c], expected[c], 2e-4) << "pixel " << x << ", " << y;
            EXPECT_EQ(rgba[4 * pixel + 3], 1.0f);
        }
    }
    EXPECT_GT(covered, int(scene.width * scene.height) / 10);
}

TEST_F(DifferentiableRasterTest, shadingGradientsMatchFiniteDifferences) {
    Scene scene(*vc, 64, 48, 16, 8);
    const uint32_t pixels = scene.width * scene.height, view = 2;
    const uint32_t vertexCount = uint32_t(scene.mesh.positions.size()),
                   texelCount = scene.textureWidth * scene.textureHeight;
    auto targets = vc->create<TensorElement<float>>(std::vector<uint32_t>{uint32_t(scene.cameras.size()) * pixels, 4});
    auto vertexGradients = vc->create<TensorElement<float>>(std::vector<uint32_t>{vertexCount, 4});
    auto textureGradients = vc->create<TensorElement<float>>(std::vector<uint32_t>{texelCount, 4});
    auto loss = vc->create<TensorElement<float>>(std::vector<uint32_t>{1});
    auto clear = raster::clearBuffers(*vc);
    clear->setInput(vertexGradients, 0);
    clear->setInput(textureGradients, 1);
    clear->setInput(loss, 2);
    auto backward = raster::shadeBackward(*vc, scene.width, scene.height, scene.textureWidth, scene.textureHeight);
    scene.connectShading(backward);
    backward->setInput(targets, 7);
    backward->setInput(clear, 8, 0);
    backward->setInput(clear, 9, 1);
    backward->setInput(clear, 10, 2);
    ComputeGraph graph(*vc, 1);
    graph.compileFrom(backward);
    scene.upload(view);
    // Targets: a pattern, masked out on a diagonal band.
    std::vector<float> target(4 * size_t(scene.cameras.size()) * pixels);
    for (size_t i = 0; i < target.size() / 4; ++i) {
        const uint32_t x = uint32_t(i % scene.width), y = uint32_t(i / scene.width % scene.height);
        target[4 * i] = 0.2f + 0.6f * float((x * 5 + y) % 7) / 7.0f;
        target[4 * i + 1] = 0.4f;
        target[4 * i + 2] = float(y) / float(scene.height);
        target[4 * i + 3] = (x + y) % 9 == 0 ? 0.0f : 1.0f;
    }
    targets->getDataBuffer(0).memcopyFrom(target);
    graph.submitAndWait(vc->getGraphicsQueue(), 0);

    std::vector<uint32_t> ids(pixels);
    scene.visibility->getBuffer(0).memcopyTo(ids);
    std::vector<float> gpuLoss(1), gpuVertex(4 * vertexCount), gpuTexture(4 * texelCount);
    loss->getDataBuffer(0).memcopyTo(gpuLoss);
    vertexGradients->getDataBuffer(0).memcopyTo(gpuVertex);
    textureGradients->getDataBuffer(0).memcopyTo(gpuTexture);

    std::vector<double> texture(scene.texture.begin(), scene.texture.end());
    std::vector<glm::dvec3> positions(scene.mesh.positions.begin(), scene.mesh.positions.end());
    auto cpuLoss = [&]() {
        double sum = 0.0;
        for (uint32_t pixel = 0; pixel < pixels; ++pixel) {
            const float* t = &target[4 * (size_t(view) * pixels + pixel)];
            if (ids[pixel] == 0 || t[3] < 0.5f)
                continue;
            const glm::dvec3 color =
                cpuShade(scene.mesh, texture, scene.textureWidth, scene.textureHeight, scene.cameras[view], scene.width,
                         scene.height, ids[pixel] - 1, pixel % scene.width, pixel / scene.width, &positions);
            const glm::dvec3 d = color - glm::dvec3(t[0], t[1], t[2]);
            sum += glm::dot(d, d);
        }
        return sum / double(pixels);
    };
    const double reference = cpuLoss();
    ASSERT_GT(reference, 1e-4);
    EXPECT_NEAR(gpuLoss[0], reference, 1e-5 * std::max(1.0, reference));

    int texelsChecked = 0, verticesChecked = 0;
    for (uint32_t texel = 0; texel < texelCount; texel += 3) {
        for (int c = 0; c < 3; ++c) {
            const double original = texture[4 * texel + c], h = 1e-3;
            texture[4 * texel + c] = original + h;
            const double up = cpuLoss();
            texture[4 * texel + c] = original - h;
            const double down = cpuLoss();
            texture[4 * texel + c] = original;
            const double expected = (up - down) / (2 * h);
            texelsChecked += expected != 0.0;
            EXPECT_NEAR(gpuTexture[4 * texel + c], expected, 1e-6 + 1e-3 * std::abs(expected)) << "texel " << texel;
        }
    }
    for (uint32_t vertex = 0; vertex < vertexCount; vertex += 5) {
        for (int c = 0; c < 3; ++c) {
            const double original = positions[vertex][c], h = 1e-5;
            positions[vertex][c] = original + h;
            const double up = cpuLoss();
            positions[vertex][c] = original - h;
            const double down = cpuLoss();
            positions[vertex][c] = original;
            const double expected = (up - down) / (2 * h);
            verticesChecked += expected != 0.0;
            EXPECT_NEAR(gpuVertex[4 * vertex + c], expected, 2e-5 + 2e-2 * std::abs(expected))
                << "vertex " << vertex << " axis " << c;
        }
    }
    EXPECT_GT(texelsChecked, 20);
    EXPECT_GT(verticesChecked, 20);
}

TEST_F(DifferentiableRasterTest, textureRecoveryOnFixedGeometry) {
    const uint32_t width = 64, height = 64, textureWidth = 32, textureHeight = 16;
    // Eight training views on two rings and one held-out view between them (the last camera).
    std::vector<glm::mat4> cameras = raster::cameraRing(4, 3.0f, 0.15f, 0.8f, 1.0f);
    const auto upper = raster::cameraRing(4, 3.0f, 0.7f, 0.8f, 1.0f, 0.785f);
    cameras.insert(cameras.end(), upper.begin(), upper.end());
    cameras.push_back(raster::cameraRing(1, 3.0f, 0.4f, 0.8f, 1.0f, 0.4f)[0]);
    const uint32_t trainingViews = 8, heldOut = 8, pixels = width * height, texelCount = textureWidth * textureHeight;

    // Renders every view of a scene with `texture`: colour plus coverage, views one after another.
    auto renderViews = [&](const std::vector<float>& texture) {
        Scene scene(*vc, width, height, textureWidth, textureHeight, cameras);
        scene.texture = texture;
        auto colors = vc->create<TensorElement<float>>(std::vector<uint32_t>{pixels, 4});
        auto render = raster::shade(*vc, width, height, textureWidth, textureHeight);
        scene.connectShading(render);
        render->setInput(colors, 7);
        ComputeGraph graph(*vc, 1);
        graph.compileFrom(render);
        scene.upload(0);
        std::vector<float> images;
        for (uint32_t view = 0; view < uint32_t(cameras.size()); ++view) {
            scene.viewIndex->getBuffer(0).memcopyFrom(std::vector<uint32_t>{view});
            graph.submitAndWait(vc->getGraphicsQueue(), 0);
            std::vector<float> image(4 * pixels);
            colors->getDataBuffer(0).memcopyTo(image);
            images.insert(images.end(), image.begin(), image.end());
        }
        return images;
    };
    const std::vector<float> truth = Scene(*vc, width, height, textureWidth, textureHeight, cameras).texture;
    const std::vector<float> targets = renderViews(truth);

    // The training step: next view, visibility, clear, loss and gradients, Adam on the texels.
    Scene scene(*vc, width, height, textureWidth, textureHeight, cameras);
    auto targetBuffer = vc->create<TensorElement<float>>(std::vector<uint32_t>{uint32_t(cameras.size()) * pixels, 4});
    auto vertexGradients =
        vc->create<TensorElement<float>>(std::vector<uint32_t>{uint32_t(scene.mesh.positions.size()), 4});
    auto textureGradients = vc->create<TensorElement<float>>(std::vector<uint32_t>{texelCount, 4});
    auto loss = vc->create<TensorElement<float>>(std::vector<uint32_t>{1});
    auto moments0 = vc->create<TensorElement<float>>(std::vector<uint32_t>{4 * texelCount});
    auto moments1 = vc->create<TensorElement<float>>(std::vector<uint32_t>{4 * texelCount});
    auto step = std::make_shared<UintBuffer>(*vc, 1);
    auto nextView = raster::cycleView(*vc, trainingViews);
    nextView->setInput(scene.viewIndex, 0);
    scene.pass->setInput(nextView, 3, 0);
    auto clear = raster::clearBuffers(*vc);
    clear->setInput(vertexGradients, 0);
    clear->setInput(textureGradients, 1);
    clear->setInput(loss, 2);
    auto backward = raster::shadeBackward(*vc, width, height, textureWidth, textureHeight);
    scene.connectShading(backward);
    backward->setInput(nextView, 5, 0);
    backward->setInput(targetBuffer, 7);
    backward->setInput(clear, 8, 0);
    backward->setInput(clear, 9, 1);
    backward->setInput(clear, 10, 2);
    auto counter = training::stepCounter(*vc);
    counter->setInput(step, 0);
    auto adam = training::adamStep(*vc, 4 * texelCount, {0.02f, 0.9f, 0.999f, 1e-8f});
    adam->setInput(scene.texels, 0);
    adam->setInput(backward, 1, 9);
    adam->setInput(moments0, 2);
    adam->setInput(moments1, 3);
    adam->setInput(step, 4);
    adam->addDependency(counter);
    ComputeGraph trainGraph(*vc, 1);
    trainGraph.compileFrom(adam);
    for (size_t i = 0; i < scene.texture.size(); ++i)
        scene.texture[i] = float((i * 2654435761u) % 1000) / 1000.0f; // noise
    scene.upload(trainingViews - 1);                                  // the first step advances to view 0
    targetBuffer->getDataBuffer(0).memcopyFrom(targets);
    for (auto* moments : {&moments0, &moments1})
        (*moments)->getDataBuffer(0).memcopyFrom(std::vector<float>(4 * texelCount, 0.0f));
    step->getBuffer(0).memcopyFrom(std::vector<uint32_t>{0});

    auto epochLoss = [&]() {
        double sum = 0.0;
        for (uint32_t v = 0; v < trainingViews; ++v) {
            trainGraph.submitAndWait(vc->getGraphicsQueue(), 0);
            std::vector<float> value(1);
            loss->getDataBuffer(0).memcopyTo(value);
            sum += value[0];
        }
        return sum;
    };
    const double first = epochLoss();
    for (int epoch = 0; epoch < 60; ++epoch)
        epochLoss();
    const double last = epochLoss();
    EXPECT_LT(last, 0.02 * first) << "first " << first << " last " << last;

    // The held-out view, rendered with the trained texture, against its target inside the mesh.
    std::vector<float> trained(4 * texelCount);
    scene.texels->getDataBuffer(0).memcopyTo(trained);
    const std::vector<float> images = renderViews(trained);
    double squaredError = 0.0;
    int count = 0;
    for (uint32_t pixel = 0; pixel < pixels; ++pixel) {
        const size_t i = 4 * (size_t(heldOut) * pixels + pixel);
        if (targets[i + 3] < 0.5f)
            continue;
        for (int c = 0; c < 3; ++c) {
            const double d = images[i + c] - targets[i + c];
            squaredError += d * d;
            ++count;
        }
    }
    ASSERT_GT(count, int(pixels) / 10);
    const double psnr = 10.0 * std::log10(1.0 / (squaredError / count));
    EXPECT_GT(psnr, 30.0) << "held-out PSNR " << psnr << " dB";
    std::cout << "texture recovery: loss " << first << " -> " << last << ", held-out PSNR " << psnr << " dB"
              << std::endl;
}

TEST_F(DifferentiableRasterTest, antialiasMatchesCpu) {
    Scene scene(*vc, 80, 64, 32, 16);
    auto targets = vc->create<TensorElement<float>>(std::vector<uint32_t>{uint32_t(scene.cameras.size()) * 80 * 64, 4});
    ImageChain chain(*vc, scene, targets, 1.0f, nullptr);
    ComputeGraph graph(*vc, 1);
    graph.compileFrom(chain.antialias);
    const uint32_t view = 1;
    scene.upload(view);
    graph.submitAndWait(vc->getGraphicsQueue(), 0);
    std::vector<uint32_t> ids(scene.width * scene.height);
    scene.visibility->getBuffer(0).memcopyTo(ids);
    std::vector<float> shaded(4 * ids.size()), antialiased(4 * ids.size());
    chain.colors->getDataBuffer(0).memcopyTo(shaded);
    chain.antialiased->getDataBuffer(0).memcopyTo(antialiased);
    const std::vector<double> texture(scene.texture.begin(), scene.texture.end());
    const std::vector<glm::dvec3> positions(scene.mesh.positions.begin(), scene.mesh.positions.end());
    const CpuImage cpu{scene, ids, positions, texture, view};
    int changed = 0, interiorUnchanged = 0;
    for (uint32_t y = 0; y < scene.height; ++y) {
        for (uint32_t x = 0; x < scene.width; ++x) {
            const uint32_t pixel = y * scene.width + x;
            const glm::dvec4 expected = cpu.antialiased(int(x), int(y));
            for (int c = 0; c < 4; ++c)
                ASSERT_NEAR(antialiased[4 * pixel + c], expected[c], 2e-4) << "pixel " << x << ", " << y;
            bool differs = false;
            for (int c = 0; c < 4; ++c)
                differs |= antialiased[4 * pixel + c] != shaded[4 * pixel + c];
            changed += differs;
            // A pixel whose neighbours all show its own triangle has no silhouette to blend across.
            bool sameNeighbours = x > 0 && y > 0 && x + 1 < scene.width && y + 1 < scene.height;
            if (sameNeighbours)
                for (const int offset : {1, -1, int(scene.width), -int(scene.width)})
                    sameNeighbours &= ids[pixel + offset] == ids[pixel];
            if (sameNeighbours) {
                EXPECT_FALSE(differs) << "pixel " << x << ", " << y;
                ++interiorUnchanged;
            }
        }
    }
    EXPECT_GT(changed, 20);                  // the outline is antialiased
    EXPECT_LT(changed, int(ids.size()) / 4); // and only the outline
    EXPECT_GT(interiorUnchanged, 100);
}

TEST_F(DifferentiableRasterTest, antialiasGradientsMatchFiniteDifferences) {
    Scene scene(*vc, 64, 48, 16, 8);
    const uint32_t pixels = scene.width * scene.height, view = 2;
    const uint32_t vertexCount = uint32_t(scene.mesh.positions.size()),
                   texelCount = scene.textureWidth * scene.textureHeight;
    const float maskWeight = 2.0f;
    auto targets = vc->create<TensorElement<float>>(std::vector<uint32_t>{uint32_t(scene.cameras.size()) * pixels, 4});
    ImageChain chain(*vc, scene, targets, maskWeight, nullptr);
    ComputeGraph graph(*vc, 1);
    graph.compileFrom(chain.shadeBackward);
    scene.upload(view);
    // Targets: a pattern inside a disc that differs from the mesh's outline.
    std::vector<float> target(4 * size_t(scene.cameras.size()) * pixels);
    for (size_t i = 0; i < target.size() / 4; ++i) {
        const uint32_t x = uint32_t(i % scene.width), y = uint32_t(i / scene.width % scene.height);
        const float dx = float(x) - 0.5f * float(scene.width), dy = float(y) - 0.5f * float(scene.height);
        target[4 * i] = 0.2f + 0.6f * float((x * 5 + y) % 7) / 7.0f;
        target[4 * i + 1] = 0.4f;
        target[4 * i + 2] = float(y) / float(scene.height);
        target[4 * i + 3] = dx * dx + dy * dy < 0.12f * float(pixels) ? 1.0f : 0.0f;
    }
    targets->getDataBuffer(0).memcopyFrom(target);
    graph.submitAndWait(vc->getGraphicsQueue(), 0);

    std::vector<uint32_t> ids(pixels);
    scene.visibility->getBuffer(0).memcopyTo(ids);
    std::vector<float> gpuLoss(1), gpuVertex(4 * vertexCount), gpuTexture(4 * texelCount);
    chain.loss->getDataBuffer(0).memcopyTo(gpuLoss);
    chain.vertexGradients->getDataBuffer(0).memcopyTo(gpuVertex);
    chain.textureGradients->getDataBuffer(0).memcopyTo(gpuTexture);

    std::vector<double> texture(scene.texture.begin(), scene.texture.end());
    std::vector<glm::dvec3> positions(scene.mesh.positions.begin(), scene.mesh.positions.end());
    const CpuImage cpu{scene, ids, positions, texture, view};
    const double reference = cpu.loss(target, maskWeight);
    ASSERT_GT(reference, 1e-3);
    EXPECT_NEAR(gpuLoss[0], reference, 1e-5 * std::max(1.0, reference));

    // Vertices on the outline: those of triangles next to background.
    std::vector<bool> silhouette(vertexCount, false);
    for (uint32_t pixel = 0; pixel < pixels; ++pixel) {
        const uint32_t x = pixel % scene.width;
        if (ids[pixel] != 0 && ((x > 0 && ids[pixel - 1] == 0) || (x + 1 < scene.width && ids[pixel + 1] == 0)))
            for (int c = 0; c < 3; ++c)
                silhouette[scene.mesh.triangles[ids[pixel] - 1][c]] = true;
    }
    int texelsChecked = 0, verticesChecked = 0, silhouetteChecked = 0;
    for (uint32_t texel = 0; texel < texelCount; texel += 5) {
        for (int c = 0; c < 3; ++c) {
            const double original = texture[4 * texel + c], h = 1e-3;
            texture[4 * texel + c] = original + h;
            const double up = cpu.loss(target, maskWeight);
            texture[4 * texel + c] = original - h;
            const double down = cpu.loss(target, maskWeight);
            texture[4 * texel + c] = original;
            const double expected = (up - down) / (2 * h);
            texelsChecked += expected != 0.0;
            EXPECT_NEAR(gpuTexture[4 * texel + c], expected, 1e-6 + 1e-3 * std::abs(expected)) << "texel " << texel;
        }
    }
    for (uint32_t vertex = 0; vertex < vertexCount; ++vertex) {
        if (!silhouette[vertex] && vertex % 7 != 0)
            continue;
        for (int c = 0; c < 3; ++c) {
            const double original = positions[vertex][c], h = 1e-5;
            positions[vertex][c] = original + h;
            const double up = cpu.loss(target, maskWeight);
            positions[vertex][c] = original - h;
            const double down = cpu.loss(target, maskWeight);
            positions[vertex][c] = original;
            const double expected = (up - down) / (2 * h);
            verticesChecked += expected != 0.0;
            silhouetteChecked += silhouette[vertex] && expected != 0.0;
            EXPECT_NEAR(gpuVertex[4 * vertex + c], expected, 2e-5 + 2e-2 * std::abs(expected))
                << "vertex " << vertex << " axis " << c << (silhouette[vertex] ? " (silhouette)" : "");
        }
    }
    EXPECT_GT(texelsChecked, 10);
    EXPECT_GT(verticesChecked, 20);
    EXPECT_GT(silhouetteChecked, 20);
}

TEST_F(DifferentiableRasterTest, silhouetteFittingFromMasks) {
    const uint32_t width = 96, height = 96, textureWidth = 8, textureHeight = 4;
    std::vector<glm::mat4> cameras = raster::cameraRing(4, 4.0f, 0.2f, 0.8f, 1.0f);
    const auto upper = raster::cameraRing(4, 4.0f, 0.9f, 0.8f, 1.0f, 0.785f);
    cameras.insert(cameras.end(), upper.begin(), upper.end());
    cameras.push_back(raster::cameraRing(1, 4.0f, 0.5f, 0.8f, 1.0f, 0.4f)[0]);
    const uint32_t trainingViews = 8, heldOut = 8, pixels = width * height;
    raster::TriangleMesh ellipsoid = raster::icosphere(3);
    for (auto& p : ellipsoid.positions)
        p *= glm::vec3(1.25f, 0.7f, 0.9f);
    const std::vector<float> gray(4 * textureWidth * textureHeight, 0.5f);

    // The coverage (1 inside, 0 outside) of `mesh` in every view, views one after another.
    auto masks = [&](const raster::TriangleMesh& mesh) {
        Scene scene(*vc, width, height, textureWidth, textureHeight, cameras, mesh);
        ComputeGraph graph(*vc, 1);
        graph.compileFrom(scene.pass);
        scene.positions->getDataBuffer(0).memcopyFrom(packPositions(mesh));
        scene.triangles->getBuffer(0).memcopyFrom(packTriangles(mesh));
        scene.cameraBuffer->getDataBuffer(0).memcopyFrom(packCameras(cameras));
        std::vector<float> result;
        for (uint32_t view = 0; view < uint32_t(cameras.size()); ++view) {
            scene.viewIndex->getBuffer(0).memcopyFrom(std::vector<uint32_t>{view});
            graph.submitAndWait(vc->getGraphicsQueue(), 0);
            std::vector<uint32_t> ids(pixels);
            scene.visibility->getBuffer(0).memcopyTo(ids);
            for (const uint32_t id : ids)
                result.push_back(id != 0 ? 1.0f : 0.0f);
        }
        return result;
    };
    auto heldOutIou = [&](const std::vector<float>& a, const std::vector<float>& b) {
        int both = 0, either = 0;
        for (uint32_t pixel = 0; pixel < pixels; ++pixel) {
            const size_t i = size_t(heldOut) * pixels + pixel;
            both += a[i] > 0.5f && b[i] > 0.5f;
            either += a[i] > 0.5f || b[i] > 0.5f;
        }
        return double(both) / double(either);
    };
    const std::vector<float> truth = masks(ellipsoid);
    // Targets: the antialiased renderings of the ellipsoid, coverage as the mask (a soft matte).
    std::vector<float> targets;
    {
        Scene target(*vc, width, height, textureWidth, textureHeight, cameras, ellipsoid);
        target.texture = gray;
        auto unused = vc->create<TensorElement<float>>(std::vector<uint32_t>{pixels, 4});
        ImageChain render(*vc, target, unused, 1.0f, nullptr);
        ComputeGraph graph(*vc, 1);
        graph.compileFrom(render.antialias);
        target.upload(0);
        for (uint32_t view = 0; view < uint32_t(cameras.size()); ++view) {
            target.viewIndex->getBuffer(0).memcopyFrom(std::vector<uint32_t>{view});
            graph.submitAndWait(vc->getGraphicsQueue(), 0);
            std::vector<float> image(4 * pixels);
            render.antialiased->getDataBuffer(0).memcopyTo(image);
            targets.insert(targets.end(), image.begin(), image.end());
        }
    }

    // The training step: next view, visibility, the image chain, Adam on the vertex positions.
    Scene scene(*vc, width, height, textureWidth, textureHeight, cameras, raster::icosphere(3));
    const uint32_t vertexCount = uint32_t(scene.mesh.positions.size());
    auto targetBuffer = vc->create<TensorElement<float>>(std::vector<uint32_t>{uint32_t(cameras.size()) * pixels, 4});
    auto moments0 = vc->create<TensorElement<float>>(std::vector<uint32_t>{4 * vertexCount});
    auto moments1 = vc->create<TensorElement<float>>(std::vector<uint32_t>{4 * vertexCount});
    auto step = std::make_shared<UintBuffer>(*vc, 1);
    auto nextView = raster::cycleView(*vc, trainingViews);
    nextView->setInput(scene.viewIndex, 0);
    scene.pass->setInput(nextView, 3, 0);
    ImageChain chain(*vc, scene, targetBuffer, 1.0f, nextView);
    auto counter = training::stepCounter(*vc);
    counter->setInput(step, 0);
    auto adam = training::adamStep(*vc, 4 * vertexCount, {0.005f, 0.9f, 0.999f, 1e-8f});
    adam->setInput(scene.positions, 0);
    adam->setInput(chain.shadeBackward, 1, 8);
    adam->setInput(moments0, 2);
    adam->setInput(moments1, 3);
    adam->setInput(step, 4);
    adam->addDependency(counter);
    ComputeGraph trainGraph(*vc, 1);
    trainGraph.compileFrom(adam);
    scene.texture = gray;
    scene.upload(trainingViews - 1); // the first step advances to view 0
    targetBuffer->getDataBuffer(0).memcopyFrom(targets);
    for (auto* moments : {&moments0, &moments1})
        (*moments)->getDataBuffer(0).memcopyFrom(std::vector<float>(4 * vertexCount, 0.0f));
    step->getBuffer(0).memcopyFrom(std::vector<uint32_t>{0});

    auto epochLoss = [&]() {
        double sum = 0.0;
        for (uint32_t v = 0; v < trainingViews; ++v) {
            trainGraph.submitAndWait(vc->getGraphicsQueue(), 0);
            std::vector<float> value(1);
            chain.loss->getDataBuffer(0).memcopyTo(value);
            sum += value[0];
        }
        return sum;
    };
    const double startIou = heldOutIou(masks(raster::icosphere(3)), truth);
    const double first = epochLoss();
    for (int epoch = 0; epoch < 60; ++epoch)
        epochLoss();
    const double last = epochLoss();

    std::vector<float> trained(4 * vertexCount);
    scene.positions->getDataBuffer(0).memcopyTo(trained);
    raster::TriangleMesh fitted = scene.mesh;
    for (uint32_t v = 0; v < vertexCount; ++v)
        fitted.positions[v] = glm::vec3(trained[4 * v], trained[4 * v + 1], trained[4 * v + 2]);
    const double iou = heldOutIou(masks(fitted), truth);
    std::cout << "silhouette fitting: loss " << first << " -> " << last << ", held-out IoU " << startIou << " -> "
              << iou << std::endl;
    EXPECT_LT(last, 0.2 * first);
    EXPECT_GT(iou, 0.95);
}
