// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

/**
 * TESTS:
 * - projectionMatchesCpu: the projected means, depths, conics, opacities and radii of Gaussians seen by a ring camera
 *   equal a CPU evaluation, and Gaussians behind the camera are culled
 * - renderMatchesCpu: the blended colours and coverages of a few dozen Gaussians over an image of several tiles
 *   (partial ones at the edges) equal CPU front-to-back blending in depth order with the same tile, alpha and
 *   transmittance rules, and the emitted instance count is the sum of the tiles the Gaussians touch
 * - gradientsMatchFiniteDifferences: the image loss of the render (colour in the mask, coverage everywhere) and its
 *   gradients with respect to every parameter of every Gaussian (position, log scale, rotation, colour, opacity) equal
 *   the CPU loss and its central differences, wherever the loss is smooth (central differences at two step sizes
 *   agree to 0.5 %); at most 10 % of the components are skipped as not smooth
 * - transformedSplatsRenderAlike: Gaussians moved by a similarity transform (rotation, scale, translation) and seen
 *   by the camera moved alike render the same image (up to float rounding), and writeSplatPly() writes a header
 *   for every Gaussian and 17 floats each
 * - trainingFitsRenderedViews: SplatTraining, from Gaussians on a sphere, fits 24 rendered views of the textured
 *   synthetic pumpkin with densification, one submit per step; three held-out views reach a silhouette IoU above
 *   0.97 and a PSNR inside the object above 29 dB
 **/

#include <gtest/gtest.h>

#include <algorithm>
#include <iostream>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <string>
#include <memory>
#include <numeric>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "klartraum/computegraph/bufferelement.hpp"
#include "klartraum/computegraph/computegraph.hpp"
#include "klartraum/computegraph/tensorelement.hpp"
#include "klartraum/headless_frontend.hpp"
#include "klartraum/raster/differentiable_shading.hpp"
#include "klartraum/raster/mesh.hpp"
#include "klartraum/raster/mesh_training.hpp"
#include "klartraum/splatting/splat_training.hpp"

using namespace klartraum;

namespace {

using UintBuffer = BufferElement<VulkanBuffer<uint32_t>>;

class SplatTrainingTest : public ::testing::Test {
protected:
    void SetUp() override {
        frontend = std::make_unique<HeadlessFrontend>();
        vc = &frontend->getKlartraumEngine().getVulkanContext();
    }

    std::unique_ptr<HeadlessFrontend> frontend;
    VulkanContext* vc = nullptr;
};

// Gaussians on the CPU, in the layout of the parameter tensors.
struct Gaussians {
    std::vector<float> means, logScales, rotations, colors, opacities;
    size_t size() const { return opacities.size(); }
};

// `count` Gaussians in a ball of radius `spread` around the origin with varied shapes, orientations and colours.
Gaussians randomGaussians(uint32_t count, float spread, float scale) {
    Gaussians g;
    uint32_t state = 12345u;
    const auto next = [&]() {
        state = state * 1664525u + 1013904223u;
        return float(state >> 8) / float(1u << 24);
    };
    for (uint32_t i = 0; i < count; ++i) {
        glm::vec3 p;
        do
            p = glm::vec3(next(), next(), next()) * 2.0f - 1.0f;
        while (glm::dot(p, p) > 1.0f);
        p *= spread;
        g.means.insert(g.means.end(), {p.x, p.y, p.z, 0.0f});
        g.logScales.insert(g.logScales.end(), {std::log(scale * (0.4f + next())), std::log(scale * (0.4f + next())),
                                               std::log(scale * (0.4f + next())), 0.0f});
        g.rotations.insert(g.rotations.end(), {next() - 0.5f, next() - 0.5f, next() - 0.5f, 0.6f + next()});
        g.colors.insert(g.colors.end(), {next(), next(), next(), 0.0f});
        g.opacities.push_back(4.0f * next() - 1.0f);
    }
    return g;
}

// A projected Gaussian, in double precision as splat_common.slang computes it.
struct Projected {
    glm::dvec2 mean;
    double depth = 0.0, radius = 0.0, opacity = 0.0;
    glm::dvec3 conic, covariance, color;
};

Projected cpuProject(const Gaussians& g, size_t i, const glm::mat4& camera, glm::dvec2 size) {
    const glm::dmat4 m(camera);
    const glm::dvec3 p(g.means[4 * i], g.means[4 * i + 1], g.means[4 * i + 2]);
    const glm::dvec4 clip = m * glm::dvec4(p, 1.0);
    Projected r;
    r.mean = (glm::dvec2(clip) / clip.w * 0.5 + 0.5) * size;
    r.depth = clip.w;
    // Rows of m: glm is column-major, m[column][row].
    const auto row = [&](int k) { return glm::dvec3(m[0][k], m[1][k], m[2][k]); };
    const glm::dvec3 j0 = 0.5 * size.x * (row(0) * clip.w - clip.x * row(3)) / (clip.w * clip.w);
    const glm::dvec3 j1 = 0.5 * size.y * (row(1) * clip.w - clip.y * row(3)) / (clip.w * clip.w);
    glm::dvec4 q(g.rotations[4 * i], g.rotations[4 * i + 1], g.rotations[4 * i + 2], g.rotations[4 * i + 3]);
    q /= glm::length(q);
    const double x = q.x, y = q.y, z = q.z, w = q.w;
    const glm::dvec3 s(std::exp(double(g.logScales[4 * i])), std::exp(double(g.logScales[4 * i + 1])),
                       std::exp(double(g.logScales[4 * i + 2])));
    const glm::dvec3 m0 = s.x * glm::dvec3(1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y));
    const glm::dvec3 m1 = s.y * glm::dvec3(2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x));
    const glm::dvec3 m2 = s.z * glm::dvec3(2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y));
    const glm::dvec3 a(glm::dot(j0, m0), glm::dot(j0, m1), glm::dot(j0, m2));
    const glm::dvec3 b(glm::dot(j1, m0), glm::dot(j1, m1), glm::dot(j1, m2));
    r.covariance = glm::dvec3(glm::dot(a, a) + 0.3, glm::dot(a, b), glm::dot(b, b) + 0.3);
    const double determinant = r.covariance.x * r.covariance.z - r.covariance.y * r.covariance.y;
    r.conic = glm::dvec3(r.covariance.z, -r.covariance.y, r.covariance.x) / determinant;
    const double middle = 0.5 * (r.covariance.x + r.covariance.z);
    r.radius = std::ceil(3.0 * std::sqrt(middle + std::sqrt(std::max(0.1, middle * middle - determinant))));
    r.opacity = 1.0 / (1.0 + std::exp(-double(g.opacities[i])));
    r.color = glm::dvec3(g.colors[4 * i], g.colors[4 * i + 1], g.colors[4 * i + 2]);
    if (clip.w <= 0.05)
        r.radius = 0.0;
    return r;
}

// The tiles [low, high] (inclusive) a projected Gaussian touches, as splat_project.comp.slang: none if culled.
bool cpuTiles(const Projected& r, glm::dvec2 size, glm::ivec2 tiles, glm::ivec2& low, glm::ivec2& high) {
    if (r.radius <= 0.0 || r.mean.x + r.radius < 0 || r.mean.y + r.radius < 0 || r.mean.x - r.radius >= size.x ||
        r.mean.y - r.radius >= size.y)
        return false;
    low = glm::clamp(glm::ivec2(glm::floor((r.mean - r.radius) / 16.0)), glm::ivec2(0), tiles - 1);
    high = glm::clamp(glm::ivec2(glm::floor((r.mean + r.radius) / 16.0)), glm::ivec2(0), tiles - 1);
    return true;
}

// The CPU image of `g`: per pixel premultiplied rgb and coverage, blending front to back in depth order with the
// tile, alpha and transmittance rules of splat_render.comp.slang.
std::vector<glm::dvec4> cpuRender(const Gaussians& g, const glm::mat4& camera, uint32_t width, uint32_t height) {
    const glm::dvec2 size(width, height);
    const glm::ivec2 tiles((width + 15) / 16, (height + 15) / 16);
    std::vector<Projected> projected;
    std::vector<glm::ivec4> rects;
    for (size_t i = 0; i < g.size(); ++i) {
        projected.push_back(cpuProject(g, i, camera, size));
        glm::ivec2 low(0), high(-1);
        cpuTiles(projected.back(), size, tiles, low, high);
        rects.emplace_back(low, high);
    }
    std::vector<size_t> order(g.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) { return projected[a].depth < projected[b].depth; });
    std::vector<glm::dvec4> image;
    for (uint32_t y = 0; y < height; ++y) {
        for (uint32_t x = 0; x < width; ++x) {
            const glm::ivec2 tile(x / 16, y / 16);
            const glm::dvec2 centre(x + 0.5, y + 0.5);
            glm::dvec3 color(0.0);
            double transmittance = 1.0;
            for (const size_t i : order) {
                const glm::ivec4 rect = rects[i];
                if (tile.x < rect.x || tile.y < rect.y || tile.x > rect.z || tile.y > rect.w)
                    continue;
                const Projected& r = projected[i];
                const glm::dvec2 d = centre - r.mean;
                const double power = -0.5 * (r.conic.x * d.x * d.x + r.conic.z * d.y * d.y) - r.conic.y * d.x * d.y;
                const double alpha = std::min(r.opacity * std::exp(std::min(power, 0.0)), 0.99);
                if (alpha < 1.0 / 255.0)
                    continue;
                if (transmittance * (1.0 - alpha) < 1e-4)
                    break;
                color += r.color * alpha * transmittance;
                transmittance *= 1.0 - alpha;
            }
            image.emplace_back(color, 1.0 - transmittance);
        }
    }
    return image;
}

// Uploads `g` into freshly created parameter tensors (after the graph using them is compiled).
void upload(const splatting::SplatParameters& p, const Gaussians& g) {
    p.means->getDataBuffer(0).memcopyFrom(g.means);
    p.logScales->getDataBuffer(0).memcopyFrom(g.logScales);
    p.rotations->getDataBuffer(0).memcopyFrom(g.rotations);
    p.colors->getDataBuffer(0).memcopyFrom(g.colors);
    p.opacities->getDataBuffer(0).memcopyFrom(g.opacities);
}

} // namespace

TEST_F(SplatTrainingTest, projectionMatchesCpu) {
    Gaussians g = randomGaussians(50, 0.8f, 0.05f);
    // One Gaussian behind the camera.
    g.means[0] = 0.0f, g.means[1] = 0.0f, g.means[2] = 5.0f;
    const glm::mat4 camera = raster::cameraRing(1, 3.0f, 0.3f, 0.8f, 4.0f / 3.0f, 1.57f)[0];
    splatting::SplatRasterOptions options;
    options.width = 120;
    options.height = 90;
    options.capacity = 4096;
    const auto parameters = splatting::createSplatParameters(*vc, uint32_t(g.size()));
    auto cameras = vc->create<TensorElement<float>>(std::vector<uint32_t>{1, 16});
    auto viewIndex = std::make_shared<UintBuffer>(*vc, 1);
    splatting::SplatForward forward(*vc, parameters, cameras, viewIndex, -1, options);
    ComputeGraph graph(*vc, 1);
    graph.compileFrom(forward.project);
    upload(parameters, g);
    cameras->getDataBuffer(0).memcopyFrom(std::vector<float>(glm::value_ptr(camera), glm::value_ptr(camera) + 16));
    viewIndex->getBuffer(0).memcopyFrom(std::vector<uint32_t>{0});
    graph.submitAndWait(vc->getGraphicsQueue(), 0);
    std::vector<float> projected(12 * g.size());
    forward.projected->getDataBuffer(0).memcopyTo(projected);

    const glm::dvec2 size(options.width, options.height);
    int visible = 0;
    for (size_t i = 0; i < g.size(); ++i) {
        const Projected r = cpuProject(g, i, camera, size);
        glm::ivec2 low, high;
        const bool touches = cpuTiles(r, size, glm::ivec2(forward.tilesX, forward.tilesY), low, high);
        const float* gpu = &projected[12 * i];
        if (!touches) {
            EXPECT_EQ(gpu[3], 0.0f) << "Gaussian " << i << " should be culled";
            continue;
        }
        ++visible;
        EXPECT_NEAR(gpu[0], r.mean.x, 1e-3) << i;
        EXPECT_NEAR(gpu[1], r.mean.y, 1e-3) << i;
        EXPECT_NEAR(gpu[2], r.depth, 1e-5) << i;
        EXPECT_EQ(gpu[3], float(r.radius)) << i;
        for (int c = 0; c < 3; ++c)
            EXPECT_NEAR(gpu[4 + c], r.conic[c], 1e-4 * std::max(1.0, std::abs(r.conic[c]))) << i;
        EXPECT_NEAR(gpu[7], r.opacity, 1e-6) << i;
        for (int c = 0; c < 3; ++c)
            EXPECT_EQ(gpu[8 + c], float(r.color[c])) << i;
    }
    EXPECT_EQ(projected[3], 0.0f); // behind the camera
    EXPECT_GT(visible, 40);
}

TEST_F(SplatTrainingTest, renderMatchesCpu) {
    const Gaussians g = randomGaussians(60, 0.7f, 0.15f);
    const glm::mat4 camera = raster::cameraRing(1, 3.0f, 0.3f, 0.8f, 72.0f / 56.0f, 0.4f)[0];
    splatting::SplatRasterOptions options;
    options.width = 72;
    options.height = 56;
    options.capacity = 8192;
    const auto parameters = splatting::createSplatParameters(*vc, uint32_t(g.size()));
    auto cameras = vc->create<TensorElement<float>>(std::vector<uint32_t>{1, 16});
    auto viewIndex = std::make_shared<UintBuffer>(*vc, 1);
    splatting::SplatForward forward(*vc, parameters, cameras, viewIndex, -1, options);
    ComputeGraph graph(*vc, 1);
    graph.compileFrom(forward.render);
    upload(parameters, g);
    cameras->getDataBuffer(0).memcopyFrom(std::vector<float>(glm::value_ptr(camera), glm::value_ptr(camera) + 16));
    viewIndex->getBuffer(0).memcopyFrom(std::vector<uint32_t>{0});
    graph.submitAndWait(vc->getGraphicsQueue(), 0);
    const uint32_t pixels = options.width * options.height;
    std::vector<float> colors(4 * pixels);
    forward.colors->getDataBuffer(0).memcopyTo(colors);
    std::vector<uint32_t> counters(1);
    forward.counters->getBuffer(0).memcopyTo(counters);

    const glm::dvec2 size(options.width, options.height);
    const glm::ivec2 tiles(forward.tilesX, forward.tilesY);
    std::vector<Projected> projected;
    std::vector<glm::ivec4> rects;
    uint32_t instances = 0;
    for (size_t i = 0; i < g.size(); ++i) {
        projected.push_back(cpuProject(g, i, camera, size));
        glm::ivec2 low(0), high(-1);
        if (cpuTiles(projected.back(), size, tiles, low, high))
            instances += uint32_t((high.x - low.x + 1) * (high.y - low.y + 1));
        rects.emplace_back(low, high);
    }
    EXPECT_EQ(counters[0], instances);
    const std::vector<glm::dvec4> expected = cpuRender(g, camera, options.width, options.height);
    double covered = 0.0;
    for (uint32_t pixel = 0; pixel < pixels; ++pixel) {
        for (int c = 0; c < 4; ++c)
            ASSERT_NEAR(colors[4 * pixel + c], expected[pixel][c], 2e-4)
                << "pixel " << pixel % options.width << ", " << pixel / options.width;
        covered += expected[pixel].a;
    }
    EXPECT_GT(covered, 0.2 * pixels); // the Gaussians fill a good part of the image
}

TEST_F(SplatTrainingTest, gradientsMatchFiniteDifferences) {
    Gaussians g = randomGaussians(30, 0.6f, 0.15f);
    const glm::mat4 camera = raster::cameraRing(1, 3.0f, 0.3f, 0.8f, 40.0f / 32.0f, 0.9f)[0];
    splatting::SplatRasterOptions options;
    options.width = 40;
    options.height = 32;
    options.capacity = 4096;
    const uint32_t pixels = options.width * options.height;
    const float maskWeight = 1.5f;
    const auto parameters = splatting::createSplatParameters(*vc, uint32_t(g.size()));
    auto cameras = vc->create<TensorElement<float>>(std::vector<uint32_t>{1, 16});
    auto viewIndex = std::make_shared<UintBuffer>(*vc, 1);
    auto targets = vc->create<TensorElement<float>>(std::vector<uint32_t>{pixels, 4});
    auto imageGradients = vc->create<TensorElement<float>>(std::vector<uint32_t>{pixels, 4});
    auto loss = vc->create<TensorElement<float>>(std::vector<uint32_t>{1});
    splatting::SplatForward forward(*vc, parameters, cameras, viewIndex, -1, options);
    auto clearLoss = raster::clearBuffers(*vc);
    clearLoss->setInput(loss, 0);
    auto imageLoss = raster::imageLoss(*vc, options.width, options.height, maskWeight);
    imageLoss->setInput(forward.render, 0, 3);
    imageLoss->setInput(targets, 1);
    imageLoss->setInput(viewIndex, 2);
    imageLoss->setInput(imageGradients, 3);
    imageLoss->setInput(clearLoss, 4, 0);
    splatting::SplatBackward backward(*vc, parameters, cameras, viewIndex, -1, forward, imageLoss, 3);
    ComputeGraph graph(*vc, 1);
    graph.compileFrom(backward.projectBackward);
    upload(parameters, g);
    cameras->getDataBuffer(0).memcopyFrom(std::vector<float>(glm::value_ptr(camera), glm::value_ptr(camera) + 16));
    viewIndex->getBuffer(0).memcopyFrom(std::vector<uint32_t>{0});
    // Targets: a pattern inside a disc.
    std::vector<float> target(4 * pixels);
    for (uint32_t i = 0; i < pixels; ++i) {
        const float x = float(i % options.width) - 20.0f, y = float(i / options.width) - 16.0f;
        const float mask = x * x + y * y < 150.0f ? 1.0f : 0.0f;
        target[4 * i] = mask * (0.3f + 0.5f * float(i % 7) / 7.0f);
        target[4 * i + 1] = mask * 0.4f;
        target[4 * i + 2] = mask * float(i / options.width) / 32.0f;
        target[4 * i + 3] = mask;
    }
    targets->getDataBuffer(0).memcopyFrom(target);
    backward.statistics->getDataBuffer(0).memcopyFrom(std::vector<float>(2 * g.size(), 0.0f));
    graph.submitAndWait(vc->getGraphicsQueue(), 0);
    std::vector<float> gpuLoss(1);
    loss->getDataBuffer(0).memcopyTo(gpuLoss);
    const auto read = [](const std::shared_ptr<TensorElement<float>>& tensor) {
        std::vector<float> values(tensor->getDataElementCount());
        tensor->getDataBuffer(0).memcopyTo(values);
        return values;
    };
    const std::vector<std::vector<float>> gpu = {read(backward.gradients.means), read(backward.gradients.logScales),
                                                 read(backward.gradients.rotations), read(backward.gradients.colors),
                                                 read(backward.gradients.opacities)};

    const auto cpuLoss = [&](const Gaussians& gaussians) {
        const std::vector<glm::dvec4> image = cpuRender(gaussians, camera, options.width, options.height);
        double sum = 0.0;
        for (uint32_t i = 0; i < pixels; ++i) {
            const float* t = &target[4 * i];
            const glm::dvec3 d = glm::dvec3(image[i]) - glm::dvec3(t[0], t[1], t[2]);
            sum += t[3] * glm::dot(d, d) + maskWeight * (image[i].a - t[3]) * (image[i].a - t[3]);
        }
        return sum / pixels;
    };
    EXPECT_NEAR(gpuLoss[0], cpuLoss(g), 1e-5);

    // Parameter groups: the vector, its components per Gaussian, the step.
    struct Group {
        std::vector<float> Gaussians::* values;
        int stride, components;
        double step;
        const char* name;
    };
    const Group groups[] = {{&Gaussians::means, 4, 3, 1e-4, "mean"},
                            {&Gaussians::logScales, 4, 3, 1e-4, "log scale"},
                            {&Gaussians::rotations, 4, 4, 1e-4, "rotation"},
                            {&Gaussians::colors, 4, 3, 1e-3, "colour"},
                            {&Gaussians::opacities, 1, 1, 1e-3, "opacity"}};
    int checked = 0, skipped = 0, nonzero = 0;
    for (size_t group = 0; group < 5; ++group) {
        const Group& p = groups[group];
        for (size_t i = 0; i < g.size(); ++i) {
            for (int c = 0; c < p.components; ++c) {
                float& value = (g.*p.values)[p.stride * i + c];
                const float original = value;
                const auto difference = [&](double h) {
                    value = float(original + h);
                    const double up = cpuLoss(g);
                    value = float(original - h);
                    const double down = cpuLoss(g);
                    value = original;
                    return (up - down) / (double(float(original + h)) - double(float(original - h)));
                };
                const double coarse = difference(p.step), fine = difference(0.5 * p.step);
                // Near a kink of the loss (a splat at an alpha or tile threshold) the differences depend on the step.
                if (std::abs(coarse - fine) > 1e-6 + 0.005 * std::abs(fine)) {
                    ++skipped;
                    continue;
                }
                ++checked;
                nonzero += fine != 0.0;
                EXPECT_NEAR(gpu[group][p.stride * i + c], fine, 2e-6 + 0.02 * std::abs(fine))
                    << p.name << " " << c << " of Gaussian " << i;
            }
        }
    }
    std::cout << "gradients: " << checked << " components checked (" << nonzero << " non-zero), " << skipped
              << " skipped as not smooth" << std::endl;
    EXPECT_LT(skipped, (checked + skipped) / 10);
    EXPECT_GT(nonzero, 200);
}

TEST_F(SplatTrainingTest, transformedSplatsRenderAlike) {
    const Gaussians g = randomGaussians(80, 0.6f, 0.12f);
    const splatting::SplatSet splats{g.means, g.logScales, g.rotations, g.colors, g.opacities};
    const glm::mat4 camera = raster::cameraRing(1, 3.0f, 0.3f, 0.8f, 1.0f, 0.7f)[0];
    const glm::mat4 similarity = glm::translate(glm::mat4(1.0f), glm::vec3(0.3f, -1.2f, 2.0f)) *
                                 glm::rotate(glm::mat4(1.0f), 1.1f, glm::normalize(glm::vec3(1.0f, 2.0f, -0.5f))) *
                                 glm::scale(glm::mat4(1.0f), glm::vec3(0.37f));
    splatting::SplatRasterOptions options;
    options.width = options.height = 48;
    options.capacity = 8192;
    options.depthScale = 1024.0f;
    options.nearDepth = 0.01f;
    const std::vector<float> original = splatting::renderSplats(*vc, splats, {camera}, options);
    // Depths shrink with the scale: quantize them as finely.
    options.depthScale = 1024.0f / 0.37f;
    options.nearDepth = 0.01f * 0.37f;
    const std::vector<float> moved = splatting::renderSplats(*vc, splatting::transformSplats(splats, similarity),
                                                             {camera * glm::inverse(similarity)}, options);
    double coverage = 0.0;
    double difference = 0.0;
    for (size_t i = 0; i < original.size(); ++i) {
        // Single values differ by float rounding of the moved parameters; on average they agree closely.
        ASSERT_NEAR(moved[i], original[i], 5e-3) << "value " << i;
        difference += std::abs(moved[i] - original[i]);
        coverage += i % 4 == 3 ? original[i] : 0.0;
    }
    EXPECT_LT(difference / original.size(), 1e-4);
    EXPECT_GT(coverage, 0.2 * 48 * 48);

    std::filesystem::create_directories("build/TestingOutput");
    const std::filesystem::path path = "build/TestingOutput/splats.ply";
    splatting::writeSplatPly(splats, path);
    std::ifstream in(path, std::ios::binary);
    std::string header, line;
    while (std::getline(in, line) && line != "end_header")
        header += line + "\n";
    EXPECT_NE(header.find("element vertex 80\n"), std::string::npos);
    EXPECT_NE(header.find("property float rot_3\n"), std::string::npos);
    const auto bodyStart = in.tellg();
    in.seekg(0, std::ios::end);
    EXPECT_EQ(size_t(in.tellg() - bodyStart), 80 * 17 * sizeof(float));
}

TEST_F(SplatTrainingTest, trainingFitsRenderedViews) {
    splatting::SplatTrainingOptions options;
    options.raster.width = options.raster.height = 128;
    options.raster.capacity = 1u << 20;
    options.steps = 3000;
    options.densifyUntil = 2000;
    std::vector<glm::mat4> cameras, heldOut;
    for (const auto& [elevation, start] : {std::pair{-0.25f, 0.0f}, {0.3f, 0.39f}, {0.85f, 0.0f}}) {
        const auto ring = raster::cameraRing(8, 3.5f, elevation, 0.8f, 1.0f, start);
        cameras.insert(cameras.end(), ring.begin(), ring.end());
    }
    for (const float angle : {0.2f, 2.3f, 4.4f})
        heldOut.push_back(raster::cameraRing(1, 3.5f, 0.05f + 0.1f * angle, 0.8f, 1.0f, angle)[0]);
    const raster::TriangleMesh pumpkin = raster::syntheticPumpkin(4);
    const std::vector<float> texture = raster::syntheticPumpkinTexture(64, 32);
    const std::vector<float> targets = raster::renderImages(*vc, pumpkin, texture, 64, 32, cameras, 128, 128);
    const std::vector<float> heldOutTargets = raster::renderImages(*vc, pumpkin, texture, 64, 32, heldOut, 128, 128);

    std::vector<glm::vec3> points;
    for (const auto& p : raster::icosphere(3).positions)
        points.push_back(0.8f * p);
    splatting::SplatTraining training(*vc, splatting::splatsAt(points, 0.06f), cameras, targets, options);
    double first = 0.0, last = 0.0;
    for (uint32_t step = 0; step < options.steps; ++step) {
        training.step();
        if (step < cameras.size())
            first += training.loss();
        if (step + cameras.size() >= options.steps)
            last += training.loss();
    }
    const splatting::SplatSet fitted = training.splats();
    const std::vector<float> images = splatting::renderSplats(*vc, fitted, heldOut, options.raster);
    int both = 0, either = 0, count = 0;
    double squaredError = 0.0;
    for (size_t p = 0; p < images.size() / 4; ++p) {
        const float *a = &images[4 * p], *b = &heldOutTargets[4 * p];
        both += a[3] > 0.5f && b[3] > 0.5f;
        either += a[3] > 0.5f || b[3] > 0.5f;
        if (a[3] > 0.99f && b[3] > 0.99f)
            for (int c = 0; c < 3; ++c, ++count)
                squaredError += (a[c] - b[c]) * (a[c] - b[c]);
    }
    const double iou = double(both) / double(either), psnr = 10.0 * std::log10(count / squaredError);
    std::cout << "splat training: " << fitted.size() << " Gaussians, loss " << first << " -> " << last
              << ", held-out IoU " << iou << ", PSNR " << psnr << " dB, instances " << training.instanceCount()
              << std::endl;
    EXPECT_LT(last, 0.1 * first);
    EXPECT_GT(iou, 0.97);
    EXPECT_GT(psnr, 29.0);
}
