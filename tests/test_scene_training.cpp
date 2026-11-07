// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

/**
 * TESTS:
 * - renderSceneComposites: renderScene() of the pumpkin with fully transparent Gaussians equals the mesh alone
 *   (renderImages()), and with Gaussians all in front of it equals the Gaussians alone (renderSplats()) over the mesh
 * - appearanceFitsRenderedViews: SceneTraining fits the texture of the pumpkin (from noise) and the colours and
 *   opacities of a ring of Gaussians around it (from grey and half opacity) to 24 rendered views of the true scene;
 *   three held-out views of the scene reach a PSNR above 28 dB, and rendered alone, the mesh and the Gaussians each
 *   reach a PSNR above 26 dB against their true appearance
 **/

#include <gtest/gtest.h>

#include <cmath>
#include <iostream>
#include <memory>
#include <vector>

#include <glm/glm.hpp>

#include "klartraum/headless_frontend.hpp"
#include "klartraum/raster/mesh.hpp"
#include "klartraum/raster/mesh_training.hpp"
#include "klartraum/raster/scene_training.hpp"
#include "klartraum/splatting/splat_training.hpp"

using namespace klartraum;

namespace {

class SceneTrainingTest : public ::testing::Test {
protected:
    void SetUp() override {
        frontend = std::make_unique<HeadlessFrontend>();
        vc = &frontend->getKlartraumEngine().getVulkanContext();
    }

    std::unique_ptr<HeadlessFrontend> frontend;
    VulkanContext* vc = nullptr;
};

// `count` Gaussians on a horizontal ring of radius `radius` around the y axis, coloured by their angle.
splatting::SplatSet ring(uint32_t count, float radius, float y, float opacity) {
    std::vector<glm::vec3> points;
    for (uint32_t i = 0; i < count; ++i) {
        const float angle = 6.2831853f * float(i) / float(count);
        points.emplace_back(radius * std::cos(angle), y, radius * std::sin(angle));
    }
    splatting::SplatSet splats = splatting::splatsAt(points, 0.06f, glm::vec3(0.5f), opacity);
    for (uint32_t i = 0; i < count; ++i) {
        const float angle = 6.2831853f * float(i) / float(count);
        splats.colors[4 * i] = 0.5f + 0.45f * std::cos(angle);
        splats.colors[4 * i + 1] = 0.5f + 0.45f * std::sin(3.0f * angle);
        splats.colors[4 * i + 2] = 0.2f + 0.6f * float(i % 5) / 4.0f;
    }
    return splats;
}

// The PSNR of the rgb of `a` against `b` (four floats per pixel) over the pixels either covers.
double psnr(const std::vector<float>& a, const std::vector<float>& b) {
    double squared = 0.0;
    size_t count = 0;
    for (size_t p = 0; p < a.size() / 4; ++p) {
        if (a[4 * p + 3] < 0.01f && b[4 * p + 3] < 0.01f)
            continue;
        for (int c = 0; c < 3; ++c, ++count)
            squared += (a[4 * p + c] - b[4 * p + c]) * (a[4 * p + c] - b[4 * p + c]);
    }
    return 10.0 * std::log10(double(count) / std::max(squared, 1e-12));
}

} // namespace

TEST_F(SceneTrainingTest, renderSceneComposites) {
    raster::SceneTrainingOptions options;
    options.raster.width = 96;
    options.raster.height = 80;
    options.raster.capacity = 1u << 16;
    const raster::TriangleMesh pumpkin = raster::syntheticPumpkin(3);
    const std::vector<float> texture = raster::syntheticPumpkinTexture(options.textureWidth, options.textureHeight);
    const std::vector<glm::mat4> cameras = raster::cameraRing(2, 3.5f, 0.3f, 0.8f, 96.0f / 80.0f);
    const std::vector<float> mesh =
        raster::renderImages(*vc, pumpkin, texture, options.textureWidth, options.textureHeight, cameras, 96, 80);

    splatting::SplatSet hidden = ring(40, 1.4f, 0.0f, 0.5f);
    for (float& opacity : hidden.opacities)
        opacity = -30.0f;
    const std::vector<float> alone = raster::renderScene(*vc, pumpkin, texture, hidden, cameras, options);
    ASSERT_EQ(alone.size(), mesh.size());
    float largest = 0.0f;
    for (size_t i = 0; i < mesh.size(); ++i)
        largest = std::max(largest, std::abs(alone[i] - mesh[i]));
    EXPECT_LT(largest, 1e-6f) << "transparent Gaussians change the mesh's image";

    // Each camera's own ring of Gaussians between it and the pumpkin, so all of them are in front of the mesh.
    for (uint32_t view = 0; view < 2; ++view) {
        // The eye: the point the camera maps to clip x = y = w = 0.
        const glm::mat4& m = cameras[view];
        const auto row = [&](int k) { return glm::vec3(m[0][k], m[1][k], m[2][k]); };
        const glm::mat3 rows = glm::transpose(glm::mat3(row(0), row(1), row(3)));
        const glm::vec3 eye = glm::inverse(rows) * -glm::vec3(m[3][0], m[3][1], m[3][3]);
        std::vector<glm::vec3> points;
        for (int i = 0; i < 30; ++i) {
            const float angle = 6.2831853f * float(i) / 30.0f;
            const glm::vec3 side = glm::normalize(glm::cross(eye, glm::vec3(0, 1, 0)));
            points.push_back(0.45f * eye + 0.6f * (std::cos(angle) * side + std::sin(angle) * glm::vec3(0, 1, 0)));
        }
        const splatting::SplatSet front = splatting::splatsAt(points, 0.08f, glm::vec3(0.9f, 0.2f, 0.1f), 0.6f);
        const std::vector<glm::mat4> one = {cameras[view]};
        const std::vector<float> scene = raster::renderScene(*vc, pumpkin, texture, front, one, options);
        const std::vector<float> splats = splatting::renderSplats(*vc, front, one, options.raster);
        const size_t pixels = size_t(96) * 80, offset = 4 * pixels * view;
        double covered = 0.0;
        largest = 0.0f;
        for (size_t p = 0; p < pixels; ++p) {
            covered += splats[4 * p + 3];
            for (int c = 0; c < 4; ++c) {
                const float expected = splats[4 * p + c] + (1.0f - splats[4 * p + 3]) * mesh[offset + 4 * p + c];
                largest = std::max(largest, std::abs(scene[4 * p + c] - expected));
            }
        }
        EXPECT_LT(largest, 1e-5f) << "view " << view;
        EXPECT_GT(covered, 0.05 * double(pixels)) << "the Gaussians should cover part of view " << view;
    }
}

TEST_F(SceneTrainingTest, appearanceFitsRenderedViews) {
    raster::SceneTrainingOptions options;
    options.raster.width = options.raster.height = 128;
    options.raster.capacity = 1u << 18;
    std::vector<glm::mat4> cameras, heldOut;
    for (const auto& [elevation, start] : {std::pair{-0.25f, 0.0f}, {0.3f, 0.39f}, {0.85f, 0.0f}}) {
        const auto ring = raster::cameraRing(8, 3.5f, elevation, 0.8f, 1.0f, start);
        cameras.insert(cameras.end(), ring.begin(), ring.end());
    }
    for (const float angle : {0.2f, 2.3f, 4.4f})
        heldOut.push_back(raster::cameraRing(1, 3.5f, 0.05f + 0.1f * angle, 0.8f, 1.0f, angle)[0]);
    const raster::TriangleMesh pumpkin = raster::syntheticPumpkin(4);
    const std::vector<float> trueTexture = raster::syntheticPumpkinTexture(options.textureWidth, options.textureHeight);
    const splatting::SplatSet trueSplats = ring(160, 1.35f, 0.1f, 0.7f);
    const std::vector<float> targets = raster::renderScene(*vc, pumpkin, trueTexture, trueSplats, cameras, options);

    // Start: a noisy grey texture, grey Gaussians at half opacity (same positions and shapes).
    std::vector<float> texture(trueTexture.size());
    uint32_t state = 7u;
    for (size_t i = 0; i < texture.size(); ++i) {
        state = state * 1664525u + 1013904223u;
        texture[i] = i % 4 == 3 ? 1.0f : 0.3f + 0.4f * float(state >> 8) / float(1u << 24);
    }
    splatting::SplatSet splats = trueSplats;
    for (uint32_t i = 0; i < splats.size(); ++i) {
        for (int c = 0; c < 3; ++c)
            splats.colors[4 * i + c] = 0.5f;
        splats.opacities[i] = 0.0f;
    }

    raster::SceneTraining training(*vc, pumpkin, texture, splats, cameras, targets, options);
    const uint32_t steps = 2000;
    double first = 0.0, last = 0.0;
    for (uint32_t step = 0; step < steps; ++step) {
        training.step();
        if (step < cameras.size())
            first += training.loss();
        if (step + cameras.size() >= steps)
            last += training.loss();
    }
    const std::vector<float> fittedTexture = training.texture();
    const splatting::SplatSet fittedSplats = training.splats();
    const double scene = psnr(raster::renderScene(*vc, pumpkin, fittedTexture, fittedSplats, heldOut, options),
                              raster::renderScene(*vc, pumpkin, trueTexture, trueSplats, heldOut, options));
    const double mesh = psnr(raster::renderImages(*vc, pumpkin, fittedTexture, options.textureWidth,
                                                  options.textureHeight, heldOut, 128, 128),
                             raster::renderImages(*vc, pumpkin, trueTexture, options.textureWidth,
                                                  options.textureHeight, heldOut, 128, 128));
    const double gaussians = psnr(splatting::renderSplats(*vc, fittedSplats, heldOut, options.raster),
                                  splatting::renderSplats(*vc, trueSplats, heldOut, options.raster));
    std::cout << "scene training: loss " << first << " -> " << last << ", held-out PSNR scene " << scene
              << " dB, mesh alone " << mesh << " dB, Gaussians alone " << gaussians << " dB" << std::endl;
    EXPECT_LT(last, 0.05 * first);
    EXPECT_GT(scene, 28.0);
    EXPECT_GT(mesh, 26.0);
    EXPECT_GT(gaussians, 26.0);
}
