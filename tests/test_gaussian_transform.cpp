// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

/**
 * TESTS:
 * - identityKeepsGaussians: the identity transform leaves every field unchanged
 * - translateAndScale: positions are scaled about the origin and moved; sizes scale along
 * - rotationTurnsPositionsAndOrientations: positions rotate and each Gaussian's covariance
 *   becomes R * C * R^T
 * - rotationTurnsViewDependentColour: the rotated coefficients give, at direction d, the
 *   colour the original ones give at R^T d, for all three bands
 * - loadGaussiansSpzFlipsY: loading with flipY mirrors positions across the Y axis
 **/

#include <gtest/gtest.h>

#include <filesystem>
#include <random>

#include "klartraum/gaussian_data_standard.hpp"
#include "klartraum/gaussian_transform.hpp"

#include "test_scene.hpp"

using namespace klartraum;

namespace {

Gaussian3D randomGaussian(std::mt19937& rng) {
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    Gaussian3D g{};
    g.position = {u(rng), u(rng), u(rng)};
    const glm::quat q = glm::normalize(glm::quat(u(rng), u(rng), u(rng), u(rng)));
    g.rotation = {q.x, q.y, q.z, q.w};
    g.scale = {0.1f + 0.5f * (u(rng) + 1.0f), 0.1f + 0.5f * (u(rng) + 1.0f), 0.1f + 0.5f * (u(rng) + 1.0f)};
    g.color = {u(rng), u(rng), u(rng)};
    g.alpha = 0.5f * (u(rng) + 1.0f);
    for (int i = 0; i < 15; ++i) {
        g.shR[i] = u(rng);
        g.shG[i] = u(rng);
        g.shB[i] = u(rng);
    }
    return g;
}

glm::mat3 covariance(const Gaussian3D& g) {
    const glm::mat3 r = glm::mat3_cast(glm::quat(g.rotation[3], g.rotation[0], g.rotation[1], g.rotation[2]));
    const glm::mat3 s2(g.scale[0] * g.scale[0], 0, 0, 0, g.scale[1] * g.scale[1], 0, 0, 0, g.scale[2] * g.scale[2]);
    return r * s2 * glm::transpose(r);
}

float shColour(const std::array<float, 15>& c, const glm::vec3& dir) {
    const auto basis = shBasis(dir);
    float value = 0.0f;
    for (int i = 0; i < 15; ++i) {
        value += c[i] * basis[i];
    }
    return value;
}

const glm::quat kRotation = glm::angleAxis(1.1f, glm::normalize(glm::vec3(0.3f, -0.8f, 0.5f)));

} // namespace

TEST(GaussianTransform, identityKeepsGaussians) {
    std::mt19937 rng(1);
    std::vector<Gaussian3D> gaussians{randomGaussian(rng), randomGaussian(rng)};
    const auto original = gaussians;
    transformGaussians(gaussians, glm::quat(1.0f, 0.0f, 0.0f, 0.0f));
    for (size_t i = 0; i < gaussians.size(); ++i) {
        EXPECT_EQ(gaussians[i].position, original[i].position);
        EXPECT_EQ(gaussians[i].rotation, original[i].rotation);
        EXPECT_EQ(gaussians[i].scale, original[i].scale);
        EXPECT_EQ(gaussians[i].shR, original[i].shR);
    }
}

TEST(GaussianTransform, translateAndScale) {
    std::mt19937 rng(2);
    std::vector<Gaussian3D> gaussians{randomGaussian(rng)};
    const Gaussian3D original = gaussians[0];
    transformGaussians(gaussians, glm::quat(1.0f, 0.0f, 0.0f, 0.0f), 2.0f, glm::vec3(1.0f, -2.0f, 3.0f));
    const Gaussian3D& g = gaussians[0];
    EXPECT_FLOAT_EQ(g.position[0], 2.0f * original.position[0] + 1.0f);
    EXPECT_FLOAT_EQ(g.position[1], 2.0f * original.position[1] - 2.0f);
    EXPECT_FLOAT_EQ(g.position[2], 2.0f * original.position[2] + 3.0f);
    for (int i = 0; i < 3; ++i) {
        EXPECT_FLOAT_EQ(g.scale[i], 2.0f * original.scale[i]);
    }
    EXPECT_EQ(g.alpha, original.alpha);
    EXPECT_EQ(g.color, original.color);
    EXPECT_THROW(transformGaussians(gaussians, glm::quat(1.0f, 0.0f, 0.0f, 0.0f), 0.0f), std::invalid_argument);
}

TEST(GaussianTransform, rotationTurnsPositionsAndOrientations) {
    std::mt19937 rng(3);
    std::vector<Gaussian3D> gaussians{randomGaussian(rng), randomGaussian(rng), randomGaussian(rng)};
    const auto original = gaussians;
    transformGaussians(gaussians, kRotation);
    const glm::mat3 r = glm::mat3_cast(kRotation);
    for (size_t i = 0; i < gaussians.size(); ++i) {
        const glm::vec3 expected =
            r * glm::vec3(original[i].position[0], original[i].position[1], original[i].position[2]);
        for (int k = 0; k < 3; ++k) {
            EXPECT_NEAR(gaussians[i].position[k], expected[k], 1e-5f);
        }
        const glm::mat3 before = r * covariance(original[i]) * glm::transpose(r);
        const glm::mat3 after = covariance(gaussians[i]);
        for (int c = 0; c < 3; ++c) {
            for (int k = 0; k < 3; ++k) {
                EXPECT_NEAR(after[c][k], before[c][k], 1e-5f);
            }
        }
    }
}

TEST(GaussianTransform, rotationTurnsViewDependentColour) {
    std::mt19937 rng(4);
    std::vector<Gaussian3D> gaussians{randomGaussian(rng)};
    const Gaussian3D original = gaussians[0];
    transformGaussians(gaussians, kRotation);
    const glm::mat3 inverse = glm::transpose(glm::mat3_cast(kRotation));

    // One band at a time, so each is checked on its own.
    for (int band = 0; band < 3; ++band) {
        SCOPED_TRACE(band + 1);
        const int start = band == 0 ? 0 : band == 1 ? 3 : 8;
        const int size = 2 * band + 3;
        std::array<float, 15> onlyBand{}, rotated = original.shR;
        for (int i = start; i < start + size; ++i) {
            onlyBand[i] = original.shR[i];
        }
        std::vector<Gaussian3D> single(1, original);
        single[0].shR = onlyBand;
        transformGaussians(single, kRotation);
        rotated = single[0].shR;
        std::uniform_real_distribution<float> u(-1.0f, 1.0f);
        for (int i = 0; i < 20; ++i) {
            const glm::vec3 d = glm::normalize(glm::vec3(u(rng), u(rng), u(rng)));
            EXPECT_NEAR(shColour(rotated, d), shColour(onlyBand, inverse * d), 1e-4f);
        }
    }
    // All bands together, for every channel.
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    for (int i = 0; i < 20; ++i) {
        const glm::vec3 d = glm::normalize(glm::vec3(u(rng), u(rng), u(rng)));
        EXPECT_NEAR(shColour(gaussians[0].shR, d), shColour(original.shR, inverse * d), 1e-4f);
        EXPECT_NEAR(shColour(gaussians[0].shG, d), shColour(original.shG, inverse * d), 1e-4f);
        EXPECT_NEAR(shColour(gaussians[0].shB, d), shColour(original.shB, inverse * d), 1e-4f);
    }
}

TEST(GaussianTransform, loadGaussiansSpzFlipsY) {
    const std::string& path = test_scene::kLanternPath;
    if (!std::filesystem::exists(path)) {
        GTEST_SKIP() << "SPZ scene not found: " << path;
    }
    const auto plain = loadGaussiansSpz(path);
    const auto flipped = loadGaussiansSpz(path, true);
    ASSERT_GT(plain.size(), 0u);
    ASSERT_EQ(plain.size(), flipped.size());
    for (size_t i = 0; i < plain.size(); i += plain.size() / 17) {
        EXPECT_FLOAT_EQ(flipped[i].position[0], plain[i].position[0]);
        EXPECT_FLOAT_EQ(flipped[i].position[1], -plain[i].position[1]);
        EXPECT_FLOAT_EQ(flipped[i].position[2], plain[i].position[2]);
    }
}
