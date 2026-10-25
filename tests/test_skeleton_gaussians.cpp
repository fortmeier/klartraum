// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

/**
 * TESTS:
 * - characterBonesFormASkeleton: the plain skeleton's 19 bones use valid joints, leave out the face
 *   (nose, eyes, ears) and connect every other joint to the hips
 * - samplesLieAlongBones: each bone's Gaussians are spaced evenly along it, long axis along the bone,
 *   as long as their share of the bone and as thin as the bone radius
 * - jointsAreRoundAndColoured: each used joint gets a round Gaussian of the joint radius; all
 *   Gaussians have the requested colour and opacity, and opacity 0 hides them
 * - transformPlacesSkeleton: the toScene matrix moves and scales the skeleton
 * - sceneHidesSkeletonBehindIt: merged with an opaque wall and rendered, the skeleton shows in front
 *   of the wall and not behind it
 **/

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <set>
#include <vector>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

#include "klartraum/character_controller.hpp"
#include "klartraum/computegraph/computegraph.hpp"
#include "klartraum/rendering/gaussianmerge.hpp"
#include "klartraum/computegraph/hostvalues.hpp"
#include "klartraum/rendering/skeletongaussians.hpp"
#include "klartraum/rendering/gaussian_data_standard.hpp"
#include "klartraum/rendering/gaussian_splatting_factory.hpp"
#include "klartraum/headless_frontend.hpp"
#include "klartraum/interface_camera_orbit.hpp"
#include "klartraum/offscreen_target.hpp"

#include "test_image.hpp"

using namespace klartraum;

namespace {

constexpr float kC0 = 0.282095f;

template <typename T>
std::vector<T> readBack(const BufferRef& ref) {
    auto element = std::dynamic_pointer_cast<TemplatedBufferElementInterface<VulkanBuffer<T>>>(ref.buffer());
    if (!element) {
        throw std::runtime_error("unexpected buffer type");
    }
    std::vector<T> data(element->getBuffer(0).getBufferMemSize() / sizeof(T));
    element->getBuffer(0).memcopyTo(data);
    return data;
}

struct Skeleton {
    std::vector<glm::vec3> positions;
    std::vector<glm::vec4> rotations;
    std::vector<glm::vec3> scales;
    std::vector<glm::vec4> colours;
};

class SkeletonGaussiansGpu : public ::testing::Test {
protected:
    void SetUp() override { frontend = std::make_unique<HeadlessFrontend>(); }
    void TearDown() override { frontend.reset(); }
    VulkanContext& vc() { return frontend->getKlartraumEngine().getVulkanContext(); }

    // Runs a SkeletonGaussians on `joints` once and reads its output back.
    Skeleton run(const std::vector<glm::vec4>& joints, const std::vector<std::pair<uint32_t, uint32_t>>& bones,
                 const SkeletonGaussianParams& params, uint32_t samples) {
        auto jointValues = std::make_shared<HostValues<glm::vec4>>(vc(), joints);
        auto paramValues =
            std::make_shared<HostValues<SkeletonGaussianParams>>(vc(), std::vector<SkeletonGaussianParams>{params});
        const auto skeleton =
            createSkeletonGaussians(vc(), BufferRef{jointValues}, bones, BufferRef{paramValues}, samples);
        ComputeGraph graph(vc(), 1);
        graph.compileFrom(skeleton.element);
        graph.submitAndWait(vc().getGraphicsQueue(), 0);
        return {readBack<glm::vec3>(skeleton.output.pos), readBack<glm::vec4>(skeleton.output.rot),
                readBack<glm::vec3>(skeleton.output.scale), readBack<glm::vec4>(skeleton.output.colAlpha)};
    }

    std::unique_ptr<HeadlessFrontend> frontend;
};

glm::vec3 longAxis(const glm::vec4& r) { return glm::quat(r.w, r.x, r.y, r.z) * glm::vec3(0.0f, 1.0f, 0.0f); }

} // namespace

TEST(SkeletonGaussians, characterBonesFormASkeleton) {
    const auto& bones = characterSkeletonBones();
    EXPECT_EQ(bones.size(), 19u);
    const std::set<uint32_t> face = {5, 6, 7, 8, 9};
    std::set<uint32_t> used;
    for (const auto& [a, b] : bones) {
        ASSERT_LT(a, CharacterController::kJointCount);
        ASSERT_LT(b, CharacterController::kJointCount);
        EXPECT_EQ(face.count(a) + face.count(b), 0u);
        used.insert(a);
        used.insert(b);
    }
    EXPECT_EQ(used.size(), CharacterController::kJointCount - face.size());
    // Every used joint connects to the hips (joint 0).
    std::set<uint32_t> reached = {0};
    for (size_t pass = 0; pass < bones.size(); ++pass) {
        for (const auto& [a, b] : bones) {
            if (reached.count(a) || reached.count(b)) {
                reached.insert(a);
                reached.insert(b);
            }
        }
    }
    EXPECT_EQ(reached, used);
}

TEST_F(SkeletonGaussiansGpu, samplesLieAlongBones) {
    const std::vector<glm::vec4> joints = {{0, 0, 0, 1}, {0, 1, 0, 1}, {1, 1, 1, 1}};
    const std::vector<std::pair<uint32_t, uint32_t>> bones = {{0, 1}, {1, 2}};
    SkeletonGaussianParams params;
    params.shape.x = 0.02f;
    const uint32_t samples = 4;
    const Skeleton s = run(joints, bones, params, samples);
    ASSERT_EQ(s.positions.size(), bones.size() * samples + 3);
    for (size_t b = 0; b < bones.size(); ++b) {
        const glm::vec3 a(joints[bones[b].first]), c(joints[bones[b].second]);
        const float length = glm::distance(a, c);
        for (uint32_t k = 0; k < samples; ++k) {
            const size_t i = b * samples + k;
            const glm::vec3 expected = glm::mix(a, c, (float(k) + 0.5f) / float(samples));
            EXPECT_NEAR(glm::distance(s.positions[i], expected), 0.0f, 1e-5f) << i;
            EXPECT_NEAR(std::abs(glm::dot(longAxis(s.rotations[i]), glm::normalize(c - a))), 1.0f, 1e-4f) << i;
            EXPECT_NEAR(s.scales[i].y, 0.6f * length / float(samples), 1e-5f) << i;
            EXPECT_NEAR(s.scales[i].x, 0.02f, 1e-6f) << i;
            EXPECT_NEAR(s.scales[i].z, 0.02f, 1e-6f) << i;
        }
    }
}

TEST_F(SkeletonGaussiansGpu, jointsAreRoundAndColoured) {
    const std::vector<glm::vec4> joints = {{0, 0, 0, 1}, {0, 1, 0, 1}, {5, 5, 5, 1}};
    const std::vector<std::pair<uint32_t, uint32_t>> bones = {{0, 1}};
    SkeletonGaussianParams params;
    params.colourOpacity = glm::vec4(0.9f, 0.5f, 0.1f, 0.75f);
    params.shape = glm::vec4(0.02f, 1.5f, 0.0f, 0.0f);
    const uint32_t samples = 3;
    const Skeleton s = run(joints, bones, params, samples);
    ASSERT_EQ(s.positions.size(), samples + 2u); // joint 2 is not used by a bone
    for (size_t j = 0; j < 2; ++j) {
        const size_t i = samples + j;
        EXPECT_EQ(s.positions[i], glm::vec3(joints[j]));
        EXPECT_NEAR(s.scales[i].x, 0.03f, 1e-6f);
        EXPECT_NEAR(s.scales[i].y, 0.03f, 1e-6f);
        EXPECT_NEAR(s.scales[i].z, 0.03f, 1e-6f);
    }
    for (const glm::vec4& c : s.colours) {
        EXPECT_NEAR(c.r * kC0 + 0.5f, 0.9f, 1e-5f);
        EXPECT_NEAR(c.g * kC0 + 0.5f, 0.5f, 1e-5f);
        EXPECT_NEAR(c.b * kC0 + 0.5f, 0.1f, 1e-5f);
        EXPECT_FLOAT_EQ(c.a, 0.75f);
    }
    params.colourOpacity.a = 0.0f;
    for (const glm::vec4& c : run(joints, bones, params, samples).colours) {
        EXPECT_EQ(c.a, 0.0f);
    }
}

TEST_F(SkeletonGaussiansGpu, transformPlacesSkeleton) {
    const std::vector<glm::vec4> joints = {{0, 0, 0, 1}, {0, 2, 0, 1}};
    SkeletonGaussianParams params;
    params.toScene =
        glm::translate(glm::mat4(1.0f), glm::vec3(1.0f, 2.0f, 3.0f)) * glm::scale(glm::mat4(1.0f), glm::vec3(0.5f));
    const Skeleton s = run(joints, {{0, 1}}, params, 2);
    EXPECT_NEAR(glm::distance(s.positions[0], glm::vec3(1.0f, 2.25f, 3.0f)), 0.0f, 1e-5f);
    EXPECT_NEAR(glm::distance(s.positions[1], glm::vec3(1.0f, 2.75f, 3.0f)), 0.0f, 1e-5f);
    EXPECT_NEAR(glm::distance(s.positions[2], glm::vec3(1.0f, 2.0f, 3.0f)), 0.0f, 1e-5f);
    EXPECT_NEAR(glm::distance(s.positions[3], glm::vec3(1.0f, 3.0f, 3.0f)), 0.0f, 1e-5f);
    EXPECT_NEAR(s.scales[0].y, 0.6f * 1.0f / 2.0f, 1e-5f);
}

TEST_F(SkeletonGaussiansGpu, sceneHidesSkeletonBehindIt) {
    // An opaque grey wall at z = 0 and a vertical red bone at x = 0, seen from +z.
    std::vector<Gaussian3D> wall;
    for (int x = -20; x <= 20; ++x) {
        for (int y = -20; y <= 20; ++y) {
            Gaussian3D g{};
            g.position = {0.05f * x, 0.05f * y, 0.0f};
            g.rotation = {0.0f, 0.0f, 0.0f, 1.0f};
            g.scale = {0.04f, 0.04f, 0.005f};
            g.alpha = 1.0f;
            wall.push_back(g); // colour 0: grey 0.5
        }
    }
    const VkExtent2D extent{128, 128};
    auto countRed = [&](float boneZ) {
        auto wallData = std::make_shared<GaussianDataStandard>(vc(), wall);
        auto jointValues = std::make_shared<HostValues<glm::vec4>>(
            vc(), std::vector<glm::vec4>{{0.0f, -0.5f, boneZ, 1.0f}, {0.0f, 0.5f, boneZ, 1.0f}});
        SkeletonGaussianParams params;
        params.colourOpacity = glm::vec4(1.0f, 0.0f, 0.0f, 1.0f);
        params.shape.x = 0.03f;
        auto paramValues =
            std::make_shared<HostValues<SkeletonGaussianParams>>(vc(), std::vector<SkeletonGaussianParams>{params});
        const auto skeleton =
            createSkeletonGaussians(vc(), BufferRef{jointValues}, {{0, 1}}, BufferRef{paramValues}, 8);
        const auto merged = createGaussianMerge(vc(), wallData->buffers(), skeleton.output);
        auto target = std::make_shared<OffscreenTarget>(vc(), extent, 1u);
        auto cameraUBO = std::make_shared<CameraUboType>();
        auto splatting = createGaussianSplatting(vc(), GsplatBackend::Compute, target, cameraUBO, merged.output);
        ComputeGraph graph(vc(), 1);
        graph.compileFrom(splatting);
        InterfaceCameraOrbit orbit(InterfaceCameraOrbit::UpDirection::Y);
        orbit.initialize(vc());
        orbit.setAzimuth(0.5 * 3.14159265358979);
        orbit.setElevation(0.0);
        orbit.setDistance(3.0);
        orbit.setProjectionAspectRatio(1.0f);
        orbit.update(cameraUBO->ubo);
        cameraUBO->update(0);
        graph.submitAndWait(vc().getGraphicsQueue(), 0);
        const auto pixels = test_image::readImage(vc(), target->getImage(0), extent);
        size_t red = 0;
        for (size_t i = 0; i < pixels.size(); i += 4) {
            red += pixels[i + 2] > 180 && pixels[i + 1] < 100; // BGRA
        }
        return red;
    };
    EXPECT_GT(countRed(0.5f), 50u);
    EXPECT_EQ(countRed(-0.5f), 0u);
}
