// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

/**
 * TESTS:
 * - transformBufferMatchesCpu: the transform buffer holds the translation, scale and the
 *   rotation quaternion of pitch/yaw/roll (about X, then Y, then Z), read from HostValues
 * - transformMatchesCpu: GPU-transformed Gaussians equal klartraum::transformGaussians
 *   on the CPU, including the rotated SH coefficients
 * - mergeConcatenates: merged Gaussians are A's followed by B's, with the SH buffers
 *   re-laid out for the merged count
 * - hostValuesReachTheNextRun: a value set between two runs of a compiled graph is used by
 *   the second run
 **/

#include <gtest/gtest.h>

#include <random>

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

#include "klartraum/computegraph/computegraph.hpp"
#include "klartraum/computegraph/hostvalues.hpp"
#include "klartraum/gaussian_data_standard.hpp"
#include "klartraum/computegraph/gaussianmerge.hpp"
#include "klartraum/computegraph/gaussiantransform.hpp"
#include "klartraum/computegraph/transformbuffer.hpp"
#include "klartraum/gaussian_transform.hpp"
#include "klartraum/headless_frontend.hpp"

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

std::vector<Gaussian3D> randomGaussians(uint32_t count, unsigned seed) {
    std::mt19937 rng(seed);
    std::vector<Gaussian3D> gaussians;
    for (uint32_t i = 0; i < count; ++i) {
        gaussians.push_back(randomGaussian(rng));
    }
    return gaussians;
}

// Reads a buffer (path 0) back as T values.
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

// Reads Gaussians back into Gaussian3D values.
std::vector<Gaussian3D> readBack(const GaussianSoABuffers& buffers) {
    const auto pos = readBack<glm::vec3>(buffers.pos);
    const auto rot = readBack<glm::vec4>(buffers.rot);
    const auto scale = readBack<glm::vec3>(buffers.scale);
    const auto colAlpha = readBack<glm::vec4>(buffers.colAlpha);
    const auto shR = readBack<float>(buffers.shR);
    const auto shG = readBack<float>(buffers.shG);
    const auto shB = readBack<float>(buffers.shB);
    const uint32_t n = buffers.count;
    std::vector<Gaussian3D> result(n);
    for (uint32_t i = 0; i < n; ++i) {
        auto& g = result[i];
        g.position = {pos[i].x, pos[i].y, pos[i].z};
        g.rotation = {rot[i].x, rot[i].y, rot[i].z, rot[i].w};
        g.scale = {scale[i].x, scale[i].y, scale[i].z};
        g.color = {colAlpha[i].x, colAlpha[i].y, colAlpha[i].z};
        g.alpha = colAlpha[i].w;
        for (uint32_t k = 0; k < 15; ++k) {
            g.shR[k] = shR[k * n + i];
            g.shG[k] = shG[k * n + i];
            g.shB[k] = shB[k * n + i];
        }
    }
    return result;
}

// Compares orientations up to the quaternion's sign.
void expectSameGaussian(const Gaussian3D& a, const Gaussian3D& b, float tolerance) {
    for (int k = 0; k < 3; ++k) {
        EXPECT_NEAR(a.position[k], b.position[k], tolerance);
        EXPECT_NEAR(a.scale[k], b.scale[k], tolerance);
        EXPECT_NEAR(a.color[k], b.color[k], tolerance);
    }
    const float sign = (a.rotation[0] * b.rotation[0] + a.rotation[1] * b.rotation[1] + a.rotation[2] * b.rotation[2] +
                        a.rotation[3] * b.rotation[3]) < 0.0f
                           ? -1.0f
                           : 1.0f;
    for (int k = 0; k < 4; ++k) {
        EXPECT_NEAR(a.rotation[k], sign * b.rotation[k], tolerance);
    }
    EXPECT_NEAR(a.alpha, b.alpha, tolerance);
    for (int k = 0; k < 15; ++k) {
        EXPECT_NEAR(a.shR[k], b.shR[k], tolerance) << "coefficient " << k;
        EXPECT_NEAR(a.shG[k], b.shG[k], tolerance) << "coefficient " << k;
        EXPECT_NEAR(a.shB[k], b.shB[k], tolerance) << "coefficient " << k;
    }
}

// x, y, z, pitch, yaw, roll (degrees), scale.
const std::vector<float> kParameters = {0.5f, -1.0f, 2.0f, 10.0f, 35.0f, -20.0f, 1.5f};

glm::quat expectedRotation(const std::vector<float>& p) {
    return glm::angleAxis(glm::radians(p[5]), glm::vec3(0, 0, 1)) *
           glm::angleAxis(glm::radians(p[4]), glm::vec3(0, 1, 0)) *
           glm::angleAxis(glm::radians(p[3]), glm::vec3(1, 0, 0));
}

class GaussianElementsTest : public ::testing::Test {
protected:
    void SetUp() override { frontend = std::make_unique<HeadlessFrontend>(); }
    void TearDown() override {
        // Buffers must go before the device.
        parameters = {};
        frontend.reset();
    }
    VulkanContext& vc() { return frontend->getKlartraumEngine().getVulkanContext(); }

    // One HostValues per parameter, as the studio feeds them.
    std::array<BufferRef, 7> parameterRefs(const std::vector<float>& values) {
        std::array<BufferRef, 7> refs;
        for (int i = 0; i < 7; ++i) {
            parameters[i] = std::make_shared<HostFloat>(vc(), std::vector<float>{values[i]});
            refs[i] = BufferRef{parameters[i]};
        }
        return refs;
    }

    void run(const ComputeGraphElementPtr& root) {
        ComputeGraph graph(vc(), 1);
        graph.compileFrom(root);
        graph.submitAndWait(vc().getGraphicsQueue(), 0);
    }

    std::unique_ptr<HeadlessFrontend> frontend;
    std::array<std::shared_ptr<HostFloat>, 7> parameters;
};

} // namespace

TEST_F(GaussianElementsTest, transformBufferMatchesCpu) {
    const TransformBufferResult transform = createTransformBuffer(vc(), parameterRefs(kParameters));
    run(transform.element);
    const auto t = readBack<float>(transform.transform);
    ASSERT_EQ(t.size(), TransformBuffer::kSize);
    EXPECT_NEAR(t[0], 0.5f, 1e-6f);
    EXPECT_NEAR(t[1], -1.0f, 1e-6f);
    EXPECT_NEAR(t[2], 2.0f, 1e-6f);
    EXPECT_NEAR(t[3], 1.5f, 1e-6f);
    const glm::quat q = expectedRotation(kParameters);
    const float sign = (t[4] * q.x + t[5] * q.y + t[6] * q.z + t[7] * q.w) < 0.0f ? -1.0f : 1.0f;
    EXPECT_NEAR(t[4], sign * q.x, 1e-5f);
    EXPECT_NEAR(t[5], sign * q.y, 1e-5f);
    EXPECT_NEAR(t[6], sign * q.z, 1e-5f);
    EXPECT_NEAR(t[7], sign * q.w, 1e-5f);
}

TEST_F(GaussianElementsTest, transformMatchesCpu) {
    auto gaussians = randomGaussians(1000, 7);
    auto source = std::make_shared<GaussianDataStandard>(vc(), gaussians);
    const TransformBufferResult transform = createTransformBuffer(vc(), parameterRefs(kParameters));
    const auto moved = createGaussianTransform(vc(), source->buffers(), transform.transform);
    run(moved.element);

    transformGaussians(gaussians, expectedRotation(kParameters), kParameters[6],
                       glm::vec3(kParameters[0], kParameters[1], kParameters[2]));
    const auto result = readBack(moved.output);
    ASSERT_EQ(result.size(), gaussians.size());
    for (size_t i = 0; i < result.size(); i += 97) {
        SCOPED_TRACE(i);
        expectSameGaussian(result[i], gaussians[i], 2e-4f);
    }
}

TEST_F(GaussianElementsTest, mergeConcatenates) {
    const auto a = randomGaussians(5, 1);
    const auto b = randomGaussians(7, 2);
    auto sourceA = std::make_shared<GaussianDataStandard>(vc(), a);
    auto sourceB = std::make_shared<GaussianDataStandard>(vc(), b);
    const auto merged = createGaussianMerge(vc(), sourceA->buffers(), sourceB->buffers());
    EXPECT_EQ(merged.output.count, 12u);
    run(merged.element);

    const auto result = readBack(merged.output);
    ASSERT_EQ(result.size(), 12u);
    for (size_t i = 0; i < 12; ++i) {
        SCOPED_TRACE(i);
        expectSameGaussian(result[i], i < 5 ? a[i] : b[i - 5], 0.0f);
    }
}

TEST_F(GaussianElementsTest, hostValuesReachTheNextRun) {
    const TransformBufferResult transform = createTransformBuffer(vc(), parameterRefs(kParameters));
    ComputeGraph graph(vc(), 1);
    graph.compileFrom(transform.element);
    graph.submitAndWait(vc().getGraphicsQueue(), 0);
    EXPECT_NEAR(readBack<float>(transform.transform)[0], 0.5f, 1e-6f);

    parameters[0]->set(0, 4.0f);
    graph.submitAndWait(vc().getGraphicsQueue(), 0);
    EXPECT_NEAR(readBack<float>(transform.transform)[0], 4.0f, 1e-6f);
}
