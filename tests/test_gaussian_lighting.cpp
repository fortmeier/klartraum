// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

/**
 * TESTS:
 * - visibilityBasisIsOrthonormal: the nine SH basis functions integrate to the identity over the sphere
 * - isolatedSplatSeesEverything: a splat with nothing around it is visible from every direction
 * - enclosedSplatIsShadowed: a splat inside a closed shell of opaque splats sees little in any direction
 * - normalsPointToTheOpenSide: on a thick slab of flat splats, the normals of the top layer point up
 *   and those of the bottom layer down
 * - relightZeroKeepsColours: at relight 0 the output colours equal the input bit for bit, lights or not
 * - ambientOnlyScalesAlbedo: without lights the colour is albedo x ambient x average visibility, alpha
 *   unchanged
 * - lightBehindAddsNothing: a light behind a splat's surface leaves the ambient-only colour
 * - lambertFollowsCosine: without specular, the light a splat receives at 60 degrees is half of that
 *   head-on, and at the reference distance head-on it is albedo x colour x intensity / pi
 * - unshadowedLightIgnoresVisibility: a light marked unshadowed lights an enclosed splat as if it
 *   were open
 * - glowLightsSplatsAroundIt: within its glow radius a light makes splats emit its colour, fading with
 *   the squared distance to the edge of the radius, whichever way they face; beyond it nothing changes
 * - relitLanternDiffersWhereLit: the lantern relit at relight 0 renders exactly like the plain scene;
 *   with a warm light next to it and dimmed ambient it changes visibly; frames in
 *   build/TestingOutput/gaussian_lighting/
 **/

#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <memory>
#include <random>
#include <vector>

#include <glm/gtc/quaternion.hpp>

#include "klartraum/computegraph/computegraph.hpp"
#include "klartraum/computegraph/hostvalues.hpp"
#include "klartraum/gaussian_data_standard.hpp"
#include "klartraum/gaussian_lighting.hpp"
#include "klartraum/gaussian_splatting_factory.hpp"
#include "klartraum/headless_frontend.hpp"
#include "klartraum/interface_camera_orbit.hpp"
#include "klartraum/offscreen_target.hpp"

#include "test_scene.hpp"

using namespace klartraum;

namespace {

constexpr float kC0 = 0.282095f;
constexpr float kPi = 3.14159265358979f;

std::vector<glm::vec3> fibonacciSphere(int n) {
    std::vector<glm::vec3> directions;
    for (int i = 0; i < n; ++i) {
        const float y = 1.0f - 2.0f * (float(i) + 0.5f) / float(n);
        const float r = std::sqrt(1.0f - y * y);
        const float angle = float(i) * 2.39996323f;
        directions.push_back({r * std::cos(angle), y, r * std::sin(angle)});
    }
    return directions;
}

Gaussian3D splat(const glm::vec3& p, const glm::vec3& scale, float alpha = 1.0f,
                 const glm::vec3& rgb = glm::vec3(0.5f)) {
    Gaussian3D g{};
    g.position = {p.x, p.y, p.z};
    g.rotation = {0.0f, 0.0f, 0.0f, 1.0f};
    g.scale = {scale.x, scale.y, scale.z};
    g.color = {(rgb.r - 0.5f) / kC0, (rgb.g - 0.5f) / kC0, (rgb.b - 0.5f) / kC0};
    g.alpha = alpha;
    return g;
}

std::vector<Gaussian3D> randomGaussians(size_t n, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    std::vector<Gaussian3D> result(n);
    for (auto& g : result) {
        g = splat({u(rng), u(rng), u(rng)}, glm::vec3(0.02f + 0.01f * u(rng)), 0.5f + 0.4f * u(rng),
                  glm::vec3(0.5f) + 0.4f * glm::vec3(u(rng), u(rng), u(rng)));
        for (int k = 0; k < 15; ++k) {
            g.shR[k] = 0.1f * u(rng);
        }
    }
    return result;
}

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

// The linear colour a shaded SH degree-0 value stands for.
glm::vec3 decode(const glm::vec4& colAlpha) { return glm::vec3(colAlpha) * kC0 + glm::vec3(0.5f); }

// Lights `gaussians` with `materials` for one frame and reads the colours back.
class LightingRunner {
public:
    LightingRunner(VulkanContext& vc, const std::vector<Gaussian3D>& gaussians,
                   const std::vector<SplatMaterial>& materials)
        : vc(vc),
          data(std::make_shared<GaussianDataStandard>(vc, gaussians)),
          frame(std::make_shared<HostValues<LightingFrame>>(vc, std::vector<LightingFrame>{LightingFrame{}})) {
        lighting = createGaussianLighting(vc, data->buffers(), materials, BufferRef{frame});
        graph = std::make_unique<ComputeGraph>(vc, 1);
        graph->compileFrom(lighting.element);
    }

    std::vector<glm::vec4> run(const LightingFrame& values) {
        frame->set(0, values);
        graph->submitAndWait(vc.getGraphicsQueue(), 0);
        return readBack<glm::vec4>(lighting.output.colAlpha);
    }

private:
    VulkanContext& vc;
    std::shared_ptr<GaussianDataStandard> data;
    std::shared_ptr<HostValues<LightingFrame>> frame;
    GaussianLightingResult lighting;
    std::unique_ptr<ComputeGraph> graph;
};

class GaussianLightingGpu : public ::testing::Test {
protected:
    void SetUp() override { frontend = std::make_unique<HeadlessFrontend>(); }
    void TearDown() override { frontend.reset(); }
    VulkanContext& vc() { return frontend->getKlartraumEngine().getVulkanContext(); }
    std::unique_ptr<HeadlessFrontend> frontend;
};

std::vector<uint8_t> readImage(VulkanContext& vc, VkImage image, VkExtent2D extent) {
    const VkDeviceSize bytes = VkDeviceSize(extent.width) * extent.height * 4;
    VkBuffer buf;
    VkDeviceMemory mem;
    vc.createBuffer(bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, buf, mem);
    VkCommandBufferAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    ai.commandPool = vc.getCommandPool();
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    VkCommandBuffer cmd;
    vkAllocateCommandBuffers(vc.getDevice(), &ai, &cmd);
    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &bi);
    VkBufferImageCopy region{};
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.layerCount = 1;
    region.imageExtent = {extent.width, extent.height, 1};
    vkCmdCopyImageToBuffer(cmd, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, buf, 1, &region);
    vkEndCommandBuffer(cmd);
    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    vkQueueSubmit(vc.getGraphicsQueue(), 1, &si, VK_NULL_HANDLE);
    vkQueueWaitIdle(vc.getGraphicsQueue());
    vkFreeCommandBuffers(vc.getDevice(), vc.getCommandPool(), 1, &cmd);
    void* mapped;
    vkMapMemory(vc.getDevice(), mem, 0, bytes, 0, &mapped);
    std::vector<uint8_t> result(static_cast<const uint8_t*>(mapped), static_cast<const uint8_t*>(mapped) + bytes);
    vkUnmapMemory(vc.getDevice(), mem);
    vkFreeMemory(vc.getDevice(), mem, nullptr);
    vkDestroyBuffer(vc.getDevice(), buf, nullptr);
    return result;
}

void writePPM(const std::filesystem::path& path, const std::vector<uint8_t>& bgra, VkExtent2D extent) {
    std::ofstream out(path, std::ios::binary);
    out << "P6\n" << extent.width << " " << extent.height << "\n255\n";
    for (size_t i = 0; i < bgra.size(); i += 4) {
        const char rgb[3] = {char(bgra[i + 2]), char(bgra[i + 1]), char(bgra[i])};
        out.write(rgb, 3);
    }
}

} // namespace

TEST(GaussianLighting, visibilityBasisIsOrthonormal) {
    const auto directions = fibonacciSphere(8192);
    for (int a = 0; a < 9; ++a) {
        for (int b = 0; b < 9; ++b) {
            double integral = 0.0;
            for (const auto& d : directions) {
                const auto basis = visibilityBasis(d);
                integral += double(basis[a]) * basis[b];
            }
            integral *= 4.0 * kPi / double(directions.size());
            EXPECT_NEAR(integral, a == b ? 1.0 : 0.0, 0.01) << a << ", " << b;
        }
    }
}

TEST(GaussianLighting, isolatedSplatSeesEverything) {
    const std::vector<Gaussian3D> scene = {splat(glm::vec3(0.0f), glm::vec3(0.1f, 0.1f, 0.01f))};
    const auto materials = bakeSplatMaterials(scene);
    ASSERT_EQ(materials.size(), 1u);
    for (const auto& d : fibonacciSphere(64)) {
        EXPECT_GT(evaluateVisibility(materials[0], d), 0.95f);
    }
}

TEST(GaussianLighting, enclosedSplatIsShadowed) {
    // A centre splat inside a dense shell of large opaque splats.
    std::vector<Gaussian3D> scene = {splat(glm::vec3(0.0f), glm::vec3(0.02f))};
    for (const auto& d : fibonacciSphere(4000)) {
        scene.push_back(splat(d, glm::vec3(0.08f), 1.0f));
    }
    const auto materials = bakeSplatMaterials(scene, 0.8f, 0.0f, 1.0f, 16);
    for (const auto& d : fibonacciSphere(64)) {
        EXPECT_LT(evaluateVisibility(materials[0], d), 0.3f) << d.x << " " << d.y << " " << d.z;
    }
}

TEST(GaussianLighting, normalsPointToTheOpenSide) {
    // Flat splats (shortest axis y) filling a slab from y = -0.5 to y = 0.
    std::vector<Gaussian3D> scene;
    for (int x = 0; x < 30; ++x) {
        for (int z = 0; z < 30; ++z) {
            for (int y = 0; y <= 10; ++y) {
                scene.push_back(splat({-1.5f + 0.1f * x, -0.05f * y, -1.5f + 0.1f * z}, {0.08f, 0.01f, 0.08f}));
            }
        }
    }
    const auto materials = bakeSplatMaterials(scene);
    int topUp = 0, bottomDown = 0, top = 0, bottom = 0;
    for (size_t i = 0; i < scene.size(); ++i) {
        const float x = scene[i].position[0], z = scene[i].position[2];
        if (std::abs(x) > 1.0f || std::abs(z) > 1.0f) {
            continue; // away from the slab's edges
        }
        if (scene[i].position[1] == 0.0f) {
            ++top;
            topUp += materials[i].normalRoughness.y > 0.9f;
        } else if (scene[i].position[1] < -0.49f) {
            ++bottom;
            bottomDown += materials[i].normalRoughness.y < -0.9f;
        }
    }
    ASSERT_GT(top, 0);
    EXPECT_EQ(topUp, top);
    EXPECT_EQ(bottomDown, bottom);
}

TEST_F(GaussianLightingGpu, relightZeroKeepsColours) {
    const auto scene = randomGaussians(1000, 1);
    const auto materials = bakeSplatMaterials(scene);
    LightingRunner runner(vc(), scene, materials);
    LightingFrame frame;
    frame.counts.x = 1;
    frame.lights[0] = {{0.0f, 2.0f, 0.0f, 0.0f}, {1.0f, 0.5f, 0.2f, 5.0f}, {1.0f, 0.0f, 0.0f, 0.0f}};
    frame.ambient = glm::vec4(0.1f);
    frame.control.x = 0.0f;
    const auto colours = runner.run(frame);
    size_t wrong = 0;
    for (size_t i = 0; i < scene.size(); ++i) {
        wrong += colours[i] != glm::vec4(scene[i].color[0], scene[i].color[1], scene[i].color[2], scene[i].alpha);
    }
    EXPECT_EQ(wrong, 0u);
}

TEST_F(GaussianLightingGpu, ambientOnlyScalesAlbedo) {
    const auto scene = randomGaussians(500, 2);
    const auto materials = bakeSplatMaterials(scene);
    LightingRunner runner(vc(), scene, materials);
    LightingFrame frame;
    frame.ambient = glm::vec4(0.5f, 0.4f, 0.3f, 1.0f);
    frame.control.x = 1.0f;
    const auto colours = runner.run(frame);
    for (size_t i = 0; i < scene.size(); ++i) {
        const glm::vec3 albedo =
            glm::max(decode(glm::vec4(scene[i].color[0], scene[i].color[1], scene[i].color[2], 0)), glm::vec3(0.0f));
        const float ao = std::clamp(materials[i].visibility0.x * kC0, 0.0f, 1.0f);
        const glm::vec3 expected = albedo * glm::vec3(frame.ambient) * ao;
        const glm::vec3 got = decode(colours[i]);
        EXPECT_NEAR(got.r, expected.r, 1e-4f) << i;
        EXPECT_NEAR(got.g, expected.g, 1e-4f) << i;
        EXPECT_NEAR(got.b, expected.b, 1e-4f) << i;
        EXPECT_EQ(colours[i].w, scene[i].alpha) << i;
    }
}

TEST_F(GaussianLightingGpu, lightBehindAddsNothing) {
    const std::vector<Gaussian3D> scene = {splat(glm::vec3(0.0f), glm::vec3(0.1f, 0.1f, 0.01f))};
    std::vector<SplatMaterial> materials(1);
    materials[0].normalRoughness = glm::vec4(0.0f, 0.0f, 1.0f, 0.5f);
    LightingRunner runner(vc(), scene, materials);
    LightingFrame frame;
    frame.camera = glm::vec4(0.0f, 0.0f, 3.0f, 0.0f);
    frame.ambient = glm::vec4(0.2f);
    frame.control.x = 1.0f;
    const glm::vec4 ambientOnly = runner.run(frame)[0];
    frame.counts.x = 1;
    frame.lights[0] = {{0.3f, 0.0f, -1.0f, 0.0f}, {1.0f, 1.0f, 1.0f, 10.0f}, {1.0f, 0.0f, 0.0f, 0.0f}};
    EXPECT_EQ(runner.run(frame)[0], ambientOnly);
}

TEST_F(GaussianLightingGpu, lambertFollowsCosine) {
    const glm::vec3 albedo(0.6f, 0.5f, 0.4f);
    const std::vector<Gaussian3D> scene = {splat(glm::vec3(0.0f), glm::vec3(0.1f, 0.1f, 0.01f), 1.0f, albedo)};
    std::vector<SplatMaterial> materials(1);
    materials[0].normalRoughness = glm::vec4(0.0f, 0.0f, 1.0f, 1.0f);
    materials[0].parameters = glm::vec4(0.0f, 0.0f, 0.0f, 0.0f); // no metal, no specular
    LightingRunner runner(vc(), scene, materials);
    LightingFrame frame;
    frame.camera = glm::vec4(0.0f, 0.0f, 3.0f, 0.0f);
    frame.ambient = glm::vec4(0.0f);
    frame.control.x = 1.0f;
    frame.counts.x = 1;
    const float distance = 2.0f;
    auto shadeFrom = [&](float degrees) {
        const float a = glm::radians(degrees);
        frame.lights[0] = {{distance * std::sin(a), 0.0f, distance * std::cos(a), 0.0f},
                           {1.0f, 0.8f, 0.6f, 2.0f},
                           {distance, 0.0f, 0.0f, 0.0f}};
        return decode(runner.run(frame)[0]);
    };
    const glm::vec3 headOn = shadeFrom(0.0f);
    const glm::vec3 oblique = shadeFrom(60.0f);
    const glm::vec3 expected = albedo * glm::vec3(1.0f, 0.8f, 0.6f) * 2.0f / kPi;
    for (int c = 0; c < 3; ++c) {
        EXPECT_NEAR(headOn[c], expected[c], 1e-4f) << c;
        EXPECT_NEAR(oblique[c], 0.5f * headOn[c], 1e-4f) << c;
    }
}

TEST_F(GaussianLightingGpu, unshadowedLightIgnoresVisibility) {
    const std::vector<Gaussian3D> scene = {splat(glm::vec3(0.0f), glm::vec3(0.1f, 0.1f, 0.01f))};
    std::vector<SplatMaterial> open(1), enclosed(1);
    open[0].normalRoughness = enclosed[0].normalRoughness = glm::vec4(0.0f, 0.0f, 1.0f, 1.0f);
    enclosed[0].visibility0 = glm::vec4(0.3f, 0.0f, 0.0f, 0.0f); // about 8 % visible everywhere
    LightingFrame frame;
    frame.camera = glm::vec4(0.0f, 0.0f, 3.0f, 0.0f);
    frame.ambient = glm::vec4(0.0f);
    frame.control.x = 1.0f;
    frame.counts.x = 1;
    frame.lights[0] = {{0.0f, 0.0f, 1.0f, 0.0f}, {1.0f, 1.0f, 1.0f, 2.0f}, {1.0f, 0.0f, 0.0f, 0.0f}};
    LightingRunner openRunner(vc(), scene, open);
    LightingRunner enclosedRunner(vc(), scene, enclosed);
    const glm::vec3 shadowed = decode(enclosedRunner.run(frame)[0]);
    frame.lights[0].shape.z = 1.0f;
    const glm::vec3 unshadowed = decode(enclosedRunner.run(frame)[0]);
    const glm::vec3 reference = decode(openRunner.run(frame)[0]);
    EXPECT_LT(shadowed.r, 0.2f * reference.r);
    for (int c = 0; c < 3; ++c) {
        EXPECT_NEAR(unshadowed[c], reference[c], 1e-5f) << c;
    }
}

TEST_F(GaussianLightingGpu, glowLightsSplatsAroundIt) {
    const glm::vec3 albedo(0.3f);
    // Facing away from the light at x = 0: no direct light, only the glow.
    std::vector<Gaussian3D> scene;
    for (float x : {0.1f, 0.25f, 0.6f}) {
        scene.push_back(splat({x, 0.0f, 0.0f}, glm::vec3(0.1f, 0.1f, 0.01f), 1.0f, albedo));
    }
    std::vector<SplatMaterial> materials(3);
    for (auto& m : materials) {
        m.normalRoughness = glm::vec4(1.0f, 0.0f, 0.0f, 1.0f);
    }
    LightingRunner runner(vc(), scene, materials);
    LightingFrame frame;
    frame.ambient = glm::vec4(0.0f);
    frame.control.x = 1.0f;
    frame.counts.x = 1;
    frame.lights[0] = {{0.0f, 0.0f, 0.0f, 0.0f}, {1.0f, 0.5f, 0.25f, 2.0f}, {1.0f, 0.0f, 0.0f, 0.5f}};
    const auto colours = runner.run(frame);
    for (size_t i = 0; i < scene.size(); ++i) {
        const float d = scene[i].position[0];
        const float fade = d < 0.5f ? (1.0f - d / 0.5f) * (1.0f - d / 0.5f) : 0.0f;
        const glm::vec3 expected = glm::vec3(1.0f, 0.5f, 0.25f) * 2.0f * fade;
        const glm::vec3 got = decode(colours[i]);
        for (int c = 0; c < 3; ++c) {
            EXPECT_NEAR(got[c], expected[c], 1e-4f) << i << ", " << c;
        }
    }
}

TEST_F(GaussianLightingGpu, relitLanternDiffersWhereLit) {
    if (!std::filesystem::exists(test_scene::kLanternPath)) {
        GTEST_SKIP() << "SPZ scene not found: " << test_scene::kLanternPath;
    }
    const auto scene = test_scene::loadLantern();
    const VkExtent2D extent{640, 360};
    auto target = std::make_shared<OffscreenTarget>(vc(), extent, 1u);
    auto cameraUBO = std::make_shared<CameraUboType>();
    InterfaceCameraOrbit orbit(InterfaceCameraOrbit::UpDirection::Y);
    orbit.initialize(vc());
    test_scene::frameLantern(orbit, scene);
    orbit.setProjectionAspectRatio(float(extent.width) / float(extent.height));

    auto data = std::make_shared<GaussianDataStandard>(vc(), scene.gaussians);
    auto frame = std::make_shared<HostValues<LightingFrame>>(vc(), std::vector<LightingFrame>{LightingFrame{}});
    const auto lighting =
        createGaussianLighting(vc(), data->buffers(), bakeSplatMaterials(scene.gaussians), BufferRef{frame});
    auto render = [&](const GaussianSoABuffers& buffers, const LightingFrame& values) {
        auto splatting = createGaussianSplatting(vc(), GsplatBackend::Compute, target, cameraUBO, buffers);
        ComputeGraph graph(vc(), 1);
        graph.compileFrom(splatting);
        orbit.update(cameraUBO->ubo);
        cameraUBO->update(0);
        frame->set(0, values);
        graph.submitAndWait(vc().getGraphicsQueue(), 0);
        return readImage(vc(), target->getImage(0), extent);
    };
    auto meanAbsDifference = [](const std::vector<uint8_t>& a, const std::vector<uint8_t>& b) {
        double sum = 0.0;
        for (size_t i = 0; i < a.size(); i += 4) {
            for (int c = 0; c < 3; ++c) {
                sum += std::abs(int(a[i + c]) - int(b[i + c]));
            }
        }
        return sum / double(a.size() / 4 * 3);
    };

    const auto plain = render(data->buffers(), LightingFrame{});
    LightingFrame values;
    values.camera = glm::vec4(glm::vec3(cameraUBO->ubo.cameraWorldPos), 0.0f);
    values.ambient = glm::vec4(0.25f);
    values.counts.x = 1;
    values.lights[0] = {{scene.centre.x + 0.15f, scene.centre.y + 0.1f, scene.centre.z + 0.15f, 0.0f},
                        {1.0f, 0.6f, 0.3f, 3.0f},
                        {0.2f, 0.02f, 0.0f, 0.0f}};
    values.control.x = 0.0f;
    const auto unrelit = render(lighting.output, values);
    values.control.x = 1.0f;
    const auto relit = render(lighting.output, values);

    const std::filesystem::path outDir = "build/TestingOutput/gaussian_lighting";
    std::filesystem::create_directories(outDir);
    writePPM(outDir / "plain.ppm", plain, extent);
    writePPM(outDir / "relit.ppm", relit, extent);
    EXPECT_EQ(meanAbsDifference(unrelit, plain), 0.0);
    const double difference = meanAbsDifference(relit, plain);
    std::cout << "relit lantern differs by " << difference << std::endl;
    EXPECT_GT(difference, 3.0);
}
