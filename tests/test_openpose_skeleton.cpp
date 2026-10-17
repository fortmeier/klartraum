// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

/**
 * TESTS:
 * - tablesFollowTheOpenPoseConvention: the limb table has 17 limbs over 18 COCO joints, and both colour tables follow
 *   the ControlNet OpenPose palette (limbs in 60 % of the joint colour)
 * - emptySkeletonLeavesBlackImage: a skeleton whose joints are all invisible draws nothing
 * - singleJointDrawsDisc: a lone visible joint is a filled disc of radius 4 in its full colour, with exactly the 49
 *   pixels whose integer offset from the centre lies within the radius
 * - limbHasColorAndWidth: a horizontal limb is drawn in 60 % of its colour, and its height in the middle is
 *   2 * stick width + 1 pixels
 * - invisibleJointHidesItsLimbs: a joint with visibility 0 removes its disc and every limb attached to it
 * - jointBehindCameraIsSkipped: with a perspective camera, a joint behind the camera draws nothing
 * - perspectiveProjectionMatchesCpu: the centroid of each joint's disc matches a CPU projection of the joint with the
 *   same camera matrices to within one pixel
 * - styleScalesWithResolution: by default stick width and joint radius are 4 pixels at 512 and scale with the
 *   shorter side of the target
 * - matchesControlNetReference: two skeletons drawn from the keypoints of data/openpose match the reference drawing
 *   made with ControlNet's draw_bodypose algorithm (scripts/openpose_reference): mean absolute difference below
 *   2/255, fewer than 2 % of the drawn pixels differ, and every limb and joint with at least 20 visible pixels covers
 *   nearly the same pixels (IoU above 0.9)
 * - limbIsCutAtSurfaceEdge: with an occlusion depth buffer whose surface covers half of the image, a limb running
 *   behind it is drawn exactly where the image is uncovered and nowhere else, and so is the joint disc there; the joint
 *   behind the surface disappears
 * - jointDiscIsCutAtSurfaceEdge: a joint disc centred on the edge of a surface in front of it keeps exactly its
 *   uncovered pixels
 * - slopedLimbIsCutWherePassingBehind: a limb from in front of a surface to behind it is cut where it passes behind,
 *   which perspective puts away from the middle of the limb on screen
 * - occlusionToleratesBias: a joint behind the surface by less than the relative bias is still drawn
 * - faceTowardsCameraIsDrawn: with the head facing the camera, nose, eyes and ears are drawn
 * - faceAwayHidesNoseAndEyes: with the head facing away from the camera, nose and eyes and their limbs are hidden,
 *   the ears are drawn
 * - profileHidesFarEarAndEye: seen from the side, the far ear and the far eye are hidden, the near ones are drawn
 * - faceHidingCanBeTurnedOff: with setHideOccludedFace(false), a face turned away is drawn completely
 * - splatOccludesSkeleton: with the depth output of either Gaussian splatting backend, a joint behind an opaque
 *   Gaussian is hidden and a joint in front of it is drawn
 **/

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "klartraum/computegraph/bufferelement.hpp"
#include "klartraum/computegraph/computegraph.hpp"
#include "klartraum/computegraph/renderpass.hpp"
#include "klartraum/gaussian_data_standard.hpp"
#include "klartraum/gaussian_splatting_factory.hpp"
#include "klartraum/headless_frontend.hpp"
#include "klartraum/offscreen_target.hpp"
#include "klartraum/openpose_skeleton_renderer.hpp"
#include "klartraum/vulkan_buffer.hpp"

using namespace klartraum;

namespace {

using JointBuffer = BufferElement<VulkanBuffer<glm::vec4>>;
using JointDepthBuffer = BufferElement<VulkanBuffer<float>>;

struct Rgb {
    uint8_t r, g, b;
    bool operator==(const Rgb& other) const { return r == other.r && g == other.g && b == other.b; }
};

// An RGB image read back from a BGRA target.
struct Image {
    uint32_t width = 0, height = 0;
    std::vector<Rgb> pixels;
    const Rgb& at(uint32_t x, uint32_t y) const { return pixels[size_t(y) * width + x]; }
};

Image readImage(VulkanContext& vc, VkImage image, VkImageLayout layout, VkExtent2D extent) {
    const VkDeviceSize bytes = VkDeviceSize{extent.width} * extent.height * 4;
    VkBuffer buffer;
    VkDeviceMemory memory;
    vc.createBuffer(bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, buffer, memory);
    vc.submitImmediate([&](VkCommandBuffer commandBuffer) {
        VkBufferImageCopy region{};
        region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        region.imageExtent = {extent.width, extent.height, 1};
        vkCmdCopyImageToBuffer(commandBuffer, image, layout, buffer, 1, &region);
    });
    void* data;
    vkMapMemory(vc.getDevice(), memory, 0, bytes, 0, &data);
    const auto* bgra = static_cast<const uint8_t*>(data);
    Image result{extent.width, extent.height, {}};
    result.pixels.resize(size_t(extent.width) * extent.height);
    for (size_t i = 0; i < result.pixels.size(); ++i) {
        result.pixels[i] = {bgra[4 * i + 2], bgra[4 * i + 1], bgra[4 * i]};
    }
    vkUnmapMemory(vc.getDevice(), memory);
    vkDestroyBuffer(vc.getDevice(), buffer, nullptr);
    vkFreeMemory(vc.getDevice(), memory, nullptr);
    return result;
}

Image readPpm(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("Could not open " + path.string());
    }
    std::string magic;
    uint32_t width, height, maximum;
    input >> magic >> width >> height >> maximum;
    input.get();
    if (magic != "P6" || maximum != 255) {
        throw std::runtime_error("Unsupported PPM " + path.string());
    }
    Image image{width, height, std::vector<Rgb>(size_t(width) * height)};
    input.read(reinterpret_cast<char*>(image.pixels.data()), image.pixels.size() * 3);
    return image;
}

size_t countPixels(const Image& image, Rgb color) {
    return static_cast<size_t>(std::count(image.pixels.begin(), image.pixels.end(), color));
}

bool allBlack(const Image& image) { return countPixels(image, {0, 0, 0}) == image.pixels.size(); }

Rgb jointColor(uint32_t joint) {
    const auto& color = openpose::kJointColors[joint];
    return {color[0], color[1], color[2]};
}

Rgb limbColor(uint32_t limb) {
    const auto& color = openpose::kLimbColors[limb];
    return {color[0], color[1], color[2]};
}

// A camera whose matrices are the identity, so joint xy are Vulkan NDC: (-1, -1) is the top-left corner.
CameraMVP identityCamera() { return CameraMVP{glm::mat4(1.0f), glm::mat4(1.0f), glm::mat4(1.0f), glm::vec4(0.0f)}; }

// Joint position in NDC for a normalized image coordinate (0..1, y down).
glm::vec4 normalizedJoint(float x, float y, float visible = 1.0f) {
    return {2.0f * x - 1.0f, 2.0f * y - 1.0f, 0.0f, visible};
}

std::vector<glm::vec4> invisibleSkeleton() { return std::vector<glm::vec4>(openpose::kJointCount, glm::vec4(0.0f)); }

class OpenPoseSkeletonTest : public ::testing::Test {
protected:
    void SetUp() override {
        frontend = std::make_unique<HeadlessFrontend>();
        vc = &frontend->getKlartraumEngine().getVulkanContext();
    }

    // Renders the skeletons in `joints` (kJointCount entries each) into a cleared target.
    Image render(const std::vector<glm::vec4>& joints, const CameraMVP& cameraMatrices, VkExtent2D extent,
                 std::shared_ptr<OpenPoseSkeletonRenderer>* rendererOut = nullptr,
                 std::optional<float> stickWidth = std::nullopt, std::optional<float> jointRadius = std::nullopt,
                 const std::vector<float>* occlusionDepth = nullptr, float relativeBias = 0.02f,
                 bool hideOccludedFace = true) {
        const auto skeletonCount = static_cast<uint32_t>(joints.size() / openpose::kJointCount);
        auto target = std::make_shared<OffscreenTarget>(*vc, extent, 1);
        auto camera = std::make_shared<CameraUboType>();
        camera->ubo = cameraMatrices;
        auto jointBuffer = std::make_shared<JointBuffer>(*vc, static_cast<uint32_t>(joints.size()),
                                                         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        auto renderer = std::make_shared<OpenPoseSkeletonRenderer>(jointBuffer, skeletonCount, extent);
        if (stickWidth) {
            renderer->setStickWidth(*stickWidth);
        }
        if (jointRadius) {
            renderer->setJointRadius(*jointRadius);
        }
        renderer->setHideOccludedFace(hideOccludedFace);
        std::shared_ptr<JointDepthBuffer> depthBuffer;
        if (occlusionDepth) {
            depthBuffer = std::make_shared<JointDepthBuffer>(*vc, static_cast<uint32_t>(occlusionDepth->size()),
                                                             VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
            renderer->setOcclusionDepth(depthBuffer, relativeBias);
        }
        auto renderPass = std::make_shared<RenderPass>(vc->getSwapChainImageFormat(), extent);
        renderPass->setInput(target, 0);
        renderPass->setInput(camera, 1);
        renderPass->addDrawComponent(renderer);
        renderPass->addComputeDependency(jointBuffer);
        if (depthBuffer) {
            renderPass->addComputeDependency(depthBuffer);
        }

        ComputeGraph graph(*vc, 1);
        graph.compileFrom(renderPass);
        jointBuffer->getBuffer(0).memcopyFrom(joints);
        if (depthBuffer) {
            depthBuffer->getBuffer(0).memcopyFrom(*occlusionDepth);
        }
        camera->update(0);
        graph.submitAndWait(vc->getGraphicsQueue(), 0);
        if (rendererOut) {
            *rendererOut = renderer;
        }
        return readImage(*vc, target->getImage(0), *target->getFinalLayoutOverride(), extent);
    }

    std::unique_ptr<HeadlessFrontend> frontend;
    VulkanContext* vc = nullptr;
};

constexpr VkExtent2D kExtent{64, 64};

} // namespace

TEST(OpenPoseSkeletonTables, tablesFollowTheOpenPoseConvention) {
    ASSERT_EQ(openpose::kLimbs.size(), 17u);
    for (const auto& limb : openpose::kLimbs) {
        EXPECT_LT(limb[0], openpose::kJointCount);
        EXPECT_LT(limb[1], openpose::kJointCount);
        EXPECT_NE(limb[0], limb[1]);
    }
    // The first limb joins the neck and the right shoulder, the last one the left eye and the left ear.
    EXPECT_EQ(openpose::kLimbs.front()[0], openpose::Neck);
    EXPECT_EQ(openpose::kLimbs.front()[1], openpose::RightShoulder);
    EXPECT_EQ(openpose::kLimbs.back()[0], openpose::LeftEye);
    EXPECT_EQ(openpose::kLimbs.back()[1], openpose::LeftEar);

    EXPECT_EQ(jointColor(0), (Rgb{255, 0, 0}));
    EXPECT_EQ(jointColor(6), (Rgb{0, 255, 0}));
    EXPECT_EQ(jointColor(12), (Rgb{0, 0, 255}));
    EXPECT_EQ(jointColor(17), (Rgb{255, 0, 85}));
    for (uint32_t limb = 0; limb < openpose::kLimbCount; ++limb) {
        for (int channel = 0; channel < 3; ++channel) {
            EXPECT_EQ(openpose::kLimbColors[limb][channel],
                      static_cast<uint8_t>(static_cast<int>(openpose::kJointColors[limb][channel] * 0.6)));
        }
    }
}

TEST_F(OpenPoseSkeletonTest, emptySkeletonLeavesBlackImage) {
    EXPECT_TRUE(allBlack(render(invisibleSkeleton(), identityCamera(), kExtent)));
}

TEST_F(OpenPoseSkeletonTest, singleJointDrawsDisc) {
    auto joints = invisibleSkeleton();
    // Pixel (32, 20) in OpenCV's convention, i.e. the pixel whose centre is (32.5, 20.5).
    joints[openpose::Nose] = normalizedJoint(32.3f / 64.0f, 20.7f / 64.0f);
    const auto image = render(joints, identityCamera(), kExtent, nullptr, 4.0f, 4.0f);

    EXPECT_EQ(countPixels(image, jointColor(openpose::Nose)), 49u);
    EXPECT_EQ(countPixels(image, {0, 0, 0}), image.pixels.size() - 49u);
    for (int dy = -5; dy <= 5; ++dy) {
        for (int dx = -5; dx <= 5; ++dx) {
            const bool inside = dx * dx + dy * dy <= 16;
            EXPECT_EQ(image.at(32 + dx, 20 + dy) == jointColor(openpose::Nose), inside) << dx << ", " << dy;
        }
    }
}

TEST_F(OpenPoseSkeletonTest, limbHasColorAndWidth) {
    auto joints = invisibleSkeleton();
    // Limb 0 joins neck and right shoulder; only the limb is checked between the discs.
    joints[openpose::Neck] = normalizedJoint(10.0f / 64.0f, 30.0f / 64.0f);
    joints[openpose::RightShoulder] = normalizedJoint(50.0f / 64.0f, 30.0f / 64.0f);
    const auto image = render(joints, identityCamera(), kExtent, nullptr, 4.0f, 4.0f);

    EXPECT_EQ(image.at(30, 30), limbColor(0));
    uint32_t height = 0;
    for (uint32_t y = 0; y < kExtent.height; ++y) {
        height += image.at(30, y) == limbColor(0) ? 1 : 0;
    }
    EXPECT_EQ(height, 9u);
    EXPECT_EQ(image.at(10, 30), jointColor(openpose::Neck));
    EXPECT_EQ(image.at(50, 30), jointColor(openpose::RightShoulder));
}

TEST_F(OpenPoseSkeletonTest, invisibleJointHidesItsLimbs) {
    auto joints = invisibleSkeleton();
    joints[openpose::Neck] = normalizedJoint(0.2f, 0.5f);
    joints[openpose::RightShoulder] = normalizedJoint(0.8f, 0.5f, 0.0f);
    const auto image = render(joints, identityCamera(), kExtent);

    EXPECT_EQ(countPixels(image, limbColor(0)), 0u);
    EXPECT_EQ(countPixels(image, jointColor(openpose::RightShoulder)), 0u);
    EXPECT_GT(countPixels(image, jointColor(openpose::Neck)), 0u);
}

TEST_F(OpenPoseSkeletonTest, jointBehindCameraIsSkipped) {
    CameraMVP camera = identityCamera();
    camera.view = glm::lookAt(glm::vec3(0.0f, 0.0f, 3.0f), glm::vec3(0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    camera.proj = glm::perspective(glm::radians(60.0f), 1.0f, 0.1f, 100.0f);
    auto joints = invisibleSkeleton();
    joints[openpose::Nose] = {0.0f, 0.0f, 5.0f, 1.0f};
    EXPECT_TRUE(allBlack(render(joints, camera, kExtent)));
}

TEST_F(OpenPoseSkeletonTest, perspectiveProjectionMatchesCpu) {
    CameraMVP camera = identityCamera();
    camera.model = glm::translate(glm::mat4(1.0f), glm::vec3(0.1f, -0.2f, 0.0f));
    camera.view = glm::lookAt(glm::vec3(1.0f, 1.5f, 4.0f), glm::vec3(0.0f, 1.0f, 0.0f), glm::vec3(0.0f, -1.0f, 0.0f));
    camera.proj = glm::perspective(glm::radians(50.0f), 1.0f, 0.1f, 100.0f);
    const VkExtent2D extent{128, 128};

    auto joints = invisibleSkeleton();
    const std::map<uint32_t, glm::vec3> placed = {
        {openpose::Nose, {0.0f, 1.7f, 0.0f}},
        {openpose::RightWrist, {-0.6f, 1.0f, 0.2f}},
        {openpose::LeftAnkle, {0.2f, 0.0f, -0.3f}},
    };
    for (const auto& [joint, position] : placed) {
        joints[joint] = glm::vec4(position, 1.0f);
    }
    const auto image = render(joints, camera, extent, nullptr, 1.0f, 3.0f);

    for (const auto& [joint, position] : placed) {
        const glm::vec4 clip = camera.proj * camera.view * camera.model * glm::vec4(position, 1.0f);
        const glm::vec2 expected((clip.x / clip.w * 0.5f + 0.5f) * extent.width,
                                 (clip.y / clip.w * 0.5f + 0.5f) * extent.height);
        glm::vec2 sum(0.0f);
        float count = 0.0f;
        for (uint32_t y = 0; y < extent.height; ++y) {
            for (uint32_t x = 0; x < extent.width; ++x) {
                if (image.at(x, y) == jointColor(joint)) {
                    sum += glm::vec2(x + 0.5f, y + 0.5f);
                    count += 1.0f;
                }
            }
        }
        ASSERT_GT(count, 0.0f) << "joint " << joint;
        const glm::vec2 centroid = sum / count;
        EXPECT_LT(glm::length(centroid - expected), 1.0f) << "joint " << joint;
    }
}

TEST_F(OpenPoseSkeletonTest, styleScalesWithResolution) {
    std::shared_ptr<OpenPoseSkeletonRenderer> renderer;
    render(invisibleSkeleton(), identityCamera(), {256, 128}, &renderer);
    EXPECT_FLOAT_EQ(renderer->getStickWidth(), 1.0f);
    EXPECT_FLOAT_EQ(renderer->getJointRadius(), 1.0f);
    render(invisibleSkeleton(), identityCamera(), {512, 768}, &renderer);
    EXPECT_FLOAT_EQ(renderer->getStickWidth(), 4.0f);
    EXPECT_FLOAT_EQ(renderer->getJointRadius(), 4.0f);
}

TEST_F(OpenPoseSkeletonTest, matchesControlNetReference) {
    const std::filesystem::path directory = "data/openpose";
    const Image reference = readPpm(directory / "reference_256.ppm");

    std::ifstream keypointFile(directory / "reference_keypoints.txt");
    ASSERT_TRUE(keypointFile) << "run the tests from the repository root";
    std::vector<glm::vec4> joints;
    float size = 0.0f, stickWidth = 0.0f, jointRadius = 0.0f;
    std::string line;
    bool header = true;
    while (std::getline(keypointFile, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }
        std::istringstream fields(line);
        if (header) {
            fields >> size >> stickWidth >> jointRadius;
            header = false;
            continue;
        }
        uint32_t person, joint;
        float x, y, visible;
        fields >> person >> joint >> x >> y >> visible;
        joints.resize(std::max<size_t>(joints.size(), (person + 1) * openpose::kJointCount), glm::vec4(0.0f));
        joints[person * openpose::kJointCount + joint] = normalizedJoint(x, y, visible);
    }
    ASSERT_EQ(joints.size(), 2 * openpose::kJointCount);
    ASSERT_EQ(reference.width, static_cast<uint32_t>(size));

    const VkExtent2D extent{reference.width, reference.height};
    const auto image = render(joints, identityCamera(), extent, nullptr, stickWidth, jointRadius);

    std::filesystem::create_directories("build/TestingOutput");
    {
        std::ofstream output("build/TestingOutput/openpose_skeleton_256.ppm", std::ios::binary);
        output << "P6\n" << image.width << " " << image.height << "\n255\n";
        output.write(reinterpret_cast<const char*>(image.pixels.data()), image.pixels.size() * 3);
    }

    double absoluteDifference = 0.0;
    for (size_t i = 0; i < image.pixels.size(); ++i) {
        absoluteDifference += std::abs(int(image.pixels[i].r) - int(reference.pixels[i].r)) +
                              std::abs(int(image.pixels[i].g) - int(reference.pixels[i].g)) +
                              std::abs(int(image.pixels[i].b) - int(reference.pixels[i].b));
    }
    const double meanDifference = absoluteDifference / (3.0 * image.pixels.size());
    EXPECT_LT(meanDifference, 2.0);

    size_t drawn = 0, differing = 0;
    for (size_t i = 0; i < image.pixels.size(); ++i) {
        drawn += reference.pixels[i] == Rgb{0, 0, 0} ? 0 : 1;
        differing += image.pixels[i] == reference.pixels[i] ? 0 : 1;
    }
    EXPECT_LT(double(differing), 0.02 * double(drawn));

    // Every colour a limb or joint is drawn in: the joint colours and the limb colours are all distinct.
    std::vector<Rgb> colors;
    for (uint32_t joint = 0; joint < openpose::kJointCount; ++joint) {
        colors.push_back(jointColor(joint));
    }
    for (uint32_t limb = 0; limb < openpose::kLimbCount; ++limb) {
        colors.push_back(limbColor(limb));
    }
    for (const Rgb& color : colors) {
        // Parts nearly hidden by later ones (e.g. the short limbs between face joints) are too small for an IoU.
        const size_t expected = countPixels(reference, color);
        if (expected < 20) {
            continue;
        }
        size_t intersection = 0, unionCount = 0;
        for (size_t i = 0; i < image.pixels.size(); ++i) {
            const bool a = image.pixels[i] == color;
            const bool b = reference.pixels[i] == color;
            intersection += (a && b) ? 1 : 0;
            unionCount += (a || b) ? 1 : 0;
        }
        const double iou = double(intersection) / double(unionCount);
        EXPECT_GT(iou, 0.9) << "colour " << int(color.r) << " " << int(color.g) << " " << int(color.b) << " covers "
                            << expected << " reference pixels";
    }
}

namespace {

// A perspective camera at z = 3 looking at the origin, in Vulkan clip space.
CameraMVP perspectiveCamera() {
    CameraMVP camera = identityCamera();
    camera.view = glm::lookAt(glm::vec3(0.0f, 0.0f, 3.0f), glm::vec3(0.0f), glm::vec3(0.0f, -1.0f, 0.0f));
    camera.proj = glm::perspective(glm::radians(50.0f), 1.0f, 0.1f, 100.0f);
    return camera;
}

// NDC depth of a world point.
float ndcDepth(const CameraMVP& camera, glm::vec3 position) {
    const glm::vec4 clip = camera.proj * camera.view * camera.model * glm::vec4(position, 1.0f);
    return clip.z / clip.w;
}

} // namespace

namespace {

// A depth buffer with a surface at NDC depth `depth` where `covered(x, y)` holds, and no surface elsewhere.
template <typename Covered>
std::vector<float> partialSurface(float depth, Covered covered) {
    std::vector<float> values(size_t(kExtent.width) * kExtent.height);
    for (uint32_t y = 0; y < kExtent.height; ++y) {
        for (uint32_t x = 0; x < kExtent.width; ++x) {
            values[size_t(y) * kExtent.width + x] = covered(x, y) ? depth : std::numeric_limits<float>::infinity();
        }
    }
    return values;
}

// Number of pixels of `color` where `inside(x, y)` holds.
template <typename Inside>
size_t countPixelsWhere(const Image& image, Rgb color, Inside inside) {
    size_t count = 0;
    for (uint32_t y = 0; y < image.height; ++y) {
        for (uint32_t x = 0; x < image.width; ++x) {
            count += (image.at(x, y) == color && inside(x, y)) ? 1 : 0;
        }
    }
    return count;
}

// Horizontal pixel coordinate of a world point.
float pixelX(const CameraMVP& camera, glm::vec3 position) {
    const glm::vec4 clip = camera.proj * camera.view * camera.model * glm::vec4(position, 1.0f);
    return (clip.x / clip.w * 0.5f + 0.5f) * float(kExtent.width);
}

} // namespace

TEST_F(OpenPoseSkeletonTest, limbIsCutAtSurfaceEdge) {
    const CameraMVP camera = perspectiveCamera();
    auto joints = invisibleSkeleton();
    // Limb 0 joins neck and right shoulder, both behind the plane z = 0.
    joints[openpose::Neck] = {-0.6f, 0.0f, -0.5f, 1.0f};
    joints[openpose::RightShoulder] = {0.6f, 0.0f, -0.5f, 1.0f};
    // The surface covers the half of the image the neck projects into.
    const bool neckOnLeft = pixelX(camera, glm::vec3(joints[openpose::Neck])) < 32.0f;
    auto covered = [&](uint32_t x, uint32_t) { return (x < kExtent.width / 2) == neckOnLeft; };
    const auto depth = partialSurface(ndcDepth(camera, {0.0f, 0.0f, 0.0f}), covered);
    auto uncovered = [&](uint32_t x, uint32_t y) { return !covered(x, y); };

    const auto image = render(joints, camera, kExtent, nullptr, 2.0f, 3.0f, &depth);
    const auto unoccluded = render(joints, camera, kExtent, nullptr, 2.0f, 3.0f);
    for (const Rgb color : {limbColor(0), jointColor(openpose::Neck), jointColor(openpose::RightShoulder)}) {
        EXPECT_EQ(countPixelsWhere(image, color, covered), 0u);
        EXPECT_EQ(countPixelsWhere(image, color, uncovered), countPixelsWhere(unoccluded, color, uncovered));
    }
    EXPECT_GT(countPixelsWhere(image, limbColor(0), uncovered), 0u);
    EXPECT_GT(countPixelsWhere(unoccluded, limbColor(0), covered), 0u);
    EXPECT_EQ(countPixels(image, jointColor(openpose::Neck)), 0u);
}

TEST_F(OpenPoseSkeletonTest, jointDiscIsCutAtSurfaceEdge) {
    const CameraMVP camera = perspectiveCamera();
    auto joints = invisibleSkeleton();
    joints[openpose::Nose] = {0.0f, 0.0f, -0.5f, 1.0f}; // centred in the image, behind the plane z = 0
    auto covered = [](uint32_t x, uint32_t) { return x < kExtent.width / 2; };
    const auto depth = partialSurface(ndcDepth(camera, {0.0f, 0.0f, 0.0f}), covered);
    auto uncovered = [&](uint32_t x, uint32_t y) { return !covered(x, y); };

    const auto image = render(joints, camera, kExtent, nullptr, 2.0f, 5.0f, &depth);
    const auto unoccluded = render(joints, camera, kExtent, nullptr, 2.0f, 5.0f);
    const Rgb nose = jointColor(openpose::Nose);
    EXPECT_GT(countPixelsWhere(unoccluded, nose, covered), 0u);
    EXPECT_EQ(countPixelsWhere(image, nose, covered), 0u);
    EXPECT_GT(countPixelsWhere(image, nose, uncovered), 0u);
    EXPECT_EQ(countPixelsWhere(image, nose, uncovered), countPixelsWhere(unoccluded, nose, uncovered));
}

TEST_F(OpenPoseSkeletonTest, slopedLimbIsCutWherePassingBehind) {
    const CameraMVP camera = perspectiveCamera();
    auto joints = invisibleSkeleton();
    // Limb 0 from in front of the plane z = 0 to behind it; it passes behind at the world origin.
    const glm::vec3 front(-0.6f, 0.0f, 0.5f), back(0.6f, 0.0f, -0.5f), crossing(0.0f);
    joints[openpose::Neck] = glm::vec4(front, 1.0f);
    joints[openpose::RightShoulder] = glm::vec4(back, 1.0f);
    const std::vector<float> depth(size_t(kExtent.width) * kExtent.height, ndcDepth(camera, crossing));

    const float frontX = pixelX(camera, front), backX = pixelX(camera, back), crossingX = pixelX(camera, crossing);
    // Perspective moves the crossing away from the middle of the limb on screen; interpolating depth linearly in
    // screen space would cut there instead.
    ASSERT_GT(std::abs(crossingX - 0.5f * (frontX + backX)), 2.0f);

    const auto image = render(joints, camera, kExtent, nullptr, 2.0f, 2.0f, &depth, 0.0f);
    const float towardsBack = backX > frontX ? 1.0f : -1.0f;
    // Signed distance of a column's centre from the crossing, positive towards the back joint.
    auto beyondCrossing = [&](uint32_t x) { return (float(x) + 0.5f - crossingX) * towardsBack; };
    EXPECT_EQ(countPixelsWhere(image, limbColor(0), [&](uint32_t x, uint32_t) { return beyondCrossing(x) > 1.0f; }),
              0u);
    // Up to the crossing the limb is complete: every column between the front joint's disc and the crossing holds
    // limb pixels.
    for (uint32_t x = 0; x < kExtent.width; ++x) {
        const float fromFront = (float(x) + 0.5f - frontX) * towardsBack;
        if (fromFront > 3.0f && beyondCrossing(x) < -1.0f) {
            EXPECT_GT(countPixelsWhere(image, limbColor(0), [x](uint32_t column, uint32_t) { return column == x; }), 0u)
                << "column " << x;
        }
    }
    EXPECT_EQ(countPixels(image, jointColor(openpose::RightShoulder)), 0u);
    EXPECT_GT(countPixels(image, jointColor(openpose::Neck)), 0u);
}

TEST_F(OpenPoseSkeletonTest, occlusionToleratesBias) {
    const CameraMVP camera = perspectiveCamera();
    const std::vector<float> depth(size_t(kExtent.width) * kExtent.height, ndcDepth(camera, {0.0f, 0.0f, 0.0f}));
    auto joints = invisibleSkeleton();
    // Distance 3.03 behind a surface at distance 3: 1 % behind.
    joints[openpose::Nose] = {0.0f, 0.0f, -0.03f, 1.0f};
    EXPECT_GT(
        countPixels(render(joints, camera, kExtent, nullptr, 2.0f, 3.0f, &depth, 0.02f), jointColor(openpose::Nose)),
        0u);
    EXPECT_EQ(
        countPixels(render(joints, camera, kExtent, nullptr, 2.0f, 3.0f, &depth, 0.005f), jointColor(openpose::Nose)),
        0u);
}

TEST_F(OpenPoseSkeletonTest, splatOccludesSkeleton) {
    // An opaque Gaussian at the origin; the camera at z = 3 looks at it.
    const CameraMVP cameraMatrices = perspectiveCamera();
    Gaussian3D gaussian{};
    gaussian.position = {0.0f, 0.0f, 0.0f};
    gaussian.rotation = {0.0f, 0.0f, 0.0f, 1.0f};
    gaussian.scale = {0.3f, 0.3f, 0.05f};
    gaussian.color = {1.0f, 1.0f, 1.0f};
    gaussian.alpha = 0.99f;

    for (GsplatBackend backend : {GsplatBackend::Compute, GsplatBackend::Raster}) {
        SCOPED_TRACE(backend == GsplatBackend::Compute ? "compute" : "raster");
        auto target = std::make_shared<OffscreenTarget>(*vc, kExtent, 1);
        auto camera = std::make_shared<CameraUboType>();
        camera->ubo = cameraMatrices;
        GsplatConfig config;
        config.outputDepth = true;
        config.maxMod = 16;
        auto splatting = createGaussianSplatting(
            *vc, backend, target, camera, std::make_shared<GaussianDataStandard>(*vc, std::vector{gaussian}), config);
        auto joints = std::make_shared<JointBuffer>(*vc, openpose::kJointCount, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        auto skeletonTarget = std::make_shared<OffscreenTarget>(*vc, kExtent, 1);
        auto renderer = std::make_shared<OpenPoseSkeletonRenderer>(joints, 1, kExtent);
        renderer->setOcclusionDepth(getGaussianSplattingDepth(splatting));
        renderer->setJointRadius(3.0f);
        auto renderPass = std::make_shared<RenderPass>(vc->getSwapChainImageFormat(), kExtent);
        renderPass->setInput(skeletonTarget, 0);
        renderPass->setInput(camera, 1);
        renderPass->addDrawComponent(renderer);
        renderPass->addComputeDependency(joints);
        renderPass->addComputeDependency(splatting);
        ComputeGraph graph(*vc, 1);
        graph.compileFrom(renderPass);

        auto pose = invisibleSkeleton();
        pose[openpose::Nose] = {0.0f, 0.0f, -0.5f, 1.0f}; // behind the Gaussian
        pose[openpose::Neck] = {0.0f, 0.1f, 0.5f, 1.0f};  // in front of it
        joints->getBuffer(0).memcopyFrom(pose);
        camera->update(0);
        graph.submitAndWait(vc->getGraphicsQueue(), 0);
        const Image image =
            readImage(*vc, skeletonTarget->getImage(0), *skeletonTarget->getFinalLayoutOverride(), kExtent);
        EXPECT_EQ(countPixels(image, jointColor(openpose::Nose)), 0u);
        EXPECT_GT(countPixels(image, jointColor(openpose::Neck)), 0u);
    }
}

namespace {

// A head at the origin seen by perspectiveCamera() (at z = 3), five times life size so its keypoints lie far apart on
// screen: ears 40 cm to the sides of the head centre, the face turned towards `facing` (horizontal) with the nose
// 50 cm and the eyes 40 cm in front of the centre.
std::vector<glm::vec4> headFacing(glm::vec3 facing) {
    const glm::vec3 forward = glm::normalize(facing);
    const glm::vec3 side = glm::normalize(glm::cross(glm::vec3(0.0f, 1.0f, 0.0f), forward));
    auto joints = invisibleSkeleton();
    joints[openpose::Nose] = glm::vec4(0.5f * forward, 1.0f);
    joints[openpose::RightEye] = glm::vec4(0.4f * forward - 0.175f * side + glm::vec3(0.0f, 0.15f, 0.0f), 1.0f);
    joints[openpose::LeftEye] = glm::vec4(0.4f * forward + 0.175f * side + glm::vec3(0.0f, 0.15f, 0.0f), 1.0f);
    joints[openpose::RightEar] = glm::vec4(-0.4f * side, 1.0f);
    joints[openpose::LeftEar] = glm::vec4(0.4f * side, 1.0f);
    return joints;
}

} // namespace

TEST_F(OpenPoseSkeletonTest, faceTowardsCameraIsDrawn) {
    const auto image = render(headFacing({0.0f, 0.0f, 1.0f}), perspectiveCamera(), {128, 128}, nullptr, 1.0f, 2.0f);
    for (uint32_t joint :
         {openpose::Nose, openpose::RightEye, openpose::LeftEye, openpose::RightEar, openpose::LeftEar}) {
        EXPECT_GT(countPixels(image, jointColor(joint)), 0u) << "joint " << joint;
    }
}

TEST_F(OpenPoseSkeletonTest, faceAwayHidesNoseAndEyes) {
    const auto image = render(headFacing({0.0f, 0.0f, -1.0f}), perspectiveCamera(), {128, 128}, nullptr, 1.0f, 2.0f);
    for (uint32_t joint : {openpose::Nose, openpose::RightEye, openpose::LeftEye}) {
        EXPECT_EQ(countPixels(image, jointColor(joint)), 0u) << "joint " << joint;
    }
    EXPECT_GT(countPixels(image, jointColor(openpose::RightEar)), 0u);
    EXPECT_GT(countPixels(image, jointColor(openpose::LeftEar)), 0u);
    for (uint32_t limb = 13; limb < openpose::kLimbCount; ++limb) {
        EXPECT_EQ(countPixels(image, limbColor(limb)), 0u) << "limb " << limb;
    }
}

TEST_F(OpenPoseSkeletonTest, profileHidesFarEarAndEye) {
    // Facing +x: the side vector points to -z, so the right ear and eye are on the camera's side (+z).
    const auto joints = headFacing({1.0f, 0.0f, 0.0f});
    ASSERT_GT(joints[openpose::RightEar].z, 0.0f);
    const auto image = render(joints, perspectiveCamera(), {128, 128}, nullptr, 1.0f, 2.0f);
    EXPECT_GT(countPixels(image, jointColor(openpose::RightEar)), 0u);
    EXPECT_GT(countPixels(image, jointColor(openpose::RightEye)), 0u);
    EXPECT_GT(countPixels(image, jointColor(openpose::Nose)), 0u);
    EXPECT_EQ(countPixels(image, jointColor(openpose::LeftEar)), 0u);
    EXPECT_EQ(countPixels(image, jointColor(openpose::LeftEye)), 0u);
}

TEST_F(OpenPoseSkeletonTest, faceHidingCanBeTurnedOff) {
    std::shared_ptr<OpenPoseSkeletonRenderer> renderer;
    const auto joints = headFacing({0.0f, 0.0f, -1.0f});
    const auto image = render(joints, perspectiveCamera(), {128, 128}, &renderer, 1.0f, 2.0f, nullptr, 0.02f, false);
    for (uint32_t joint :
         {openpose::Nose, openpose::RightEye, openpose::LeftEye, openpose::RightEar, openpose::LeftEar}) {
        EXPECT_GT(countPixels(image, jointColor(joint)), 0u) << "joint " << joint;
    }
}
