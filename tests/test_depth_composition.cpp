// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

/**
 * TESTS:
 * - nativePassesRespectDepthInEitherOrder: opaque triangles share depth regardless of pass order
 * - nativeDepthControlsAreIndependent: disabling testing or writing changes occlusion independently
 * - gaussianAndRasterPassesComposeInEitherOrder: both backends and depth modes occlude against ordinary geometry
 * - gaussianDepthControlsAreIndependent: Gaussian testing and writing can each be disabled
 * - withoutAttachmentTheDrawOrderWins: color-only composition works in either order
 * - depthClearsBetweenFramesAndPathsStaySeparate: reused graphs clear depth and maintain independent paths
 * - rejectsMismatchedExtents: color and depth must have the same dimensions
 * - preservesDepthWhenClearingIsDisabled: an attachment can retain occlusion across frames
 * - rejectsMismatchedPathCounts: the attachment and graph must have the same path count
 */

#include <array>
#include <memory>
#include <vector>

#include <gtest/gtest.h>
#include <glm/gtc/matrix_transform.hpp>

#include "klartraum/computegraph/computegraph.hpp"
#include "klartraum/computegraph/renderpass.hpp"
#include "klartraum/rendering/gaussian_data_standard.hpp"
#include "klartraum/rendering/gaussian_splatting_factory.hpp"
#include "klartraum/headless_frontend.hpp"
#include "klartraum/interface_camera_orbit.hpp"
#include "klartraum/offscreen_target.hpp"

using namespace klartraum;

namespace {
constexpr VkExtent2D kExtent{64, 64};

class SolidTriangle : public DrawComponent {
public:
    SolidTriangle(glm::vec4 color, float depth)
        : parameters{color, depth} {}
    ~SolidTriangle() {
        if (!vulkanContext)
            return;
        vkDestroyPipeline(vulkanContext->getDevice(), pipeline, nullptr);
        vkDestroyPipelineLayout(vulkanContext->getDevice(), layout, nullptr);
    }
    void initialize(VulkanContext& vc, VkRenderPass& pass, std::shared_ptr<CameraUboType> camera) override {
        DrawComponent::initialize(vc, pass, camera);
        auto vertex = createShaderModule(readFile("shaders/test/depth_solid.vert.spv"), vc.getDevice());
        auto fragment = createShaderModule(readFile("shaders/test/depth_solid.frag.spv"), vc.getDevice());
        VkPipelineShaderStageCreateInfo stages[2]{};
        for (auto& stage : stages) {
            stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            stage.pName = "main";
        }
        stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
        stages[0].module = vertex;
        stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        stages[1].module = fragment;
        VkPushConstantRange range{VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(Parameters)};
        VkPipelineLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        layoutInfo.pushConstantRangeCount = 1;
        layoutInfo.pPushConstantRanges = &range;
        const auto layoutResult = vkCreatePipelineLayout(vc.getDevice(), &layoutInfo, nullptr, &layout);
        VkPipelineVertexInputStateCreateInfo vertices{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
        VkPipelineInputAssemblyStateCreateInfo assembly{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
        assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        VkViewport viewport{0, 0, float(kExtent.width), float(kExtent.height), 0, 1};
        VkRect2D scissor{{0, 0}, kExtent};
        VkPipelineViewportStateCreateInfo viewportInfo{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
        viewportInfo.viewportCount = viewportInfo.scissorCount = 1;
        viewportInfo.pViewports = &viewport;
        viewportInfo.pScissors = &scissor;
        VkPipelineRasterizationStateCreateInfo raster{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
        raster.polygonMode = VK_POLYGON_MODE_FILL;
        raster.cullMode = VK_CULL_MODE_NONE;
        raster.lineWidth = 1;
        VkPipelineMultisampleStateCreateInfo samples{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
        samples.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        VkPipelineColorBlendAttachmentState color{};
        color.colorWriteMask =
            VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        VkPipelineColorBlendStateCreateInfo blend{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
        blend.attachmentCount = 1;
        blend.pAttachments = &color;
        auto depth = pipelineDepthState();
        VkGraphicsPipelineCreateInfo info{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
        info.stageCount = 2;
        info.pStages = stages;
        info.pVertexInputState = &vertices;
        info.pInputAssemblyState = &assembly;
        info.pViewportState = &viewportInfo;
        info.pRasterizationState = &raster;
        info.pMultisampleState = &samples;
        info.pDepthStencilState = &depth;
        info.pColorBlendState = &blend;
        info.layout = layout;
        info.renderPass = pass;
        const auto result = layoutResult == VK_SUCCESS ? vkCreateGraphicsPipelines(vc.getDevice(), VK_NULL_HANDLE, 1,
                                                                                   &info, nullptr, &pipeline)
                                                       : layoutResult;
        vkDestroyShaderModule(vc.getDevice(), vertex, nullptr);
        vkDestroyShaderModule(vc.getDevice(), fragment, nullptr);
        if (result != VK_SUCCESS)
            throw std::runtime_error("SolidTriangle: failed to create pipeline");
    }
    void recordCommandBuffer(VkCommandBuffer cmd, VkFramebuffer, uint32_t) override {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
        vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(parameters), &parameters);
        vkCmdDraw(cmd, 3, 1, 0, 0);
    }

private:
    struct Parameters {
        glm::vec4 color;
        float depth;
    } parameters;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
};

class DepthComposition : public ::testing::Test {
protected:
    void SetUp() override {
        frontend = std::make_unique<HeadlessFrontend>();
        vc = &frontend->getKlartraumEngine().getVulkanContext();
    }
    struct Scene {
        std::shared_ptr<OffscreenTarget> target;
        std::shared_ptr<CameraUboType> camera;
        std::shared_ptr<DepthAttachment> depth;
        std::unique_ptr<ComputeGraph> graph;
    };
    std::array<uint8_t, 4> center(Scene& scene, uint32_t path = 0) {
        VulkanBuffer<uint8_t> staging(*vc, kExtent.width * kExtent.height * 4);
        vc->submitImmediate([&](VkCommandBuffer cmd) {
            VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
            barrier.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1,
                                 &barrier, 0, nullptr, 0, nullptr);
            VkBufferImageCopy region{};
            region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            region.imageExtent = {kExtent.width, kExtent.height, 1};
            vkCmdCopyImageToBuffer(cmd, scene.target->getImage(path), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                   staging.getBuffer(), 1, &region);
        });
        std::vector<uint8_t> bytes(kExtent.width * kExtent.height * 4);
        staging.memcopyTo(bytes);
        const auto index = (kExtent.height / 2 * kExtent.width + kExtent.width / 2) * 4;
        return {bytes[index], bytes[index + 1], bytes[index + 2], bytes[index + 3]};
    }
    Scene base(uint32_t paths = 1, bool withDepth = true) {
        Scene scene;
        scene.target = std::make_shared<OffscreenTarget>(*vc, kExtent, paths);
        scene.camera = std::make_shared<CameraUboType>();
        InterfaceCameraOrbit orbit(InterfaceCameraOrbit::UpDirection::Y);
        orbit.initialize(*vc);
        orbit.setDistance(1);
        orbit.setAzimuth(1.57079632679f);
        orbit.setElevation(0);
        orbit.setProjectionAspectRatio(1);
        orbit.update(scene.camera->ubo);
        if (withDepth)
            scene.depth = std::make_shared<DepthAttachment>(*vc, kExtent, paths);
        scene.graph = std::make_unique<ComputeGraph>(*vc, paths);
        return scene;
    }
    std::shared_ptr<RenderPass> triangle(Scene& scene, std::shared_ptr<ImageViewSrc> source, float depth,
                                         glm::vec4 color, DepthState state = {}) {
        auto pass = std::make_shared<RenderPass>(vc->getSwapChainImageFormat(), kExtent);
        pass->setInput(source, 0);
        pass->setInput(scene.camera, 1);
        pass->setLoadExisting(true);
        pass->setDepthAttachment(scene.depth, state);
        pass->addDrawComponent(std::make_shared<SolidTriangle>(color, depth));
        return pass;
    }
    Scene hybrid(GsplatBackend backend, GaussianDepthMode mode, bool meshFirst, bool meshNear, DepthState gs = {},
                 DepthState geometry = {}, bool withDepth = true, uint32_t paths = 1, bool preserveDepth = false) {
        auto scene = base(paths, withDepth);
        if (scene.depth)
            scene.depth->setClear(!preserveDepth);
        const auto clip = scene.camera->ubo.proj * scene.camera->ubo.view * glm::vec4(0, 0, 0, 1);
        const float z = clip.z / clip.w;
        const float meshZ = meshNear ? z - 0.05f : (1.0f + z) * 0.5f;
        Gaussian3D gaussian{};
        gaussian.rotation = {0, 0, 0, 1};
        gaussian.scale = {0.1f, 0.1f, 0.1f};
        gaussian.color = {-0.5f, -0.5f, 1.0f};
        gaussian.alpha = 1;
        auto model = std::make_shared<GaussianDataStandard>(*vc, std::vector<Gaussian3D>{gaussian});
        GsplatConfig config;
        config.maxMod = 32;
        config.depthAttachment = scene.depth;
        config.depthState = gs;
        config.depthMode = mode;
        std::shared_ptr<ImageViewSrc> source = scene.target;
        if (meshFirst)
            source = triangle(scene, source, meshZ, {1, 0, 0, 1}, geometry);
        auto splats = createGaussianSplatting(*vc, backend, source, scene.camera, model, config);
        ComputeGraphElementPtr root = splats;
        if (!meshFirst)
            root = triangle(scene, splats, meshZ, {1, 0, 0, 1}, geometry);
        scene.graph->compileFrom(root);
        return scene;
    }
    void submit(Scene& scene, uint32_t path = 0) {
        scene.camera->update(path);
        scene.graph->submitAndWait(vc->getGraphicsQueue(), path);
    }
    void expectRed(const std::array<uint8_t, 4>& pixel) { EXPECT_GT(pixel[2], pixel[0]) << "BGRA"; }
    void expectBlue(const std::array<uint8_t, 4>& pixel) { EXPECT_GT(pixel[0], pixel[2]) << "BGRA"; }
    std::unique_ptr<HeadlessFrontend> frontend;
    VulkanContext* vc = nullptr;
};
const GsplatBackend backends[]{GsplatBackend::Compute, GsplatBackend::Raster};
const GaussianDepthMode modes[]{GaussianDepthMode::AccumulatedOpacity, GaussianDepthMode::NearestOpaqueSplat};
} // namespace

TEST_F(DepthComposition, nativePassesRespectDepthInEitherOrder) {
    for (bool nearFirst : {false, true}) {
        auto scene = base();
        auto first = triangle(scene, scene.target, nearFirst ? 0.3f : 0.7f,
                              nearFirst ? glm::vec4(1, 0, 0, 1) : glm::vec4(0, 0, 1, 1));
        auto last =
            triangle(scene, first, nearFirst ? 0.7f : 0.3f, nearFirst ? glm::vec4(0, 0, 1, 1) : glm::vec4(1, 0, 0, 1));
        scene.graph->compileFrom(last);
        submit(scene);
        expectRed(center(scene));
    }
}

TEST_F(DepthComposition, nativeDepthControlsAreIndependent) {
    for (bool disableTesting : {false, true}) {
        auto scene = base();
        auto first = triangle(scene, scene.target, 0.3f, {1, 0, 0, 1}, {true, disableTesting});
        auto last = triangle(scene, first, 0.7f, {0, 0, 1, 1}, {!disableTesting, true});
        scene.graph->compileFrom(last);
        submit(scene);
        expectBlue(center(scene));
    }
}

TEST_F(DepthComposition, gaussianAndRasterPassesComposeInEitherOrder) {
    for (auto backend : backends)
        for (auto mode : modes)
            for (bool first : {false, true})
                for (bool near : {false, true}) {
                    SCOPED_TRACE(::testing::Message()
                                 << int(backend) << "/" << int(mode) << "/" << first << "/" << near);
                    auto scene = hybrid(backend, mode, first, near);
                    submit(scene);
                    if (near)
                        expectRed(center(scene));
                    else
                        expectBlue(center(scene));
                }
}

TEST_F(DepthComposition, gaussianDepthControlsAreIndependent) {
    for (auto backend : backends)
        for (auto mode : modes) {
            auto noTest = hybrid(backend, mode, true, true, {false, true});
            submit(noTest);
            expectBlue(center(noTest));
            auto noWrite = hybrid(backend, mode, false, false, {true, false});
            submit(noWrite);
            expectRed(center(noWrite));
            auto noGeometryTest = hybrid(backend, mode, false, false, {}, {false, true});
            submit(noGeometryTest);
            expectRed(center(noGeometryTest));
        }
}

TEST_F(DepthComposition, withoutAttachmentTheDrawOrderWins) {
    for (auto backend : backends)
        for (bool first : {false, true}) {
            auto scene = hybrid(backend, GaussianDepthMode::BackendDefault, first, true, {}, {}, false);
            submit(scene);
            if (first)
                expectBlue(center(scene));
            else
                expectRed(center(scene));
        }
}

TEST_F(DepthComposition, depthClearsBetweenFramesAndPathsStaySeparate) {
    for (auto backend : backends)
        for (auto mode : modes) {
            auto scene = hybrid(backend, mode, false, false, {}, {}, true, 2);
            submit(scene, 0);
            expectBlue(center(scene, 0));
            scene.camera->ubo.model = glm::translate(glm::mat4(1), glm::vec3(1000, 0, 0));
            submit(scene, 1);
            expectRed(center(scene, 1));
            expectBlue(center(scene, 0));
            submit(scene, 0);
            expectRed(center(scene, 0));
        }
}

TEST_F(DepthComposition, rejectsMismatchedExtents) {
    auto scene = base();
    auto wrong = std::make_shared<DepthAttachment>(*vc, VkExtent2D{32, 32}, 1);
    auto pass = triangle(scene, scene.target, 0.5f, {1, 0, 0, 1});
    pass->setDepthAttachment(wrong);
    EXPECT_THROW(scene.graph->compileFrom(pass), std::invalid_argument);
}

TEST_F(DepthComposition, preservesDepthWhenClearingIsDisabled) {
    for (auto backend : backends)
        for (auto mode : modes) {
            auto scene = hybrid(backend, mode, false, false, {}, {}, true, 1, true);
            submit(scene);
            expectBlue(center(scene));
            scene.camera->ubo.model = glm::translate(glm::mat4(1), glm::vec3(1000, 0, 0));
            submit(scene);
            const auto pixel = center(scene);
            EXPECT_EQ(pixel[0], 0);
            EXPECT_EQ(pixel[1], 0);
            EXPECT_EQ(pixel[2], 0);
        }
}

TEST_F(DepthComposition, rejectsMismatchedPathCounts) {
    auto scene = base();
    auto wrong = std::make_shared<DepthAttachment>(*vc, kExtent, 2);
    auto pass = triangle(scene, scene.target, 0.5f, {1, 0, 0, 1});
    pass->setDepthAttachment(wrong);
    EXPECT_THROW(scene.graph->compileFrom(pass), std::invalid_argument);
}
