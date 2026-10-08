// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

/**
 * TESTS:
 * - classWithLanternScene: VulkanGaussianSplattingRaster (the sort-once +
 *   hardware-rasterization backend, guide §7 step 5 "Composite") loads the
 *   lantern SPZ scene, runs several frames through the full dist -> sort ->
 *   barrier -> rasterizer pipeline with the placeholder point-cloud shaders,
 *   and confirms it submits cleanly across all swapchain paths (no
 *   validation-layer errors — the debug callback throws on VK_ERROR severity)
 *   and that the rendered image is not all-black (the cull/sort/indirect-draw
 *   chain actually produced and drew visible splats end to end). Enables GPU
 *   timestamp profiling and prints the per-stage mean split (dist -> sort ->
 *   render pass) so the raster backend's per-stage cost can be tracked across
 *   perf changes, mirroring GaussianSplattingTest.classWithLanternScene
 * - meshShaderPathMatchesVertexPath: renders the lantern scene through the
 *   optional VK_EXT_mesh_shader draw path (GsplatConfig::useMeshShader) and the
 *   default vertex path and diffs them. The mesh path emits the same quads from
 *   the same precomputed Splat2D records in the same sorted order, so the images
 *   must match within a tight tolerance. When the device lacks mesh shaders the
 *   backend falls back to the vertex path, so the two renders are byte-identical
 *   — verifying the device-support gate / fallback (perf plan R5)
 * - deviceLocalWorkingBuffersRenderAcrossPaths: procedural splats render through device-local
 *   projection, sort and draw buffers with correct culling and depth order on every path
 * - deviceLocalMeshWorkingBuffersRenderOrFallBack: requesting mesh shaders retains device-local
 *   working storage and visible output, including the vertex fallback on unsupported devices
 **/
#include <gtest/gtest.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <set>
#include <string>
#include <vector>

#include <glm/glm.hpp>

#include "klartraum/headless_frontend.hpp"
#include "klartraum/computegraph/imageviewsrc.hpp"
#include "klartraum/vulkan_gaussian_splatting_raster.hpp"
#include "klartraum/gaussian_data_standard.hpp"
#include "klartraum/interface_camera_orbit.hpp"

#include "test_scene.hpp"

using namespace klartraum;

namespace {

// Minimal BGRA swapchain-image readback (mirrors the helper in
// test_gaussian_splatting.cpp; kept local since that one is file-static).
std::vector<uint8_t> readImageToHost(VulkanContext& vc, VkImage image, uint32_t W, uint32_t H) {
    const VkDeviceSize bytes = W * H * 4;
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
    region.imageExtent = {W, H, 1};
    vkCmdCopyImageToBuffer(cmd, image, VK_IMAGE_LAYOUT_GENERAL, buf, 1, &region);
    vkEndCommandBuffer(cmd);
    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    vkQueueSubmit(vc.getGraphicsQueue(), 1, &si, VK_NULL_HANDLE);
    vkQueueWaitIdle(vc.getGraphicsQueue());
    vkFreeCommandBuffers(vc.getDevice(), vc.getCommandPool(), 1, &cmd);
    void* data;
    vkMapMemory(vc.getDevice(), mem, 0, bytes, 0, &data);
    std::vector<uint8_t> result(static_cast<const uint8_t*>(data), static_cast<const uint8_t*>(data) + bytes);
    vkUnmapMemory(vc.getDevice(), mem);
    vkFreeMemory(vc.getDevice(), mem, nullptr);
    vkDestroyBuffer(vc.getDevice(), buf, nullptr);
    return result;
}

// Output goes to build/TestingOutput/ so test artifacts don't clutter the repo root.
void writePPM(const std::string& filename, const uint8_t* bgra, uint32_t W, uint32_t H) {
    const std::filesystem::path outDir = "build/TestingOutput";
    std::filesystem::create_directories(outDir);
    std::ofstream f(outDir / filename, std::ios::binary);
    f << "P6\n" << W << " " << H << "\n255\n";
    for (uint32_t y = 0; y < H; ++y)
        for (uint32_t x = 0; x < W; ++x) {
            const uint8_t* p = bgra + (y * W + x) * 4;
            uint8_t rgb[3] = {p[2], p[1], p[0]};
            f.write(reinterpret_cast<const char*>(rgb), 3);
        }
}

} // namespace

TEST(GaussianSplattingRaster, classWithLanternScene) {
    if (!std::filesystem::exists(test_scene::kLanternPath)) {
        GTEST_SKIP() << "SPZ scene not found: " << test_scene::kLanternPath;
    }
    const auto scene = test_scene::loadLantern();

    HeadlessFrontend frontend;
    auto& engine = frontend.getKlartraumEngine();
    engine.enableProfiling();
    auto& vc = engine.getVulkanContext();

    uint32_t numImages = vc.getNumberOfSwapChainImages();
    VkExtent2D ext = vc.getSwapChainExtent();
    std::vector<VkImageView> views(numImages);
    std::vector<VkImage> imgs(numImages);
    std::vector<VkExtent2D> exts(numImages, ext);
    for (uint32_t i = 0; i < numImages; ++i) {
        views[i] = vc.getImageView(i);
        imgs[i] = vc.getSwapChainImage(i);
    }
    auto imageViewSrc = std::make_shared<ImageViewSrc>(views, imgs, exts);
    for (uint32_t i = 0; i < numImages; ++i)
        imageViewSrc->setWaitFor(i, vc.imageAvailableSemaphoresPerImage[i]);

    auto cameraUBO = std::make_shared<CameraUboType>();
    InterfaceCameraOrbit orbit(InterfaceCameraOrbit::UpDirection::Y);
    orbit.initialize(vc);
    test_scene::frameLantern(orbit, scene);
    orbit.update(cameraUBO->ubo);

    auto model = std::make_shared<GaussianDataStandard>(vc, scene.gaussians);
    auto splatting = vc.create<VulkanGaussianSplattingRaster>(imageViewSrc, cameraUBO, model->buffers());
    engine.add(splatting);

    for (uint32_t i = 0; i < numImages; ++i)
        cameraUBO->update(i);

    const int FRAMES = 5;
    for (int f = 0; f < FRAMES; ++f) {
        engine.step();
        vkQueueWaitIdle(vc.getGraphicsQueue());
    }

    auto pixels = readImageToHost(vc, imgs[0], ext.width, ext.height);
    writePPM("test_gaussian_splatting_raster_render.ppm", pixels.data(), ext.width, ext.height);

    uint8_t maxVal = *std::max_element(pixels.begin(), pixels.end());
    std::cout << "\n  classWithLanternScene (raster backend): image max=" << (int)maxVal << "\n";

    std::cout << "--- GPU profiling (mean over " << FRAMES << " frames) ---\n";
    for (auto& [name, ms] : engine.getProfilingResults())
        std::cout << "  " << name << ": " << ms << " ms\n";

    EXPECT_GT(maxVal, uint8_t(10)) << "Rendered image is all-black — pipeline drew nothing";
}

TEST(GaussianSplattingRaster, meshShaderPathMatchesVertexPath) {
    if (!std::filesystem::exists(test_scene::kLanternPath)) {
        GTEST_SKIP() << "SPZ scene not found: " << test_scene::kLanternPath;
    }
    const auto scene = test_scene::loadLantern();

    // Renders the lantern scene with the given config and returns image 0.
    // Sets meshActuallyUsed to whether the mesh path was actually selected
    // (requested AND device-supported).
    auto render = [&](bool useMeshShader, bool& meshActuallyUsed) -> std::vector<uint8_t> {
        HeadlessFrontend frontend;
        auto& engine = frontend.getKlartraumEngine();
        auto& vc = engine.getVulkanContext();
        meshActuallyUsed = useMeshShader && vc.isMeshShaderSupported();

        uint32_t numImages = vc.getNumberOfSwapChainImages();
        VkExtent2D ext = vc.getSwapChainExtent();
        std::vector<VkImageView> views(numImages);
        std::vector<VkImage> imgs(numImages);
        std::vector<VkExtent2D> exts(numImages, ext);
        for (uint32_t i = 0; i < numImages; ++i) {
            views[i] = vc.getImageView(i);
            imgs[i] = vc.getSwapChainImage(i);
        }
        auto imageViewSrc = std::make_shared<ImageViewSrc>(views, imgs, exts);
        for (uint32_t i = 0; i < numImages; ++i)
            imageViewSrc->setWaitFor(i, vc.imageAvailableSemaphoresPerImage[i]);

        auto cameraUBO = std::make_shared<CameraUboType>();
        InterfaceCameraOrbit orbit(InterfaceCameraOrbit::UpDirection::Y);
        orbit.initialize(vc);
        test_scene::frameLantern(orbit, scene);
        orbit.update(cameraUBO->ubo);

        GsplatConfig config;
        config.useMeshShader = useMeshShader;
        auto model = std::make_shared<GaussianDataStandard>(vc, scene.gaussians);
        auto splatting = vc.create<VulkanGaussianSplattingRaster>(imageViewSrc, cameraUBO, model->buffers(), config);
        engine.add(splatting);

        for (uint32_t i = 0; i < numImages; ++i)
            cameraUBO->update(i);
        for (int f = 0; f < 5; ++f) {
            engine.step();
            vkQueueWaitIdle(vc.getGraphicsQueue());
        }
        return readImageToHost(vc, imgs[0], ext.width, ext.height);
    };

    bool dummy = false, meshUsed = false;
    auto vertexPixels = render(false, dummy);
    auto meshPixels = render(true, meshUsed);
    ASSERT_EQ(vertexPixels.size(), meshPixels.size());

    double sumAbsDiff = 0.0;
    uint32_t maxAbsDiff = 0;
    for (size_t i = 0; i < vertexPixels.size(); ++i) {
        uint32_t d =
            static_cast<uint32_t>(std::abs(static_cast<int>(vertexPixels[i]) - static_cast<int>(meshPixels[i])));
        sumAbsDiff += d;
        maxAbsDiff = std::max(maxAbsDiff, d);
    }
    double meanAbsDiff = sumAbsDiff / static_cast<double>(vertexPixels.size());
    std::cout << "\n  meshShaderPathMatchesVertexPath: meshUsed=" << (meshUsed ? "yes" : "no")
              << " meanAbsDiff=" << meanAbsDiff << " maxAbsDiff=" << maxAbsDiff << "\n";

    if (meshUsed) {
        // Same quads, same Splat2D records, same sorted (back-to-front) order —
        // only the primitive submission path differs, so the result should match
        // closely (a small tolerance covers any rasterization-rule differences).
        EXPECT_LT(meanAbsDiff, 2.0) << "mesh path diverges from the vertex path";
        uint8_t maxVal = *std::max_element(meshPixels.begin(), meshPixels.end());
        EXPECT_GT(maxVal, uint8_t(10)) << "mesh path produced an all-black image";
    } else {
        // No mesh-shader support: useMeshShader falls back to the vertex path, so
        // the two renders are identical — confirms the gate / fallback.
        EXPECT_EQ(meanAbsDiff, 0.0) << "fallback should reproduce the vertex path exactly";
    }
}

namespace {

void expectDeviceLocalProceduralRender(bool requestMesh) {
    HeadlessFrontend frontend;
    auto& engine = frontend.getKlartraumEngine();
    auto& vc = engine.getVulkanContext();
    const uint32_t paths = vc.getNumberOfSwapChainImages();
    const auto extent = vc.getSwapChainExtent();
    auto camera = std::make_shared<CameraUboType>();
    InterfaceCameraOrbit orbit(InterfaceCameraOrbit::UpDirection::Y);
    orbit.initialize(vc);
    orbit.setDistance(5.0f);
    orbit.update(camera->ubo);
    const glm::vec3 cameraPosition = glm::vec3(camera->ubo.cameraWorldPos);
    const glm::vec3 toOrigin = -cameraPosition;

    // Two visible splats lie at different depths; the third is behind the camera.
    std::vector<Gaussian3D> points(3);
    for (auto& point : points) {
        point.rotation = {0, 0, 0, 1};
        point.scale = {0.2f, 0.3f, 0.2f};
        point.color = {1, 0.2f, 0.1f};
        point.alpha = 0.8f;
    }
    points[0].position = {0, 0, 0};
    const auto near = cameraPosition + toOrigin * 0.6f;
    const auto behind = cameraPosition - toOrigin * 2.0f;
    points[1].position = {near.x, near.y, near.z};
    points[2].position = {behind.x, behind.y, behind.z};
    GaussianDataStandard model(vc, points);

    std::vector<VkImageView> views;
    std::vector<VkImage> images;
    for (uint32_t path = 0; path < paths; ++path) {
        views.push_back(vc.getImageView(path));
        images.push_back(vc.getSwapChainImage(path));
    }
    auto imageSource = std::make_shared<ImageViewSrc>(views, images, std::vector<VkExtent2D>(paths, extent));
    for (uint32_t path = 0; path < paths; ++path)
        imageSource->setWaitFor(path, vc.imageAvailableSemaphoresPerImage[path]);
    GsplatConfig config;
    config.shDegree = 0;
    config.numSortWGsCap = 1;
    config.useMeshShader = requestMesh;
    auto renderer = vc.create<VulkanGaussianSplattingRaster>(imageSource, camera, model.buffers(), config);
    engine.add(renderer);

    // Inspect graph-visible resources through their diagnostic names, then verify GPU results.
    std::map<std::string, std::shared_ptr<BufferElementInterface>> buffers;
    std::set<const ComputeGraphElement*> visited;
    std::function<void(ComputeGraphElementPtr)> collect = [&](ComputeGraphElementPtr element) {
        if (!element || !visited.insert(element.get()).second)
            return;
        if (auto buffer = std::dynamic_pointer_cast<BufferElementInterface>(element))
            buffers.emplace(element->getName(), buffer);
        for (const auto& [index, input] : element->getInputs())
            collect(input);
    };
    collect(renderer);
    auto expectDeviceLocal = [&](const std::string& name, auto type) {
        using T = decltype(type);
        auto found = buffers.find(name);
        ASSERT_NE(found, buffers.end()) << name;
        auto element = std::dynamic_pointer_cast<TemplatedBufferElementInterface<VulkanBuffer<T>>>(found->second);
        ASSERT_NE(element, nullptr) << name;
        for (uint32_t path = 0; path < paths; ++path)
            EXPECT_NE(element->getBuffer(path).getMemoryProperties() & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0u)
                << name << ", path " << path;
    };
    for (const char* name : {"RasterKeysA", "RasterIndicesA", "RasterKeysB", "RasterIndicesB", "RasterSortTotalCount"})
        expectDeviceLocal(name, uint32_t{});
    expectDeviceLocal("RasterSplat2D", float{});
    expectDeviceLocal("RasterDrawArgs", VkDrawIndirectCommand{});
    if (requestMesh && vc.isMeshShaderSupported())
        expectDeviceLocal("RasterMeshArgs", VkDrawMeshTasksIndirectCommandEXT{});
    else
        EXPECT_EQ(buffers.count("RasterMeshArgs"), 0u);

    auto indices = std::dynamic_pointer_cast<TemplatedBufferElementInterface<VulkanBuffer<uint32_t>>>(
        buffers.at("RasterIndicesA"));
    auto arguments = std::dynamic_pointer_cast<TemplatedBufferElementInterface<VulkanBuffer<VkDrawIndirectCommand>>>(
        buffers.at("RasterDrawArgs"));
    ASSERT_NE(indices, nullptr);
    ASSERT_NE(arguments, nullptr);
    for (uint32_t path = 0; path < paths; ++path)
        camera->update(path);
    for (uint32_t frame = 0; frame < 2 * paths; ++frame) {
        const uint32_t path = vc.currentFrame % paths;
        engine.step();
        ASSERT_EQ(vkQueueWaitIdle(vc.getGraphicsQueue()), VK_SUCCESS);
        std::vector<VkDrawIndirectCommand> draw(1);
        arguments->getBuffer(path).memcopyTo(draw);
        EXPECT_EQ(draw[0].instanceCount, 2u);
        std::vector<uint32_t> sorted(3);
        indices->getBuffer(path).memcopyTo(sorted);
        EXPECT_EQ(sorted[0], 0u);
        EXPECT_EQ(sorted[1], 1u);
        const auto pixels = readImageToHost(vc, images[path], extent.width, extent.height);
        size_t coloredPixels = 0;
        for (size_t i = 0; i < pixels.size(); i += 4)
            if (pixels[i] || pixels[i + 1] || pixels[i + 2])
                ++coloredPixels;
        EXPECT_GT(coloredPixels, 0u) << "frame " << frame << ", path " << path;
    }
}

} // namespace

TEST(GaussianSplattingRaster, deviceLocalWorkingBuffersRenderAcrossPaths) { expectDeviceLocalProceduralRender(false); }

TEST(GaussianSplattingRaster, deviceLocalMeshWorkingBuffersRenderOrFallBack) {
    expectDeviceLocalProceduralRender(true);
}
