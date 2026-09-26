/**
 * TESTS:
 * - constructsAndDestroysOnHeadlessEngine: creates an ImGuiOverlay on a
 *   headless engine, registers and unregisters it, and tears everything down
 *   without a validation error
 * - emptyFrameLeavesSceneUnchanged: renders a scene without an overlay and
 *   again with an overlay that draws nothing; both images must be identical,
 *   i.e. the overlay's LOAD pass passes the frame through unmodified
 * - drawsRectangleAtExpectedPosition: draws a solid rectangle through ImGui's
 *   foreground draw list and asserts pixels inside it have its color while
 *   pixels outside still match the scene rendered without the overlay
 * - drawsImGuiWindowWithBackgroundColor: builds a regular ImGui window with a
 *   fixed position, size and background color and asserts the window's
 *   interior has that color, exercising the backend's pipeline and font
 *   atlas upload
 * - keepsDrawingAfterSwapChainRecreated: calls onSwapChainRecreated() (which
 *   rebuilds framebuffers and semaphores) and asserts the rectangle is still
 *   drawn over the scene afterwards
 * - drawsOverGaussianSplattingRaster: renders the raccoon scene through the
 *   raster Gaussian-splatting backend with a rectangle overlay; inside the
 *   rectangle the overlay color wins, outside the image matches the scene
 *   rendered without the overlay up to the backend's frame-to-frame noise
 **/

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <imgui.h>

#include "klartraum/computegraph/imageviewsrc.hpp"
#include "klartraum/gaussian_data_standard.hpp"
#include "klartraum/gaussian_splatting_factory.hpp"
#include "klartraum/headless_frontend.hpp"
#include "klartraum/imgui_overlay.hpp"
#include "klartraum/interface_camera_orbit.hpp"
#include "klartraum/text_draw_component.hpp"

using namespace klartraum;

namespace {

const std::string kSpzPath = "3rdparty/spz/samples/racoonfamily.spz";

// Reads swapchain image 0 back to host as tightly-packed BGRA bytes (headless
// frames end in VK_IMAGE_LAYOUT_GENERAL).
std::vector<uint8_t> readSwapchainImageToHost(VulkanContext& vc) {
    VkExtent2D ext = vc.getSwapChainExtent();
    const VkDeviceSize bytes = ext.width * ext.height * 4;
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
    region.imageExtent = {ext.width, ext.height, 1};
    vkCmdCopyImageToBuffer(cmd, vc.getSwapChainImage(0), VK_IMAGE_LAYOUT_GENERAL, buf, 1, &region);
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
void writePPM(const std::string& filename, const std::vector<uint8_t>& bgra, uint32_t W, uint32_t H) {
    std::filesystem::create_directories("build/TestingOutput");
    std::ofstream f("build/TestingOutput/" + filename, std::ios::binary);
    f << "P6\n" << W << " " << H << "\n255\n";
    for (size_t i = 0; i < static_cast<size_t>(W) * H; ++i) {
        const uint8_t rgb[3] = {bgra[i * 4 + 2], bgra[i * 4 + 1], bgra[i * 4 + 0]};
        f.write(reinterpret_cast<const char*>(rgb), 3);
    }
}

struct Rect {
    uint32_t x0, y0, x1, y1;
    bool contains(uint32_t x, uint32_t y) const { return x >= x0 && x < x1 && y >= y0 && y < y1; }
};

// Pure green: unambiguous against the scenes used here and exact in UNORM.
const ImU32 kOverlayColor = IM_COL32(0, 255, 0, 255);
const Rect kRect{100, 80, 220, 160};

// Asserts every pixel inside `rect` is pure green (BGRA 0,255,0).
void expectRectFilledWithOverlayColor(const std::vector<uint8_t>& image, uint32_t W, const Rect& rect) {
    int wrong = 0;
    for (uint32_t y = rect.y0; y < rect.y1; ++y) {
        for (uint32_t x = rect.x0; x < rect.x1; ++x) {
            const uint8_t* px = image.data() + (static_cast<size_t>(y) * W + x) * 4;
            if (px[0] != 0 || px[1] != 255 || px[2] != 0) {
                ++wrong;
            }
        }
    }
    EXPECT_EQ(wrong, 0) << "pixels inside the overlay rectangle without the overlay color";
}

// Asserts pixels outside `rect` match the reference image. By default every
// pixel must be identical; `maxChannelDiff`/`maxChangedFraction` admit scenes
// that are not bit-exact from frame to frame: at most that fraction of the
// pixels may differ, each by at most `maxChannelDiff` per channel.
void expectOutsideRectUnchanged(const std::vector<uint8_t>& image, const std::vector<uint8_t>& reference,
                                uint32_t W, uint32_t H, const Rect& rect, int maxChannelDiff = 0,
                                double maxChangedFraction = 0.0) {
    ASSERT_EQ(image.size(), reference.size());
    int changed = 0;
    int largeDiffs = 0;
    int outsidePixels = 0;
    for (uint32_t y = 0; y < H; ++y) {
        for (uint32_t x = 0; x < W; ++x) {
            if (rect.contains(x, y)) continue;
            ++outsidePixels;
            const size_t i = (static_cast<size_t>(y) * W + x) * 4;
            int maxDiff = 0;
            for (int c = 0; c < 3; ++c) {
                maxDiff = std::max(maxDiff, std::abs(int(image[i + c]) - int(reference[i + c])));
            }
            if (maxDiff > 0) ++changed;
            if (maxDiff > maxChannelDiff) ++largeDiffs;
        }
    }
    EXPECT_EQ(largeDiffs, 0) << "pixels outside the overlay rectangle differ from the scene without overlay by more "
                             << "than " << maxChannelDiff << " per channel";
    EXPECT_LE(changed, static_cast<int>(maxChangedFraction * outsidePixels))
        << "too many pixels outside the overlay rectangle differ from the scene without overlay";
}

// Starts an ImGui frame without a platform backend.
void beginHeadlessFrame(ImGuiOverlay& overlay, VulkanContext& vc) {
    auto extent = vc.getSwapChainExtent();
    ImGui::GetIO().DisplaySize = ImVec2(static_cast<float>(extent.width), static_cast<float>(extent.height));
    overlay.newFrame();
}

void drawRect(const Rect& rect) {
    ImGui::GetForegroundDrawList()->AddRectFilled(
        ImVec2(static_cast<float>(rect.x0), static_cast<float>(rect.y0)),
        ImVec2(static_cast<float>(rect.x1), static_cast<float>(rect.y1)), kOverlayColor);
}

// Steps the engine a few frames, building the UI with `buildUi` before each
// (when an overlay is given), and reads back swapchain image 0.
std::vector<uint8_t> renderFrames(KlartraumEngine& engine, ImGuiOverlay* overlay,
                                  const std::function<void()>& buildUi) {
    auto& vc = engine.getVulkanContext();
    for (int f = 0; f < 4; ++f) {
        if (overlay) {
            beginHeadlessFrame(*overlay, vc);
            buildUi();
            overlay->render();
        }
        engine.step();
        vkQueueWaitIdle(vc.getGraphicsQueue());
    }
    return readSwapchainImageToHost(vc);
}

// A render pass with a text component: non-trivial, deterministic content.
void addTextScene(KlartraumEngine& engine) {
    auto renderpass = engine.createRenderPass();
    auto text = std::make_shared<TextDrawComponent>();
    renderpass->addDrawComponent(text);
    engine.add(renderpass);
    // Overlaps kRect, so pixels under the rectangle are not background.
    text->setText("KLARTRAUM", 60.0f, 90.0f, 6.0f, 1.0f, 0.5f, 0.2f, 1.0f);
}

}  // namespace

TEST(ImGuiOverlay, constructsAndDestroysOnHeadlessEngine) {
    HeadlessFrontend frontend;
    auto& engine = frontend.getKlartraumEngine();

    auto overlay = std::make_shared<ImGuiOverlay>(engine.getVulkanContext());
    engine.setOverlay(overlay);
    engine.setOverlay(nullptr);
    overlay.reset();
}

TEST(ImGuiOverlay, emptyFrameLeavesSceneUnchanged) {
    HeadlessFrontend frontend;
    auto& engine = frontend.getKlartraumEngine();
    auto& vc = engine.getVulkanContext();
    addTextScene(engine);

    auto reference = renderFrames(engine, nullptr, [] {});

    auto overlay = std::make_shared<ImGuiOverlay>(vc);
    engine.setOverlay(overlay);
    auto image = renderFrames(engine, overlay.get(), [] {});
    engine.setOverlay(nullptr);
    overlay.reset();

    auto extent = vc.getSwapChainExtent();
    writePPM("test_imgui_overlay_empty.ppm", image, extent.width, extent.height);
    EXPECT_EQ(image, reference);
}

TEST(ImGuiOverlay, drawsRectangleAtExpectedPosition) {
    HeadlessFrontend frontend;
    auto& engine = frontend.getKlartraumEngine();
    auto& vc = engine.getVulkanContext();
    addTextScene(engine);
    auto extent = vc.getSwapChainExtent();

    auto reference = renderFrames(engine, nullptr, [] {});

    auto overlay = std::make_shared<ImGuiOverlay>(vc);
    engine.setOverlay(overlay);
    auto image = renderFrames(engine, overlay.get(), [] { drawRect(kRect); });
    engine.setOverlay(nullptr);
    overlay.reset();

    writePPM("test_imgui_overlay_rect.ppm", image, extent.width, extent.height);
    expectRectFilledWithOverlayColor(image, extent.width, kRect);
    expectOutsideRectUnchanged(image, reference, extent.width, extent.height, kRect);
}

TEST(ImGuiOverlay, drawsImGuiWindowWithBackgroundColor) {
    HeadlessFrontend frontend;
    auto& engine = frontend.getKlartraumEngine();
    auto& vc = engine.getVulkanContext();
    addTextScene(engine);
    auto extent = vc.getSwapChainExtent();

    auto overlay = std::make_shared<ImGuiOverlay>(vc);
    engine.setOverlay(overlay);
    auto image = renderFrames(engine, overlay.get(), [] {
        ImGui::SetNextWindowPos(ImVec2(static_cast<float>(kRect.x0), static_cast<float>(kRect.y0)));
        ImGui::SetNextWindowSize(ImVec2(static_cast<float>(kRect.x1 - kRect.x0),
                                        static_cast<float>(kRect.y1 - kRect.y0)));
        ImGui::PushStyleColor(ImGuiCol_WindowBg, kOverlayColor);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
        ImGui::Begin("Test", nullptr,
                     ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoScrollbar);
        ImGui::End();
        ImGui::PopStyleVar(2);
        ImGui::PopStyleColor();
    });
    engine.setOverlay(nullptr);
    overlay.reset();

    writePPM("test_imgui_overlay_window.ppm", image, extent.width, extent.height);
    expectRectFilledWithOverlayColor(image, extent.width, kRect);
}

TEST(ImGuiOverlay, keepsDrawingAfterSwapChainRecreated) {
    HeadlessFrontend frontend;
    auto& engine = frontend.getKlartraumEngine();
    auto& vc = engine.getVulkanContext();
    addTextScene(engine);
    auto extent = vc.getSwapChainExtent();

    auto reference = renderFrames(engine, nullptr, [] {});

    auto overlay = std::make_shared<ImGuiOverlay>(vc);
    engine.setOverlay(overlay);
    renderFrames(engine, overlay.get(), [] { drawRect(kRect); });
    overlay->onSwapChainRecreated();
    auto image = renderFrames(engine, overlay.get(), [] { drawRect(kRect); });
    engine.setOverlay(nullptr);
    overlay.reset();

    expectRectFilledWithOverlayColor(image, extent.width, kRect);
    expectOutsideRectUnchanged(image, reference, extent.width, extent.height, kRect);
}

TEST(ImGuiOverlay, drawsOverGaussianSplattingRaster) {
    HeadlessFrontend frontend;
    auto& engine = frontend.getKlartraumEngine();
    auto& vc = engine.getVulkanContext();
    auto extent = vc.getSwapChainExtent();

    uint32_t numImages = vc.getNumberOfSwapChainImages();
    std::vector<VkImageView> views(numImages);
    std::vector<VkImage> imgs(numImages);
    std::vector<VkExtent2D> exts(numImages, extent);
    for (uint32_t i = 0; i < numImages; ++i) {
        views[i] = vc.getImageView(i);
        imgs[i] = vc.getSwapChainImage(i);
    }
    auto imageViewSrc = std::make_shared<ImageViewSrc>(views, imgs, exts);
    for (uint32_t i = 0; i < numImages; ++i) {
        imageViewSrc->setWaitFor(i, vc.imageAvailableSemaphoresPerImage[i]);
    }

    auto cameraUBO = std::make_shared<CameraUboType>();
    InterfaceCameraOrbit orbit(InterfaceCameraOrbit::UpDirection::Y);
    orbit.initialize(vc);
    orbit.setAzimuth(0.9f);
    orbit.setElevation(-0.5f);
    orbit.setPosition({-0.5f, 0.0f, 0.5f});
    orbit.setDistance(1.0f);
    orbit.update(cameraUBO->ubo);

    auto model = std::make_shared<GaussianDataStandard>(vc, kSpzPath);
    engine.add(createGaussianSplatting(vc, GsplatBackend::Raster, imageViewSrc, cameraUBO, model));
    for (uint32_t i = 0; i < numImages; ++i) {
        cameraUBO->update(i);
    }

    auto reference = renderFrames(engine, nullptr, [] {});

    auto overlay = std::make_shared<ImGuiOverlay>(vc);
    engine.setOverlay(overlay);
    auto image = renderFrames(engine, overlay.get(), [] { drawRect(kRect); });
    engine.setOverlay(nullptr);
    overlay.reset();

    writePPM("test_imgui_overlay_gsplat_raster.ppm", image, extent.width, extent.height);
    expectRectFilledWithOverlayColor(image, extent.width, kRect);
    // The raster backend is not bit-exact from frame to frame: two renders
    // without overlay already differ in a few pixels by up to ~11 levels.
    expectOutsideRectUnchanged(image, reference, extent.width, extent.height, kRect, 16, 0.001);
}
