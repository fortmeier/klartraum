/**
 * TESTS:
 * - rendersGlyphShapeAtExpectedPosition: renders a single 'A' at a known
 *   screen position/scale through the full TextDrawComponent pipeline
 *   (HeadlessFrontend -> RenderPass -> glyph atlas -> shaders) and samples
 *   the center of every glyph cell, asserting each is lit/unlit exactly as
 *   the 'A' bitmap pattern dictates — a quantitative, automatable
 *   per-pixel check rather than a human screenshot comparison
 * - colorTintAndAdvanceAreApplied: renders a two-character string in a
 *   distinct (non-white) color and asserts both glyphs appear at their
 *   expected horizontally-advanced positions, tinted with the requested color
 * - regionsOutsideTextRemainBackground: asserts pixels outside any glyph's
 *   footprint stay at the render pass's clear color (no stray coverage)
 **/

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "klartraum/glyph_atlas.hpp"
#include "klartraum/headless_frontend.hpp"
#include "klartraum/text_draw_component.hpp"

using namespace klartraum;

namespace {

// Minimal BGRA swapchain-image readback (mirrors the helper used by the
// gsplat raster golden-image tests; kept local since those are file-static).
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
    std::filesystem::create_directories("build/TestingOutput");
    std::ofstream f("build/TestingOutput/" + filename, std::ios::binary);
    f << "P6\n" << W << " " << H << "\n255\n";
    for (uint32_t y = 0; y < H; ++y) {
        for (uint32_t x = 0; x < W; ++x) {
            const uint8_t* px = bgra + (static_cast<size_t>(y) * W + x) * 4;
            uint8_t rgb[3] = {px[2], px[1], px[0]};
            f.write(reinterpret_cast<const char*>(rgb), 3);
        }
    }
}

struct Pixel {
    uint8_t r, g, b, a;
};

Pixel samplePixel(const std::vector<uint8_t>& bgra, uint32_t W, uint32_t x, uint32_t y) {
    const uint8_t* px = bgra.data() + (static_cast<size_t>(y) * W + x) * 4;
    return {px[2], px[1], px[0], px[3]};
}

bool isBright(uint8_t channel) { return channel > 128; }
bool isDark(uint8_t channel) { return channel < 64; }

// The 'A' glyph bitmap, exactly as encoded in glyph_atlas.cpp (also checked
// independently in test_glyph_atlas.cpp's atlasBitmapEncodesGlyphPixels).
const char* const kGlyphA[GLYPH_PIXEL_HEIGHT] = {"..X..", ".X.X.", "X...X", "X...X", "XXXXX", "X...X", "X...X"};
// The 'B' glyph bitmap.
const char* const kGlyphB[GLYPH_PIXEL_HEIGHT] = {"XXXX.", "X...X", "X...X", "XXXX.", "X...X", "X...X", "XXXX."};

// Renders `text` through the full pipeline and returns the read-back BGRA image.
std::vector<uint8_t> renderTextToImage(const std::string& text, float x, float y, float scale, float r, float g,
                                       float b, float a, const std::string& ppmName) {
    HeadlessFrontend frontend;
    auto& engine = frontend.getKlartraumEngine();
    auto& vulkanContext = engine.getVulkanContext();

    auto renderpass = engine.createRenderPass();
    auto textComponent = std::make_shared<TextDrawComponent>();
    renderpass->addDrawComponent(textComponent);
    engine.add(renderpass);

    textComponent->setText(text, x, y, scale, r, g, b, a);

    for (int i = 0; i < 3; ++i) {
        engine.step();
    }
    vkQueueWaitIdle(vulkanContext.getGraphicsQueue());

    auto& extent = vulkanContext.getSwapChainExtent();
    std::vector<uint8_t> image = readImageToHost(vulkanContext, vulkanContext.getSwapChainImage(0), extent.width, extent.height);
    writePPM(ppmName, image.data(), extent.width, extent.height);
    return image;
}

// Samples the center of glyph-local cell (col, row) for a glyph quad whose
// top-left pixel is at (originX, originY) and whose pixels are `scale` screen
// pixels wide/tall — i.e. the unambiguous interior of that cell's footprint.
Pixel sampleGlyphCell(const std::vector<uint8_t>& image, uint32_t W, float originX, float originY, float scale,
                      int col, int row) {
    const uint32_t x = static_cast<uint32_t>(originX + (static_cast<float>(col) + 0.5f) * scale);
    const uint32_t y = static_cast<uint32_t>(originY + (static_cast<float>(row) + 0.5f) * scale);
    return samplePixel(image, W, x, y);
}

}  // namespace

TEST(TextRenderingTest, rendersGlyphShapeAtExpectedPosition) {
    const float x = 40.0f;
    const float y = 40.0f;
    const float scale = 6.0f;

    std::vector<uint8_t> image = renderTextToImage("A", x, y, scale, 1.0f, 1.0f, 1.0f, 1.0f, "test_text_glyph_A.ppm");
    const uint32_t imageWidth = 512;  // matches BackendConfig::WIDTH used by HeadlessFrontend

    for (int row = 0; row < GLYPH_PIXEL_HEIGHT; ++row) {
        for (int col = 0; col < GLYPH_PIXEL_WIDTH; ++col) {
            const Pixel px = sampleGlyphCell(image, imageWidth, x, y, scale, col, row);
            const bool expectLit = (kGlyphA[row][col] == 'X');

            if (expectLit) {
                EXPECT_TRUE(isBright(px.r) && isBright(px.g) && isBright(px.b))
                    << "expected lit pixel at glyph cell (col=" << col << ", row=" << row << ") to be bright, got ("
                    << int(px.r) << ", " << int(px.g) << ", " << int(px.b) << ")";
            } else {
                EXPECT_TRUE(isDark(px.r) && isDark(px.g) && isDark(px.b))
                    << "expected unlit pixel at glyph cell (col=" << col << ", row=" << row << ") to be dark, got ("
                    << int(px.r) << ", " << int(px.g) << ", " << int(px.b) << ")";
            }
        }
    }
}

TEST(TextRenderingTest, colorTintAndAdvanceAreApplied) {
    const float x = 60.0f;
    const float y = 100.0f;
    const float scale = 6.0f;
    const float advance = static_cast<float>(GLYPH_CELL_WIDTH) * scale;

    // Pure red so we can confirm the color is propagated (and not, say, white).
    std::vector<uint8_t> image =
        renderTextToImage("AB", x, y, scale, 1.0f, 0.0f, 0.0f, 1.0f, "test_text_color_advance.ppm");
    const uint32_t imageWidth = 512;

    const char* const* glyphs[2] = {kGlyphA, kGlyphB};
    for (int glyphIndex = 0; glyphIndex < 2; ++glyphIndex) {
        const float originX = x + static_cast<float>(glyphIndex) * advance;
        for (int row = 0; row < GLYPH_PIXEL_HEIGHT; ++row) {
            for (int col = 0; col < GLYPH_PIXEL_WIDTH; ++col) {
                const Pixel px = sampleGlyphCell(image, imageWidth, originX, y, scale, col, row);
                const bool expectLit = (glyphs[glyphIndex][row][col] == 'X');

                if (expectLit) {
                    // Red channel bright, green/blue suppressed -> confirms color tint, not just coverage.
                    EXPECT_TRUE(isBright(px.r) && isDark(px.g) && isDark(px.b))
                        << "glyph " << glyphIndex << " cell (col=" << col << ", row=" << row
                        << "): expected red-tinted pixel, got (" << int(px.r) << ", " << int(px.g) << ", "
                        << int(px.b) << ")";
                } else {
                    EXPECT_TRUE(isDark(px.r) && isDark(px.g) && isDark(px.b))
                        << "glyph " << glyphIndex << " cell (col=" << col << ", row=" << row
                        << "): expected background, got (" << int(px.r) << ", " << int(px.g) << ", " << int(px.b)
                        << ")";
                }
            }
        }
    }
}

TEST(TextRenderingTest, regionsOutsideTextRemainBackground) {
    const float x = 200.0f;
    const float y = 200.0f;
    const float scale = 4.0f;

    std::vector<uint8_t> image = renderTextToImage("A", x, y, scale, 1.0f, 1.0f, 1.0f, 1.0f, "test_text_background.ppm");
    const uint32_t imageWidth = 512;
    const uint32_t imageHeight = 384;

    // Sample a grid of points well away from the glyph's footprint
    // (glyph spans roughly [x, x + 5*scale] x [y, y + 7*scale]).
    const std::vector<std::pair<uint32_t, uint32_t>> farPoints = {
        {10, 10}, {imageWidth - 10, 10}, {10, imageHeight - 10}, {imageWidth - 10, imageHeight - 10}, {imageWidth / 2, 10}};

    for (const auto& [px, py] : farPoints) {
        const Pixel pixel = samplePixel(image, imageWidth, px, py);
        EXPECT_TRUE(isDark(pixel.r) && isDark(pixel.g) && isDark(pixel.b))
            << "expected background at (" << px << ", " << py << "), got (" << int(pixel.r) << ", " << int(pixel.g)
            << ", " << int(pixel.b) << ")";
    }
}
