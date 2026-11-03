// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

/**
 * TESTS:
 * - fontLoadsFromAssetRoot: "fonts/SpaceGrotesk.ttf" (downloaded by the build) loads through the
 *   asset root, reports its family name and its "wght" axis; a missing file throws
 * - setAxisRejectsUnknownAxis: setting "wght" succeeds, an axis the font does not have is rejected
 * - spaceRasterizesToEmptyMask: a line with only a space has no coverage
 * - singleGlyphHasExpectedShape: "l" rasterizes to a tall, narrow stroke
 * - measureLineAddsLetterSpacing: letter spacing widens a line by (letters - 1) * spacing * size,
 *   and the line width scales with the size
 * - decodesUtf8: UTF-8 text is decoded into code points (one, two, three and four byte sequences);
 *   broken sequences become U+FFFD
 * - rasterizesNonAsciiLetters: "ü" is drawn as its own glyph, "u" with the two dots (more coverage
 *   above the u), and the line is one letter wide
 * - heavierWeightCoversMore: "klartraum" at weight 700 covers clearly more pixels than at 300
 * - titleIsCentred: the ink of a two-line title is centred on the origin, and the second line lies
 *   below the first
 * - splatsLieInsideCoverage: every splat sits where the mask is covered, on the z = 0 plane, white
 *   and opaque as requested
 * - splatCountScalesWithDensity: halving the spacing gives about four times as many splats
 * - splatsAreDeterministic: the same seed gives the same splats, another seed different ones
 * - renderedTitleMatchesMask: for every candidate font the title rendered front-on through the
 *   splatting backend at 1920x1080 covers the same pixels as the FreeType mask projected into the
 *   image (IoU >= 0.90), upright and neither bolder nor thinner than the font; the frames go to
 *   build/TestingOutput/text_gaussians/
 **/

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "klartraum/computegraph/computegraph.hpp"
#include "klartraum/rendering/gaussian_data_standard.hpp"
#include "klartraum/rendering/gaussian_splatting_factory.hpp"
#include "klartraum/headless_frontend.hpp"
#include "klartraum/interface_camera_orbit.hpp"
#include "klartraum/offscreen_target.hpp"
#include "klartraum/rendering/text_gaussians.hpp"

using namespace klartraum;

namespace {

constexpr double kPi = 3.14159265358979323846;

// Loads a font the build downloads to build/fonts/, or returns nullptr when
// the download was switched off or failed.
std::shared_ptr<Font> loadFont(const std::string& name) {
    try {
        return std::make_shared<Font>("fonts/" + name + ".ttf");
    } catch (const std::exception&) {
        return nullptr;
    }
}

#define REQUIRE_FONT(var, name)                                                                                        \
    auto var = loadFont(name);                                                                                         \
    if (!var) {                                                                                                        \
        GTEST_SKIP() << "font not available: " << name << " (configure with KLARTRAUM_DOWNLOAD_FONTS=ON)";             \
    }

TitleLayout singleLine(const std::shared_ptr<Font>& font, const std::string& text, float weight = 0.0f) {
    TitleLayout layout;
    layout.lines.push_back({text, font, weight, 1.0f, 0.0f});
    layout.pixelsPerUnit = 256.0f;
    return layout;
}

struct InkBox {
    int minX = 0, minY = 0, maxX = -1, maxY = -1;
    int width() const { return maxX - minX + 1; }
    int height() const { return maxY - minY + 1; }
};

InkBox inkBox(const CoverageMask& mask) {
    InkBox box{mask.width, mask.height, -1, -1};
    for (int y = 0; y < mask.height; ++y) {
        for (int x = 0; x < mask.width; ++x) {
            if (mask.at(x, y) >= 128) {
                box.minX = std::min(box.minX, x);
                box.minY = std::min(box.minY, y);
                box.maxX = std::max(box.maxX, x);
                box.maxY = std::max(box.maxY, y);
            }
        }
    }
    return box;
}

double coverageSum(const CoverageMask& mask) {
    double sum = 0.0;
    for (uint8_t c : mask.coverage) {
        sum += c / 255.0;
    }
    return sum;
}

// Reads an OffscreenTarget image back as tightly-packed BGRA bytes; the
// backends leave it in TRANSFER_SRC_OPTIMAL.
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

    void* data;
    vkMapMemory(vc.getDevice(), mem, 0, bytes, 0, &data);
    std::vector<uint8_t> result(static_cast<const uint8_t*>(data), static_cast<const uint8_t*>(data) + bytes);
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

TEST(TextGaussians, fontLoadsFromAssetRoot) {
    REQUIRE_FONT(font, "SpaceGrotesk");
    EXPECT_EQ(font->getFamilyName(), "Space Grotesk");
    EXPECT_TRUE(font->hasAxis("wght"));
    EXPECT_THROW(Font("fonts/DoesNotExist.ttf"), std::runtime_error);
}

TEST(TextGaussians, setAxisRejectsUnknownAxis) {
    REQUIRE_FONT(font, "SpaceGrotesk");
    EXPECT_TRUE(font->setAxis("wght", 500.0f));
    EXPECT_FALSE(font->hasAxis("WONK"));
    EXPECT_FALSE(font->setAxis("WONK", 1.0f));
}

TEST(TextGaussians, spaceRasterizesToEmptyMask) {
    REQUIRE_FONT(font, "SpaceGrotesk");
    const CoverageMask mask = rasterizeTitle(singleLine(font, " "));
    EXPECT_EQ(coverageSum(mask), 0.0);
}

TEST(TextGaussians, singleGlyphHasExpectedShape) {
    REQUIRE_FONT(font, "SpaceGrotesk");
    const CoverageMask mask = rasterizeTitle(singleLine(font, "l"));
    const InkBox box = inkBox(mask);
    ASSERT_GT(box.width(), 0);
    // An "l" is a vertical stem about as tall as the ascender (~0.7 em) and a
    // fraction of that wide.
    EXPECT_GT(box.height(), 3 * box.width());
    EXPECT_NEAR(box.height() / mask.pixelsPerUnit, 0.7f, 0.1f);
}

TEST(TextGaussians, measureLineAddsLetterSpacing) {
    REQUIRE_FONT(font, "SpaceGrotesk");
    TextLine line{"klartraum", font, 0.0f, 1.0f, 0.0f};
    const float plain = measureLine(line);
    EXPECT_GT(plain, 2.0f);
    EXPECT_LT(plain, 6.0f);

    line.letterSpacing = 0.1f;
    EXPECT_NEAR(measureLine(line), plain + 8 * 0.1f, 1e-3f);

    line.letterSpacing = 0.0f;
    line.size = 2.0f;
    EXPECT_NEAR(measureLine(line), 2.0f * plain, 0.02f * plain);
}

TEST(TextGaussians, decodesUtf8) {
    EXPECT_EQ(decodeUtf8("ab"), (std::vector<char32_t>{U'a', U'b'}));
    EXPECT_EQ(decodeUtf8("Leimk\xC3\xBChler"),
              (std::vector<char32_t>{U'L', U'e', U'i', U'm', U'k', 0xFC, U'h', U'l', U'e', U'r'}));
    EXPECT_EQ(decodeUtf8("\xE2\x80\x94"), (std::vector<char32_t>{0x2014}));
    EXPECT_EQ(decodeUtf8("\xF0\x9F\x8C\x99"), (std::vector<char32_t>{0x1F319}));
    EXPECT_EQ(decodeUtf8("a\xC3"), (std::vector<char32_t>{U'a', 0xFFFD}));
    EXPECT_EQ(decodeUtf8("\xFF"
                         "b"),
              (std::vector<char32_t>{0xFFFD, U'b'}));
}

TEST(TextGaussians, rasterizesNonAsciiLetters) {
    REQUIRE_FONT(font, "JetBrainsMono");
    const double u = coverageSum(rasterizeTitle(singleLine(font, "u")));
    const double umlaut = coverageSum(rasterizeTitle(singleLine(font, "\xC3\xBC")));
    EXPECT_GT(umlaut, 1.1 * u);
    // A monospace font: one letter, however many bytes it takes.
    EXPECT_NEAR(measureLine({"\xC3\xBC", font, 0.0f, 1.0f, 0.0f}), measureLine({"u", font, 0.0f, 1.0f, 0.0f}), 1e-4f);
}

TEST(TextGaussians, heavierWeightCoversMore) {
    REQUIRE_FONT(font, "SpaceGrotesk");
    const double light = coverageSum(rasterizeTitle(singleLine(font, "klartraum", 300.0f)));
    const double bold = coverageSum(rasterizeTitle(singleLine(font, "klartraum", 700.0f)));
    EXPECT_GT(light, 0.0);
    EXPECT_GT(bold, 1.3 * light);
}

TEST(TextGaussians, titleIsCentred) {
    REQUIRE_FONT(font, "SpaceGrotesk");
    REQUIRE_FONT(mono, "JetBrainsMono");
    TitleLayout layout;
    layout.lines.push_back({"klartraum", font, 500.0f, 1.0f, 0.0f});
    layout.lines.push_back({"tech demo 000", mono, 400.0f, 0.4f, 0.0f});
    layout.pixelsPerUnit = 256.0f;
    const CoverageMask mask = rasterizeTitle(layout);

    // Ink bounding box in world units.
    const InkBox box = inkBox(mask);
    const glm::vec2 topLeft = mask.pixelCentre(box.minX, box.minY);
    const glm::vec2 bottomRight = mask.pixelCentre(box.maxX, box.maxY);
    const glm::vec2 centre = 0.5f * (topLeft + bottomRight);
    const float tolerance = 2.0f / mask.pixelsPerUnit;
    EXPECT_NEAR(centre.x, 0.0f, tolerance);
    EXPECT_NEAR(centre.y, 0.0f, tolerance);

    // The rows of the first line's ink lie above those of the second line:
    // somewhere between them is a row without ink.
    bool gap = false;
    for (int y = box.minY; y <= box.maxY && !gap; ++y) {
        bool empty = true;
        for (int x = 0; x < mask.width && empty; ++x) {
            empty = mask.at(x, y) < 128;
        }
        gap = empty;
    }
    EXPECT_TRUE(gap);
    // The lower part of the ink is the narrower subtitle.
    int widestBottom = 0;
    for (int y = box.maxY - int(0.2f * mask.pixelsPerUnit); y <= box.maxY; ++y) {
        int minX = mask.width, maxX = -1;
        for (int x = 0; x < mask.width; ++x) {
            if (mask.at(x, y) >= 128) {
                minX = std::min(minX, x);
                maxX = std::max(maxX, x);
            }
        }
        widestBottom = std::max(widestBottom, maxX - minX + 1);
    }
    EXPECT_LT(widestBottom, box.width());
}

TEST(TextGaussians, splatsLieInsideCoverage) {
    REQUIRE_FONT(font, "SpaceGrotesk");
    const CoverageMask mask = rasterizeTitle(singleLine(font, "klartraum", 500.0f));
    TextSplatParams params;
    params.spacing = 0.01f;
    const auto splats = sampleTextGaussians(mask, params);
    ASSERT_GT(splats.size(), 100u);

    const float white = colorToSh0(1.0f);
    for (const Gaussian3D& g : splats) {
        EXPECT_GE(mask.sample(g.position[0], g.position[1]), params.threshold);
        EXPECT_EQ(g.position[2], 0.0f);
        EXPECT_FLOAT_EQ(g.scale[0], params.splatScale * params.spacing);
        EXPECT_FLOAT_EQ(g.color[0], white);
        EXPECT_FLOAT_EQ(g.alpha, 1.0f);
    }
    // 0.5 + C0 * sh0 is what the shaders compute for degree 0.
    EXPECT_NEAR(0.5f + 0.282095f * white, 1.0f, 1e-5f);
}

TEST(TextGaussians, splatCountScalesWithDensity) {
    REQUIRE_FONT(font, "SpaceGrotesk");
    const CoverageMask mask = rasterizeTitle(singleLine(font, "klartraum", 500.0f));
    TextSplatParams params;
    params.spacing = 0.01f;
    const size_t coarse = sampleTextGaussians(mask, params).size();
    params.spacing = 0.005f;
    const size_t fine = sampleTextGaussians(mask, params).size();
    const double ratio = double(fine) / double(coarse);
    EXPECT_GT(ratio, 3.5);
    EXPECT_LT(ratio, 4.5);
}

TEST(TextGaussians, splatsAreDeterministic) {
    REQUIRE_FONT(font, "SpaceGrotesk");
    const CoverageMask mask = rasterizeTitle(singleLine(font, "klartraum", 500.0f));
    TextSplatParams params;
    params.spacing = 0.01f;
    const auto a = sampleTextGaussians(mask, params);
    const auto b = sampleTextGaussians(mask, params);
    params.seed = 2;
    const auto c = sampleTextGaussians(mask, params);
    ASSERT_EQ(a.size(), b.size());
    bool samePositions = true;
    for (size_t i = 0; i < a.size(); ++i) {
        samePositions = samePositions && a[i].position == b[i].position;
    }
    EXPECT_TRUE(samePositions);
    bool differs = a.size() != c.size();
    for (size_t i = 0; i < std::min(a.size(), c.size()) && !differs; ++i) {
        differs = a[i].position != c[i].position;
    }
    EXPECT_TRUE(differs);
}

TEST(TextGaussians, renderedTitleMatchesMask) {
    const std::vector<std::string> fonts = {"SpaceGrotesk", "Geist",    "Unbounded",    "Syne",
                                            "IBMPlexSans",  "Fraunces", "JetBrainsMono"};
    REQUIRE_FONT(mono, "JetBrainsMono");

    const VkExtent2D extent{1920, 1080};
    const double distance = 3.0;
    // The orbit camera's vertical field of view is 45 degrees.
    const double focal = 0.5 * extent.height / std::tan(0.5 * glm::radians(45.0));

    const std::filesystem::path outDir = "build/TestingOutput/text_gaussians";
    std::filesystem::create_directories(outDir);

    HeadlessFrontend frontend;
    auto& vc = frontend.getKlartraumEngine().getVulkanContext();

    for (const std::string& name : fonts) {
        SCOPED_TRACE(name);
        auto font = loadFont(name);
        if (!font) {
            ADD_FAILURE() << "font not available: " << name;
            continue;
        }
        TitleLayout layout;
        layout.lines.push_back({"klartraum", font, 500.0f, 0.5f, 0.0f});
        layout.lines.push_back({"tech demo 000", mono, 400.0f, 0.2f, 0.05f});
        const CoverageMask mask = rasterizeTitle(layout);
        TextSplatParams params;
        params.spacing = 0.003f;
        const auto splats = sampleTextGaussians(mask, params);

        auto target = std::make_shared<OffscreenTarget>(vc, extent, 1u);
        auto cameraUBO = std::make_shared<CameraUboType>();
        InterfaceCameraOrbit orbit(InterfaceCameraOrbit::UpDirection::Y);
        orbit.initialize(vc);
        orbit.setPosition(glm::vec3(0.0f));
        orbit.setElevation(0.0);
        orbit.setAzimuth(0.5 * kPi); // camera on the +z axis, looking at the text
        orbit.setDistance(distance);
        orbit.setProjectionAspectRatio(float(extent.width) / float(extent.height));

        auto model = std::make_shared<GaussianDataStandard>(vc, splats);
        auto splatting = createGaussianSplatting(vc, GsplatBackend::Compute, target, cameraUBO, model);
        std::vector<uint8_t> pixels;
        {
            ComputeGraph graph(vc, 1);
            graph.compileFrom(splatting);
            orbit.update(cameraUBO->ubo);
            cameraUBO->update(0);
            graph.submitAndWait(vc.getGraphicsQueue(), 0);
            pixels = readImage(vc, target->getImage(0), extent);
        }
        splatting.reset();
        target.reset();
        writePPM(outDir / (name + ".ppm"), pixels, extent);

        // Project each pixel centre onto the z = 0 plane and compare the
        // rendered brightness with the mask coverage there; world +y is up.
        size_t both = 0, either = 0, missing = 0, extra = 0;
        for (uint32_t y = 0; y < extent.height; ++y) {
            for (uint32_t x = 0; x < extent.width; ++x) {
                const float wx = float((x + 0.5 - 0.5 * extent.width) * distance / focal);
                const float wy = float((0.5 * extent.height - (y + 0.5)) * distance / focal);
                const bool expected = mask.sample(wx, wy) >= 0.5f;
                const size_t i = (size_t(y) * extent.width + x) * 4;
                const bool rendered = (int(pixels[i]) + pixels[i + 1] + pixels[i + 2]) / 3 >= 128;
                both += expected && rendered;
                missing += expected && !rendered;
                extra += rendered && !expected;
                either += expected || rendered;
            }
        }
        ASSERT_GT(either, 0u);
        const double iou = double(both) / double(either);
        // Letters drawn bolder or thinner than the font show up as many more
        // extra than missing pixels, or the other way round.
        const double weightBias = (double(extra) - double(missing)) / double(both);
        std::cout << name << ": " << splats.size() << " splats, IoU " << iou << ", weight bias " << weightBias
                  << std::endl;
        // Measured on Apple M4 (MoltenVK and KosmicKrisp): IoU 0.92-0.96, weight bias within +-0.02.
        EXPECT_GE(iou, 0.90);
        EXPECT_LT(std::abs(weightBias), 0.04);
    }
}
