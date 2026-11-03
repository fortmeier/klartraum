// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// Renders the promo title ("klartraum" over "tech demo 000") as white
// Gaussians on black, once per candidate font, to compare how the fonts look
// as splats. Writes one PPM per font to <out-dir>/frames/ and a contact sheet
// of all of them, labelled with the font names, to <out-dir>/contact_sheet.ppm.
//
//   title_font_example [--fonts SpaceGrotesk,Geist,...] [--weight 500]
//                      [--title klartraum] [--subtitle "tech demo 000"]
//                      [--subtitle-font JetBrainsMono] [--subtitle-weight 400]
//                      [--size 0.5] [--subtitle-size 0.2] [--letter-spacing 0]
//                      [--subtitle-letter-spacing 0.05] [--line-spacing 1.2]
//                      [--spacing 0.003] [--splat-scale 0.6] [--jitter 0.35]
//                      [--inset 0.7] [--depth-jitter 0] [--seed 1]
//                      [--width 1920] [--height 1080] [--distance 3]
//                      [--out-dir build/TestingOutput/title_fonts]
//
// Font names refer to the fonts the build downloads to build/fonts/ (see
// cmake/KlartraumFonts.cmake); a name ending in .ttf or .otf is used as a path.
// Sizes and spacings are in world units; the camera looks at the text from
// --distance along +z with a 45 degree vertical field of view.

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "klartraum/computegraph/computegraph.hpp"
#include "klartraum/rendering/gaussian_data_standard.hpp"
#include "klartraum/rendering/gaussian_splatting_factory.hpp"
#include "klartraum/glyph_atlas.hpp"
#include "klartraum/headless_frontend.hpp"
#include "klartraum/interface_camera_orbit.hpp"
#include "klartraum/offscreen_target.hpp"
#include "klartraum/rendering/text_gaussians.hpp"

namespace {

constexpr double kPi = 3.14159265358979323846;

struct Image {
    uint32_t width = 0;
    uint32_t height = 0;
    std::vector<uint8_t> rgb;
};

// Reads an OffscreenTarget image back as RGB; the backends leave it in
// TRANSFER_SRC_OPTIMAL as BGRA.
Image readImage(klartraum::VulkanContext& vc, VkImage image, VkExtent2D extent) {
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
    const uint8_t* bgra = static_cast<const uint8_t*>(data);
    Image result{extent.width, extent.height, std::vector<uint8_t>(size_t(extent.width) * extent.height * 3)};
    for (size_t i = 0, j = 0; i < bytes; i += 4, j += 3) {
        result.rgb[j] = bgra[i + 2];
        result.rgb[j + 1] = bgra[i + 1];
        result.rgb[j + 2] = bgra[i];
    }
    vkUnmapMemory(vc.getDevice(), mem);
    vkFreeMemory(vc.getDevice(), mem, nullptr);
    vkDestroyBuffer(vc.getDevice(), buf, nullptr);
    return result;
}

void writePPM(const std::filesystem::path& path, const Image& image) {
    std::ofstream out(path, std::ios::binary);
    out << "P6\n" << image.width << " " << image.height << "\n255\n";
    out.write(reinterpret_cast<const char*>(image.rgb.data()), std::streamsize(image.rgb.size()));
}

// Box-filtered copy of `image`, `factor` times smaller, placed at (x0, y0) in `sheet`.
void pasteDownscaled(Image& sheet, const Image& image, uint32_t factor, uint32_t x0, uint32_t y0) {
    for (uint32_t y = 0; y < image.height / factor; ++y) {
        for (uint32_t x = 0; x < image.width / factor; ++x) {
            for (int c = 0; c < 3; ++c) {
                uint32_t sum = 0;
                for (uint32_t dy = 0; dy < factor; ++dy) {
                    for (uint32_t dx = 0; dx < factor; ++dx) {
                        sum += image.rgb[(size_t(y * factor + dy) * image.width + x * factor + dx) * 3 + c];
                    }
                }
                sheet.rgb[(size_t(y0 + y) * sheet.width + x0 + x) * 3 + c] = uint8_t(sum / (factor * factor));
            }
        }
    }
}

// Writes `text` in the built-in 5x7 bitmap font, each font pixel `scale` image pixels.
void drawLabel(Image& image, const std::string& text, uint32_t x0, uint32_t y0, uint32_t scale, uint8_t grey) {
    static const std::vector<uint8_t> atlas = klartraum::generateGlyphAtlasBitmap();
    for (size_t n = 0; n < text.size(); ++n) {
        const int cell = klartraum::glyphIndexForChar(static_cast<unsigned char>(text[n]));
        const int cellX = (cell % klartraum::GLYPH_ATLAS_COLS) * klartraum::GLYPH_CELL_WIDTH;
        const int cellY = (cell / klartraum::GLYPH_ATLAS_COLS) * klartraum::GLYPH_CELL_HEIGHT;
        for (int gy = 0; gy < klartraum::GLYPH_PIXEL_HEIGHT; ++gy) {
            for (int gx = 0; gx < klartraum::GLYPH_PIXEL_WIDTH; ++gx) {
                if (atlas[size_t(cellY + gy) * klartraum::GLYPH_ATLAS_WIDTH + cellX + gx] == 0) {
                    continue;
                }
                for (uint32_t sy = 0; sy < scale; ++sy) {
                    for (uint32_t sx = 0; sx < scale; ++sx) {
                        const uint32_t x = x0 + uint32_t(n * klartraum::GLYPH_CELL_WIDTH + gx) * scale + sx;
                        const uint32_t y = y0 + uint32_t(gy) * scale + sy;
                        if (x < image.width && y < image.height) {
                            uint8_t* p = &image.rgb[(size_t(y) * image.width + x) * 3];
                            p[0] = p[1] = p[2] = grey;
                        }
                    }
                }
            }
        }
    }
}

std::string fontPath(const std::string& name) {
    const bool isPath =
        name.size() > 4 && (name.substr(name.size() - 4) == ".ttf" || name.substr(name.size() - 4) == ".otf");
    return isPath ? name : "fonts/" + name + ".ttf";
}

std::string fontLabel(const std::string& name) { return std::filesystem::path(name).stem().string(); }

std::vector<std::string> splitList(const std::string& list) {
    std::vector<std::string> items;
    std::stringstream stream(list);
    std::string item;
    while (std::getline(stream, item, ',')) {
        if (!item.empty()) {
            items.push_back(item);
        }
    }
    return items;
}

} // namespace

int main(int argc, char** argv) {
    std::vector<std::string> fonts = {"SpaceGrotesk", "Geist", "Unbounded", "Syne", "IBMPlexSans", "Fraunces"};
    std::string title = "klartraum";
    std::string subtitle = "tech demo 000";
    std::string subtitleFont = "JetBrainsMono";
    float weight = 500.0f;
    float subtitleWeight = 400.0f;
    float size = 0.5f;
    float subtitleSize = 0.2f;
    float letterSpacing = 0.0f;
    float subtitleLetterSpacing = 0.05f;
    float lineSpacing = 1.2f;
    klartraum::TextSplatParams splat;
    splat.spacing = 0.003f;
    uint32_t width = 1920;
    uint32_t height = 1080;
    double distance = 3.0;
    std::filesystem::path outDir = "build/TestingOutput/title_fonts";

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        const bool hasValue = i + 1 < argc;
        if (arg == "--fonts" && hasValue) {
            fonts = splitList(argv[++i]);
        } else if (arg == "--title" && hasValue) {
            title = argv[++i];
        } else if (arg == "--subtitle" && hasValue) {
            subtitle = argv[++i];
        } else if (arg == "--subtitle-font" && hasValue) {
            subtitleFont = argv[++i];
        } else if (arg == "--weight" && hasValue) {
            weight = std::stof(argv[++i]);
        } else if (arg == "--subtitle-weight" && hasValue) {
            subtitleWeight = std::stof(argv[++i]);
        } else if (arg == "--size" && hasValue) {
            size = std::stof(argv[++i]);
        } else if (arg == "--subtitle-size" && hasValue) {
            subtitleSize = std::stof(argv[++i]);
        } else if (arg == "--letter-spacing" && hasValue) {
            letterSpacing = std::stof(argv[++i]);
        } else if (arg == "--subtitle-letter-spacing" && hasValue) {
            subtitleLetterSpacing = std::stof(argv[++i]);
        } else if (arg == "--line-spacing" && hasValue) {
            lineSpacing = std::stof(argv[++i]);
        } else if (arg == "--spacing" && hasValue) {
            splat.spacing = std::stof(argv[++i]);
        } else if (arg == "--splat-scale" && hasValue) {
            splat.splatScale = std::stof(argv[++i]);
        } else if (arg == "--jitter" && hasValue) {
            splat.jitter = std::stof(argv[++i]);
        } else if (arg == "--inset" && hasValue) {
            splat.inset = std::stof(argv[++i]);
        } else if (arg == "--depth-jitter" && hasValue) {
            splat.depthJitter = std::stof(argv[++i]);
        } else if (arg == "--seed" && hasValue) {
            splat.seed = uint32_t(std::stoul(argv[++i]));
        } else if (arg == "--width" && hasValue) {
            width = uint32_t(std::stoul(argv[++i]));
        } else if (arg == "--height" && hasValue) {
            height = uint32_t(std::stoul(argv[++i]));
        } else if (arg == "--distance" && hasValue) {
            distance = std::stod(argv[++i]);
        } else if (arg == "--out-dir" && hasValue) {
            outDir = argv[++i];
        } else {
            std::cerr << "unknown or incomplete argument: " << arg << std::endl;
            return 1;
        }
    }
    if (fonts.empty() || width < 2 || height < 2 || !(splat.spacing > 0.0f) || !(distance > 0.0)) {
        std::cerr << "need at least one font, an image of at least 2x2, a positive --spacing and --distance"
                  << std::endl;
        return 1;
    }

    std::shared_ptr<klartraum::Font> subFont;
    try {
        subFont = std::make_shared<klartraum::Font>(fontPath(subtitleFont));
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }

    std::filesystem::create_directories(outDir / "frames");

    klartraum::HeadlessFrontend frontend;
    auto& vc = frontend.getKlartraumEngine().getVulkanContext();
    const VkExtent2D extent{width, height};

    // Two columns of half-size panels.
    const uint32_t columns = fonts.size() > 1 ? 2 : 1;
    const uint32_t rows = uint32_t((fonts.size() + columns - 1) / columns);
    Image sheet{columns * (width / 2), rows * (height / 2), {}};
    sheet.rgb.assign(size_t(sheet.width) * sheet.height * 3, 0);

    for (size_t n = 0; n < fonts.size(); ++n) {
        std::shared_ptr<klartraum::Font> font;
        try {
            font = std::make_shared<klartraum::Font>(fontPath(fonts[n]));
        } catch (const std::exception& e) {
            std::cerr << e.what() << std::endl;
            return 1;
        }

        klartraum::TitleLayout layout;
        layout.lineSpacing = lineSpacing;
        layout.lines.push_back({title, font, weight, size, letterSpacing});
        if (!subtitle.empty()) {
            layout.lines.push_back({subtitle, subFont, subtitleWeight, subtitleSize, subtitleLetterSpacing});
        }
        const klartraum::CoverageMask mask = klartraum::rasterizeTitle(layout);
        const std::vector<klartraum::Gaussian3D> splats = klartraum::sampleTextGaussians(mask, splat);
        if (splats.empty()) {
            std::cerr << fonts[n] << ": the title produced no splats" << std::endl;
            return 1;
        }

        auto target = std::make_shared<klartraum::OffscreenTarget>(vc, extent, 1u);
        auto cameraUBO = std::make_shared<klartraum::CameraUboType>();
        klartraum::InterfaceCameraOrbit orbit(klartraum::InterfaceCameraOrbit::UpDirection::Y);
        orbit.initialize(vc);
        orbit.setPosition(glm::vec3(0.0f));
        orbit.setElevation(0.0);
        orbit.setAzimuth(0.5 * kPi); // camera on the +z axis, facing the text
        orbit.setDistance(distance);
        orbit.setProjectionAspectRatio(float(width) / float(height));

        auto model = std::make_shared<klartraum::GaussianDataStandard>(vc, splats);
        auto splatting =
            klartraum::createGaussianSplatting(vc, klartraum::GsplatBackend::Compute, target, cameraUBO, model);
        Image frame;
        {
            klartraum::ComputeGraph graph(vc, 1);
            graph.compileFrom(splatting);
            orbit.update(cameraUBO->ubo);
            cameraUBO->update(0);
            graph.submitAndWait(vc.getGraphicsQueue(), 0);
            frame = readImage(vc, target->getImage(0), extent);
        }
        splatting.reset();
        target.reset();

        const std::string label = fontLabel(fonts[n]);
        writePPM(outDir / "frames" / (label + ".ppm"), frame);
        const uint32_t x0 = uint32_t(n % columns) * (width / 2);
        const uint32_t y0 = uint32_t(n / columns) * (height / 2);
        pasteDownscaled(sheet, frame, 2, x0, y0);
        const uint32_t labelScale = std::max(1u, height / 270);
        drawLabel(sheet, label, x0 + 4 * labelScale, y0 + 4 * labelScale, labelScale, 128);
        std::cout << label << ": " << splats.size() << " splats" << std::endl;
    }

    writePPM(outDir / "contact_sheet.ppm", sheet);
    std::cout << "wrote " << fonts.size() << " frames and a contact sheet to " << outDir.string() << std::endl;
    return 0;
}
