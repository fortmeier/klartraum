// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

/**
 * TESTS:
 * - ppmHoldsHeaderAndPixels: writeImage() with a .ppm path writes a binary PPM header and the RGB pixels unchanged
 * - pngHoldsSignatureAndSize: with a .png path it writes a PNG (signature, IHDR with the image's width and height)
 *   smaller than the PPM for a smooth image
 * - bgraDropsAlphaAndSwapsChannels: writeImageBgra() of a BGRA readback writes the pixels as RGB
 * - unknownExtensionThrows: other extensions are rejected
 **/

#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <vector>

#include "klartraum/image_io.hpp"

using namespace klartraum;

namespace {

const std::filesystem::path kDirectory = "build/TestingOutput/image_io";

std::vector<uint8_t> readBytes(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}

// A smooth 64 x 32 RGB gradient.
std::vector<uint8_t> gradient(uint32_t width, uint32_t height) {
    std::vector<uint8_t> pixels(size_t(width) * height * 3);
    for (uint32_t y = 0; y < height; ++y)
        for (uint32_t x = 0; x < width; ++x) {
            uint8_t* p = &pixels[(size_t(y) * width + x) * 3];
            p[0] = uint8_t(x * 4), p[1] = uint8_t(y * 8), p[2] = 128;
        }
    return pixels;
}

uint32_t bigEndian(const std::vector<uint8_t>& bytes, size_t offset) {
    return uint32_t(bytes[offset]) << 24 | uint32_t(bytes[offset + 1]) << 16 | uint32_t(bytes[offset + 2]) << 8 |
           bytes[offset + 3];
}

} // namespace

TEST(ImageIoTest, ppmHoldsHeaderAndPixels) {
    std::filesystem::create_directories(kDirectory);
    const auto pixels = gradient(64, 32);
    writeImage(kDirectory / "gradient.ppm", pixels.data(), 64, 32, 3);
    const auto bytes = readBytes(kDirectory / "gradient.ppm");
    const std::string header = "P6\n64 32\n255\n";
    ASSERT_EQ(bytes.size(), header.size() + pixels.size());
    EXPECT_EQ(std::string(bytes.begin(), bytes.begin() + header.size()), header);
    EXPECT_TRUE(std::equal(pixels.begin(), pixels.end(), bytes.begin() + header.size()));
}

TEST(ImageIoTest, pngHoldsSignatureAndSize) {
    std::filesystem::create_directories(kDirectory);
    const auto pixels = gradient(64, 32);
    writeImage(kDirectory / "gradient.png", pixels.data(), 64, 32, 3);
    const auto bytes = readBytes(kDirectory / "gradient.png");
    ASSERT_GT(bytes.size(), 33u);
    const std::vector<uint8_t> signature{0x89, 'P', 'N', 'G', 0x0d, 0x0a, 0x1a, 0x0a};
    EXPECT_TRUE(std::equal(signature.begin(), signature.end(), bytes.begin()));
    EXPECT_EQ(std::string(bytes.begin() + 12, bytes.begin() + 16), "IHDR");
    EXPECT_EQ(bigEndian(bytes, 16), 64u);
    EXPECT_EQ(bigEndian(bytes, 20), 32u);
    EXPECT_LT(bytes.size(), pixels.size());
}

TEST(ImageIoTest, bgraDropsAlphaAndSwapsChannels) {
    std::filesystem::create_directories(kDirectory);
    const std::vector<uint8_t> bgra{10, 20, 30, 255, 40, 50, 60, 0};
    writeImageBgra(kDirectory / "bgra.ppm", bgra, 2, 1);
    const auto bytes = readBytes(kDirectory / "bgra.ppm");
    const std::vector<uint8_t> rgb{30, 20, 10, 60, 50, 40};
    ASSERT_EQ(bytes.size(), std::string("P6\n2 1\n255\n").size() + rgb.size());
    EXPECT_TRUE(std::equal(rgb.begin(), rgb.end(), bytes.end() - rgb.size()));
}

TEST(ImageIoTest, unknownExtensionThrows) {
    const auto pixels = gradient(4, 4);
    EXPECT_THROW(writeImage(kDirectory / "image.bmp", pixels.data(), 4, 4, 3), std::invalid_argument);
}
