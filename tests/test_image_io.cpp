// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

/**
 * TESTS:
 * - ppmHoldsHeaderAndPixels: writeImage() with a .ppm path writes a binary PPM header and the RGB pixels unchanged
 * - pngHoldsSignatureAndSize: with a .png path it writes a PNG (signature, IHDR with the image's width and height)
 *   smaller than the PPM for a smooth image
 * - pngDecodesToSamePixels: decoding the PNG (chunk CRCs, inflate, undoing every row's filter) gives back the
 *   pixels exactly, for RGB, grey and RGBA images with noisy and smooth rows
 * - bgraDropsAlphaAndSwapsChannels: writeImageBgra() of a BGRA readback writes the pixels as RGB
 * - unknownExtensionThrows: other extensions are rejected
 **/

#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <random>
#include <vector>

#include <zlib.h>

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

// Decodes an 8-bit PNG written by writeImage(): checks every chunk's CRC, inflates the IDAT data and undoes the
// filters (None, Sub, Up, Average, Paeth); returns the pixels and sets width, height and channels.
std::vector<uint8_t> decodePng(const std::vector<uint8_t>& bytes, uint32_t& width, uint32_t& height,
                               uint32_t& channels) {
    std::vector<uint8_t> compressed;
    size_t at = 8;
    while (at + 12 <= bytes.size()) {
        const uint32_t length = bigEndian(bytes, at);
        const std::string type(bytes.begin() + at + 4, bytes.begin() + at + 8);
        const uint32_t crc = uint32_t(::crc32(0, bytes.data() + at + 4, length + 4));
        EXPECT_EQ(crc, bigEndian(bytes, at + 8 + length)) << type;
        if (type == "IHDR") {
            width = bigEndian(bytes, at + 8);
            height = bigEndian(bytes, at + 12);
            EXPECT_EQ(bytes[at + 16], 8) << "bit depth";
            const uint8_t colour = bytes[at + 17];
            channels = colour == 0 ? 1 : colour == 2 ? 3 : colour == 4 ? 2 : 4;
        } else if (type == "IDAT") {
            compressed.insert(compressed.end(), bytes.begin() + at + 8, bytes.begin() + at + 8 + length);
        }
        at += 12 + length;
    }
    const size_t stride = size_t(width) * channels;
    std::vector<uint8_t> filtered((stride + 1) * height);
    uLongf size = uLongf(filtered.size());
    EXPECT_EQ(uncompress(filtered.data(), &size, compressed.data(), uLong(compressed.size())), Z_OK);
    std::vector<uint8_t> pixels(stride * height);
    for (uint32_t y = 0; y < height; ++y) {
        const uint8_t filter = filtered[y * (stride + 1)];
        const uint8_t* in = &filtered[y * (stride + 1) + 1];
        uint8_t* out = &pixels[y * stride];
        const uint8_t* above = y > 0 ? &pixels[(y - 1) * stride] : nullptr;
        for (size_t i = 0; i < stride; ++i) {
            const int a = i >= channels ? out[i - channels] : 0, b = above ? above[i] : 0,
                      c = above && i >= channels ? above[i - channels] : 0;
            int predicted = 0;
            if (filter == 1)
                predicted = a;
            else if (filter == 2)
                predicted = b;
            else if (filter == 3)
                predicted = (a + b) / 2;
            else if (filter == 4) {
                const int p = a + b - c, pa = std::abs(p - a), pb = std::abs(p - b), pc = std::abs(p - c);
                predicted = pa <= pb && pa <= pc ? a : pb <= pc ? b : c;
            }
            out[i] = uint8_t(in[i] + predicted);
        }
    }
    return pixels;
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

TEST(ImageIoTest, pngDecodesToSamePixels) {
    std::filesystem::create_directories(kDirectory);
    std::mt19937 rng(5);
    for (const uint32_t channels : {3u, 1u, 4u}) {
        SCOPED_TRACE(std::to_string(channels) + " channels");
        constexpr uint32_t width = 57, height = 33;
        // Noisy rows and smooth rows, so that different filters win.
        std::vector<uint8_t> pixels(size_t(width) * height * channels);
        for (uint32_t y = 0; y < height; ++y)
            for (size_t i = 0; i < size_t(width) * channels; ++i)
                pixels[y * width * channels + i] = y % 3 == 0 ? uint8_t(rng()) : uint8_t(i * 3 + y * 7);
        const auto path = kDirectory / ("pixels_" + std::to_string(channels) + ".png");
        writeImage(path, pixels.data(), width, height, channels);
        uint32_t w = 0, h = 0, c = 0;
        const auto decoded = decodePng(readBytes(path), w, h, c);
        EXPECT_EQ(w, width);
        EXPECT_EQ(h, height);
        EXPECT_EQ(c, channels);
        EXPECT_EQ(decoded, pixels);
    }
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
