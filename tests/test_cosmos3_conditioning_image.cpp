// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

/**
 * TESTS:
 * - A PPM written to build/TestingOutput reads back with its size and pixels; a non-PPM file is rejected.
 * - At the target size the image is only mapped to [-1, 1] (p / 127.5 - 1), planar RGB.
 * - A wider image at the target height is centre-cropped without resampling.
 * - An exact 2x downscale uses the antialiased triangle filter, renormalized at the borders.
 * - A 7x5 downscale-and-crop matches diffusers' _preprocess_conditioning_image.
 * - A 3x2 upscale-and-crop matches diffusers' _preprocess_conditioning_image.
 **/

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#include "klartraum/cosmos3/conditioning_image.hpp"

using klartraum::preprocessConditioningImage;
using klartraum::readPpm;
using klartraum::RgbImage;

namespace {

/** Pixel pattern of the diffusers references: byte i of the interleaved image is (37 i + 11) mod 256. */
RgbImage patternImage(uint32_t width, uint32_t height) {
    RgbImage image{width, height, std::vector<uint8_t>(size_t(width) * height * 3)};
    for (size_t i = 0; i < image.pixels.size(); ++i)
        image.pixels[i] = uint8_t((i * 37 + 11) % 256);
    return image;
}

/** Image whose red, green and blue channels all equal @p value(x, y). */
template <typename F>
RgbImage greyImage(uint32_t width, uint32_t height, F value) {
    RgbImage image{width, height, std::vector<uint8_t>(size_t(width) * height * 3)};
    for (uint32_t y = 0; y < height; ++y) {
        for (uint32_t x = 0; x < width; ++x) {
            for (uint32_t c = 0; c < 3; ++c)
                image.pixels[(size_t(y) * width + x) * 3 + c] = uint8_t(value(x, y));
        }
    }
    return image;
}

float level(float pixel) { return pixel / 127.5f - 1.0f; }

void expectTensor(const std::vector<float>& actual, const std::vector<float>& expected) {
    ASSERT_EQ(actual.size(), expected.size());
    for (size_t i = 0; i < expected.size(); ++i)
        EXPECT_NEAR(actual[i], expected[i], 1e-6f) << "element " << i;
}

} // namespace

TEST(Cosmos3ConditioningImageTest, readPpmRoundTrip) {
    const auto directory = std::filesystem::path("build/TestingOutput");
    std::filesystem::create_directories(directory);
    const RgbImage written = patternImage(3, 2);
    {
        std::ofstream file(directory / "cosmos3_conditioning.ppm", std::ios::binary);
        file << "P6\n# comment\n3 2\n255\n";
        file.write(reinterpret_cast<const char*>(written.pixels.data()), std::streamsize(written.pixels.size()));
    }
    const RgbImage read = readPpm(directory / "cosmos3_conditioning.ppm");
    EXPECT_EQ(read.width, 3u);
    EXPECT_EQ(read.height, 2u);
    EXPECT_EQ(read.pixels, written.pixels);

    {
        std::ofstream file(directory / "cosmos3_not_a_ppm.ppm", std::ios::binary);
        file << "P3\n1 1\n255\n0 0 0\n";
    }
    EXPECT_THROW(readPpm(directory / "cosmos3_not_a_ppm.ppm"), std::runtime_error);
    EXPECT_THROW(readPpm(directory / "cosmos3_missing.ppm"), std::runtime_error);
}

TEST(Cosmos3ConditioningImageTest, identitySizeOnlyNormalizes) {
    const RgbImage image = patternImage(2, 2);
    std::vector<float> expected(12);
    for (uint32_t c = 0; c < 3; ++c) {
        for (uint32_t i = 0; i < 4; ++i)
            expected[c * 4 + i] = level(image.pixels[i * 3 + c]);
    }
    expectTensor(preprocessConditioningImage(image, 2, 2), expected);
}

TEST(Cosmos3ConditioningImageTest, widerImageIsCentreCropped) {
    // 8x4 covers 4x4 at scale 1; the crop keeps columns 2..5.
    const RgbImage image = greyImage(8, 4, [](uint32_t x, uint32_t y) { return 10 * x + 100 * (y % 2); });
    std::vector<float> expected;
    for (uint32_t c = 0; c < 3; ++c) {
        for (uint32_t y = 0; y < 4; ++y) {
            for (uint32_t x = 2; x < 6; ++x)
                expected.push_back(level(float(10 * x + 100 * (y % 2))));
        }
    }
    expectTensor(preprocessConditioningImage(image, 4, 4), expected);
}

TEST(Cosmos3ConditioningImageTest, exactDownscaleUsesAntialiasedTriangleFilter) {
    // Columns 0, 70, 140, 210 -> 2 columns. Output 0 is centred at 1.0 with support 2, so the
    // taps at 0, 1, 2 weigh 0.75, 0.75, 0.25 (sum 1.75): (0 * 3 + 70 * 3 + 140) / 7 = 50.
    // Output 1 mirrors it: (70 + 140 * 3 + 210 * 3) / 7 = 160. Constant rows stay constant.
    const RgbImage image = greyImage(4, 4, [](uint32_t x, uint32_t) { return 70 * x; });
    std::vector<float> expected;
    for (uint32_t c = 0; c < 3; ++c) {
        for (uint32_t y = 0; y < 2; ++y) {
            expected.push_back(level(50.0f));
            expected.push_back(level(160.0f));
        }
    }
    expectTensor(preprocessConditioningImage(image, 2, 2), expected);
}

TEST(Cosmos3ConditioningImageTest, downscaleAndCropMatchesDiffusers) {
    // _preprocess_conditioning_image(Image.fromarray(pattern(7, 5)), height=4, width=4)
    const std::vector<float> expected = {
        0.247058868f,  0.349019647f,   0.200000048f,   -0.199999988f, 0.325490236f,   0.427451015f,    0.278431416f,
        -0.12156862f,  0.0431373119f,  -0.145098031f,  0.36470592f,   -0.0352941155f, -0.137254894f,   -0.53725493f,
        0.443137288f,  0.0431373119f,  -0.0901960731f, -0.490196049f, 0.490196109f,   0.0901961327f,   -0.0117647052f,
        -0.411764681f, 0.568627477f,   0.168627501f,   0.0745098591f, -0.325490177f,  -0.00392156839f, -0.113725483f,
        0.152941227f,  -0.247058809f,  -0.396078408f,  -0.29411763f,  0.200000048f,   -0.199999988f,   -0.349019587f,
        -0.247058809f, 0.278431416f,   -0.12156862f,   -0.270588219f, -0.168627441f,  0.36470592f,     -0.0352941155f,
        -0.184313715f, -0.0823529363f, 0.443137288f,   0.0431373119f, -0.105882347f,  -0.00392156839f};
    expectTensor(preprocessConditioningImage(patternImage(7, 5), 4, 4), expected);
}

TEST(Cosmos3ConditioningImageTest, upscaleAndCropMatchesDiffusers) {
    // _preprocess_conditioning_image(Image.fromarray(pattern(3, 2)), height=4, width=4)
    const std::vector<float> expected = {
        -0.694117665f, -0.262745082f,  0.176470637f,   0.607843161f,    -0.545098066f,  -0.105882347f,  0.200000048f,
        0.380392194f,  -0.247058809f,  0.192156911f,   0.254902005f,    -0.0666666627f, -0.0901960731f, 0.34117651f,
        0.278431416f,  -0.29411763f,   -0.403921545f,  0.0274510384f,   -0.0352941155f, -0.607843161f,  -0.254901946f,
        0.176470637f,  0.113725543f,   -0.450980365f,  0.0509804487f,   0.482352972f,   0.411764741f,   -0.152941167f,
        0.200000048f,  0.631372571f,   0.568627477f,   -0.00392156839f, -0.113725483f,  0.317647099f,   0.254902005f,
        -0.31764704f,  -0.0901960731f, 0.0980392694f,  0.0274510384f,   -0.29411763f,   -0.0431372523f, -0.356862724f,
        -0.419607818f, -0.239215672f,  -0.0117647052f, -0.58431375f,    -0.647058845f,  -0.215686262f};
    expectTensor(preprocessConditioningImage(patternImage(3, 2), 4, 4), expected);
}
