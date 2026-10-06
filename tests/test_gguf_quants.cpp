// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

/**
 * TESTS:
 * - Half-precision conversion round-trips normal, subnormal, and special values.
 * - bfloat16 values decode to the upper half of the float bits.
 * - Unsupported types and partial blocks are rejected.
 * - F16, BF16, Q8_0, Q3_K, Q4_K, Q5_K, and Q6_K rows decode to gguf-py's reference values.
 **/

#include <cmath>
#include <filesystem>
#include <limits>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "klartraum/gguf/gguf_file.hpp"
#include "klartraum/gguf/gguf_quants.hpp"

using namespace klartraum;

namespace {

const char* kFixture = "tests/data/gguf/quant_blocks.gguf";

} // namespace

TEST(GgufQuantsTest, HalfConversion) {
    EXPECT_EQ(halfToFloat(0x3c00), 1.0f);
    EXPECT_EQ(halfToFloat(0xc000), -2.0f);
    EXPECT_EQ(halfToFloat(0x7bff), 65504.0f);
    EXPECT_EQ(halfToFloat(0x0001), std::ldexp(1.0f, -24));
    EXPECT_EQ(halfToFloat(0x8000), -0.0f);
    EXPECT_TRUE(std::isinf(halfToFloat(0x7c00)));
    EXPECT_TRUE(std::isnan(halfToFloat(0x7e00)));
    for (uint32_t bits = 0; bits < 0x7c00; ++bits) {
        ASSERT_EQ(floatToHalf(halfToFloat(uint16_t(bits))), bits) << bits;
        ASSERT_EQ(floatToHalf(halfToFloat(uint16_t(bits | 0x8000))), bits | 0x8000) << bits;
    }
    EXPECT_EQ(floatToHalf(1.0f + std::ldexp(1.0f, -11)), 0x3c00); // tie rounds to even
    EXPECT_EQ(floatToHalf(1e6f), 0x7c00);
}

TEST(GgufQuantsTest, BFloat16) {
    EXPECT_EQ(bfloat16ToFloat(0x3f80), 1.0f);
    EXPECT_EQ(bfloat16ToFloat(0xc0a0), -5.0f);
}

TEST(GgufQuantsTest, RejectsUnsupportedInput) {
    std::vector<uint8_t> block(256);
    std::vector<float> output(256);
    EXPECT_THROW(dequantizeRow(GgmlType::Q2_K, block.data(), output.data(), 256), std::runtime_error);
    EXPECT_THROW(dequantizeRow(GgmlType::Q4_K, block.data(), output.data(), 128), std::runtime_error);
    EXPECT_TRUE(isSupportedWeightType(GgmlType::Q6_K));
    EXPECT_FALSE(isSupportedWeightType(GgmlType::IQ4_XS));
}

TEST(GgufQuantsTest, MatchesReferenceDequantization) {
    if (!std::filesystem::exists(kFixture))
        GTEST_SKIP() << "Run scripts/qwen_gguf/make_test_fixtures.py";
    GgufFile file(kFixture);
    for (const std::string name : {"F16", "BF16", "Q8_0", "Q3_K", "Q4_K", "Q5_K", "Q6_K"}) {
        SCOPED_TRACE(name);
        const auto& quantized = file.getTensor(name);
        const auto& expected = file.getTensor(name + ".expected");
        ASSERT_EQ(expected.type, GgmlType::F32);
        ASSERT_EQ(quantized.elementCount(), expected.elementCount());
        const auto* reference = reinterpret_cast<const float*>(file.data(expected));
        const uint64_t rowBytes = ggmlRowBytes(quantized.type, quantized.rowLength());
        std::vector<float> row(quantized.rowLength());
        for (uint64_t r = 0; r < quantized.rowCount(); ++r) {
            dequantizeRow(quantized.type, file.data(quantized) + r * rowBytes, row.data(), row.size());
            for (size_t i = 0; i < row.size(); ++i) {
                const float want = reference[r * row.size() + i];
                ASSERT_NEAR(row[i], want, 1e-6f * std::max(1.0f, std::fabs(want))) << "row " << r << " element " << i;
            }
        }
    }
}
