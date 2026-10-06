// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#include "klartraum/gguf/gguf_quants.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace klartraum {

float halfToFloat(uint16_t bits) {
    const uint32_t sign = uint32_t(bits >> 15) << 31;
    const uint32_t exponent = (bits >> 10) & 0x1f;
    const uint32_t mantissa = bits & 0x3ff;
    float magnitude;
    if (exponent == 0) {
        magnitude = std::ldexp(float(mantissa), -24); // zero and subnormals
    } else if (exponent == 31) {
        magnitude = mantissa ? std::numeric_limits<float>::quiet_NaN() : std::numeric_limits<float>::infinity();
    } else {
        magnitude = std::ldexp(float(mantissa | 0x400), int(exponent) - 25);
    }
    uint32_t word;
    std::memcpy(&word, &magnitude, sizeof(word));
    word |= sign;
    float value;
    std::memcpy(&value, &word, sizeof(value));
    return value;
}

uint16_t floatToHalf(float value) {
    uint32_t word;
    std::memcpy(&word, &value, sizeof(word));
    const uint16_t sign = uint16_t((word >> 16) & 0x8000);
    const float magnitude = std::fabs(value);
    if (std::isnan(value))
        return uint16_t(sign | 0x7e00);
    if (magnitude >= 65520.0f)
        return uint16_t(sign | 0x7c00);
    if (magnitude < std::ldexp(1.0f, -14)) {
        // Subnormal: units of 2^-24, rounded to nearest even.
        return uint16_t(sign | uint16_t(std::nearbyint(std::ldexp(magnitude, 24))));
    }
    int exponent;
    const float fraction = std::frexp(magnitude, &exponent); // magnitude = fraction * 2^exponent, fraction in [0.5, 1)
    uint32_t mantissa = uint32_t(std::nearbyint(std::ldexp(fraction, 11))); // 11 significant bits
    if (mantissa == 2048) {
        mantissa = 1024;
        ++exponent;
    }
    const uint32_t biased = uint32_t(exponent - 1 + 15);
    if (biased >= 31)
        return uint16_t(sign | 0x7c00);
    return uint16_t(sign | (biased << 10) | (mantissa & 0x3ff));
}

namespace {

uint16_t read16(const uint8_t* bytes) { return uint16_t(bytes[0] | (bytes[1] << 8)); }

// The K-quant types with 6-bit scales (Q4_K, Q5_K) pack eight (scale, min)
// pairs into 12 bytes: pairs 0-3 sit in the low six bits of bytes 0-3 (scale)
// and 4-7 (min); pairs 4-7 take their low four bits from the nibbles of
// bytes 8-11 and their top two bits from the spare top bits of bytes 0-7.
void packedScaleMin(const uint8_t* packed, uint32_t index, uint32_t& scale, uint32_t& minimum) {
    if (index < 4) {
        scale = packed[index] & 63;
        minimum = packed[index + 4] & 63;
    } else {
        scale = (packed[index + 4] & 15) | ((packed[index - 4] >> 6) << 4);
        minimum = (packed[index + 4] >> 4) | ((packed[index] >> 6) << 4);
    }
}

// Each decoder maps an element index within one 256-value block (32 for
// Q8_0) to its value.

// Q8_0, 34 bytes: half scale, 32 signed bytes.
float decodeQ8_0(const uint8_t* block, uint32_t element) {
    return halfToFloat(read16(block)) * float(int8_t(block[2 + element]));
}

// Q4_K, 144 bytes: half scale, half minimum scale, 12 bytes of packed
// (scale, min), 128 bytes of nibbles. Every 64 values share 32 bytes: the
// low nibbles hold the first 32 values, the high nibbles the next 32, each
// 32 values forming one sub-block.
float decodeQ4_K(const uint8_t* block, uint32_t element) {
    const uint32_t group = element / 64, within = element % 64;
    const uint32_t subBlock = 2 * group + within / 32;
    const uint8_t byte = block[16 + 32 * group + within % 32];
    const uint32_t quant = within < 32 ? byte & 15 : byte >> 4;
    uint32_t scale, minimum;
    packedScaleMin(block + 4, subBlock, scale, minimum);
    return halfToFloat(read16(block)) * float(scale) * float(quant) - halfToFloat(read16(block + 2)) * float(minimum);
}

// Q5_K, 176 bytes: as Q4_K with 32 bytes of fifth bits after the scales;
// bit `subBlock` of byte (element % 32) adds 16.
float decodeQ5_K(const uint8_t* block, uint32_t element) {
    const uint32_t group = element / 64, within = element % 64;
    const uint32_t subBlock = 2 * group + within / 32;
    const uint8_t byte = block[48 + 32 * group + within % 32];
    const uint32_t high = (block[16 + within % 32] >> subBlock) & 1;
    const uint32_t quant = (within < 32 ? byte & 15 : byte >> 4) | (high << 4);
    uint32_t scale, minimum;
    packedScaleMin(block + 4, subBlock, scale, minimum);
    return halfToFloat(read16(block)) * float(scale) * float(quant) - halfToFloat(read16(block + 2)) * float(minimum);
}

// Q6_K, 210 bytes: 128 bytes of low nibbles, 64 bytes of 2-bit high parts,
// 16 signed sub-block scales, half scale. Each half of 128 values uses 64
// low-nibble bytes and 32 high bytes; its four quarters take the low/high
// nibbles of the first/second 32 low bytes and successive bit pairs of the
// high bytes. Values are offset by 32; sub-blocks hold 16 values.
float decodeQ6_K(const uint8_t* block, uint32_t element) {
    const uint32_t half = element / 128, within = element % 128;
    const uint32_t quarter = within / 32, lane = within % 32;
    const uint8_t lowByte = block[64 * half + lane + 32 * (quarter & 1)];
    const uint32_t low = (lowByte >> (4 * (quarter >> 1))) & 15;
    const uint32_t high = (block[128 + 32 * half + lane] >> (2 * quarter)) & 3;
    const int8_t scale = int8_t(block[192 + 8 * half + 2 * quarter + lane / 16]);
    return halfToFloat(read16(block + 208)) * float(scale) * float(int(low | (high << 4)) - 32);
}

// Q3_K, 110 bytes: 32 bytes of high-bit masks, 64 bytes of 2-bit values,
// 12 bytes of 6-bit scales (offset by 32), half scale. Each half of 128
// values uses 32 value bytes, its four 32-value groups taking successive bit
// pairs; mask bit (4 * half + group) of byte (element % 32) is set when the
// value is not lowered by 4. Sub-blocks hold 16 values.
float decodeQ3_K(const uint8_t* block, uint32_t element) {
    const uint32_t half = element / 128, within = element % 128;
    const uint32_t group = within / 32, lane = within % 32;
    const uint32_t low = (block[32 + 32 * half + lane] >> (2 * group)) & 3;
    const uint32_t keep = (block[lane] >> (4 * half + group)) & 1;
    const uint32_t subBlock = 8 * half + 2 * group + lane / 16;
    const uint8_t* scales = block + 96;
    const uint32_t scaleLow = subBlock < 8 ? scales[subBlock] & 15 : scales[subBlock - 8] >> 4;
    const uint32_t scaleHigh = (scales[8 + subBlock % 4] >> (2 * (subBlock / 4))) & 3;
    const int scale = int(scaleLow | (scaleHigh << 4)) - 32;
    return halfToFloat(read16(block + 108)) * float(scale) * float(int(low) - (keep ? 0 : 4));
}

using Decoder = float (*)(const uint8_t*, uint32_t);

void decodeBlocks(Decoder decode, GgmlType type, const uint8_t* source, float* destination, size_t count) {
    const uint32_t blockSize = ggmlBlockSize(type), blockBytes = ggmlBlockBytes(type);
    for (size_t block = 0; block < count / blockSize; ++block) {
        for (uint32_t element = 0; element < blockSize; ++element) {
            destination[block * blockSize + element] = decode(source + block * blockBytes, element);
        }
    }
}

} // namespace

bool isSupportedWeightType(GgmlType type) {
    switch (type) {
    case GgmlType::F32:
    case GgmlType::F16:
    case GgmlType::BF16:
    case GgmlType::Q8_0:
    case GgmlType::Q3_K:
    case GgmlType::Q4_K:
    case GgmlType::Q5_K:
    case GgmlType::Q6_K:
        return true;
    default:
        return false;
    }
}

void dequantizeRow(GgmlType type, const void* source, float* destination, size_t count) {
    if (!isSupportedWeightType(type)) {
        throw std::runtime_error(std::string("Unsupported GGUF weight type ") + ggmlTypeName(type));
    }
    if (count % ggmlBlockSize(type) != 0) {
        throw std::runtime_error(std::string("Partial ") + ggmlTypeName(type) + " block");
    }
    const auto* bytes = static_cast<const uint8_t*>(source);
    switch (type) {
    case GgmlType::F32:
        std::memcpy(destination, bytes, count * sizeof(float));
        break;
    case GgmlType::F16:
        for (size_t i = 0; i < count; ++i)
            destination[i] = halfToFloat(read16(bytes + 2 * i));
        break;
    case GgmlType::BF16:
        for (size_t i = 0; i < count; ++i)
            destination[i] = bfloat16ToFloat(read16(bytes + 2 * i));
        break;
    case GgmlType::Q8_0:
        decodeBlocks(decodeQ8_0, type, bytes, destination, count);
        break;
    case GgmlType::Q3_K:
        decodeBlocks(decodeQ3_K, type, bytes, destination, count);
        break;
    case GgmlType::Q4_K:
        decodeBlocks(decodeQ4_K, type, bytes, destination, count);
        break;
    case GgmlType::Q5_K:
        decodeBlocks(decodeQ5_K, type, bytes, destination, count);
        break;
    case GgmlType::Q6_K:
        decodeBlocks(decodeQ6_K, type, bytes, destination, count);
        break;
    default:
        break;
    }
}

} // namespace klartraum
