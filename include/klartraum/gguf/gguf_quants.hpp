// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#ifndef KLARTRAUM_GGUF_GGUF_QUANTS_HPP
#define KLARTRAUM_GGUF_GGUF_QUANTS_HPP

#include <cstddef>
#include <cstdint>
#include <cstring>

#include "klartraum/gguf/gguf_file.hpp"

namespace klartraum {

/** @brief IEEE half precision bits to float. */
float halfToFloat(uint16_t bits);

/** @brief Float to IEEE half precision bits (round to nearest even). */
uint16_t floatToHalf(float value);

/** @brief bfloat16 bits to float. */
inline float bfloat16ToFloat(uint16_t bits) {
    const uint32_t word = uint32_t(bits) << 16;
    float value;
    std::memcpy(&value, &word, sizeof(value));
    return value;
}

/** @brief Whether dequantizeRow() and the GPU kernels support @p type. */
bool isSupportedWeightType(GgmlType type);

/**
 * @brief Decodes @p count values of @p type starting at the beginning of
 * @p source (which must start on a block boundary) into @p destination.
 *
 * Supported: F32, F16, BF16, Q8_0, Q3_K, Q4_K, Q5_K, Q6_K. @p count must be
 * a multiple of the type's block size.
 * @throws std::runtime_error For unsupported types or partial blocks.
 */
void dequantizeRow(GgmlType type, const void* source, float* destination, size_t count);

} // namespace klartraum

#endif // KLARTRAUM_GGUF_GGUF_QUANTS_HPP
