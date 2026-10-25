// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// Maps floats to uints so that unsigned comparison orders them like the floats
// (negative values included): atomicMin over encoded depths keeps the nearest.

const uint DEPTH_NONE = 0xFF800000u; // encodeDepth(+inf): no surface

uint encodeDepth(float depth) {
    uint bits = floatBitsToUint(depth);
    return (bits & 0x80000000u) != 0u ? ~bits : (bits | 0x80000000u);
}

float decodeDepth(uint encoded) {
    return uintBitsToFloat((encoded & 0x80000000u) != 0u ? (encoded & 0x7FFFFFFFu) : ~encoded);
}
