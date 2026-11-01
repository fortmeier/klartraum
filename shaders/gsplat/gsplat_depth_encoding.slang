// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// Maps floats to uints so that unsigned comparison orders them like the floats
// (negative values included): InterlockedMin over encoded depths keeps the nearest.
module gsplat_depth_encoding;

public static const uint DEPTH_NONE = 0xFF800000u; // encodeDepth(+inf): no surface

public uint encodeDepth(float depth) {
    uint bits = asuint(depth);
    return (bits & 0x80000000u) != 0u ? ~bits : (bits | 0x80000000u);
}

public float decodeDepth(uint encoded) {
    return asfloat((encoded & 0x80000000u) != 0u ? (encoded & 0x7FFFFFFFu) : ~encoded);
}
