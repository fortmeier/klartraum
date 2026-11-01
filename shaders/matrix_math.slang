// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// Matrix functions that Slang's core module does not provide.
module matrix_math;

// Returns the inverse of `m` (GLSL.std.450 MatrixInverse, as GLSL's inverse()).
public float2x2 inverse(float2x2 m) {
    return spirv_asm {
        result:$$float2x2 = OpExtInst glsl450 MatrixInverse $m;
    };
}
