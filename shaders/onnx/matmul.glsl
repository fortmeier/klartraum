// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// Batched MatMul, output [batch, rows, columns] = lhs [batch, rows, reduction] x rhs [batch, reduction, columns].
// A workgroup computes a 64x128 output tile; each invocation keeps a 4x8 block
// of it in registers (rows 4 ty .. 4 ty + 3, columns 4 tx .. and 64 + 4 tx ..),
// so neighbouring invocations read neighbouring vec4s of the shared tiles and
// every staged value feeds several FMAs. Tiles are staged 32 reduction steps at
// a time, the left one transposed so that both are read as vec4 rows.
//
// Variants define EPILOGUE_RELU_SQUARE (store max(x, 0)^2) or
// EPILOGUE_RESIDUAL (store x + residual, an extra input of the output's shape).
layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;
layout(push_constant) uniform PushConstants {
    uint batchCount;
    uint rows;
    uint columns;
    uint reduction;
    uint lhsBatchCount;
    uint rhsBatchCount;
} pc;
layout(set = 0, binding = 0) restrict readonly buffer LhsBuffer { float lhs[]; };
layout(set = 0, binding = 1) restrict readonly buffer RhsBuffer { float rhs[]; };
#ifdef EPILOGUE_RESIDUAL
layout(set = 0, binding = 2) restrict readonly buffer ResidualBuffer { float residualData[]; };
layout(set = 0, binding = 3) restrict writeonly buffer OutputBuffer { float outputData[]; };
#else
layout(set = 0, binding = 2) restrict writeonly buffer OutputBuffer { float outputData[]; };
#endif

float epilogue(float value, uint index) {
#if defined(EPILOGUE_RELU_SQUARE)
    const float positive = max(value, 0.0);
    return positive * positive;
#elif defined(EPILOGUE_RESIDUAL)
    return value + residualData[index];
#else
    return value;
#endif
}

const uint TILE_ROWS = 64;
const uint TILE_COLUMNS = 128;
const uint STEP = 32;

shared vec4 lhsTile[STEP][TILE_ROWS / 4];
shared vec4 rhsTile[STEP][TILE_COLUMNS / 4];

void main() {
    const uint tx = gl_LocalInvocationID.x;
    const uint ty = gl_LocalInvocationID.y;
    const uint thread = ty * 16 + tx;
    const uint rowBase = gl_WorkGroupID.y * TILE_ROWS;
    const uint columnBase = gl_WorkGroupID.x * TILE_COLUMNS;
    const uint batch = gl_WorkGroupID.z;
    const uint lhsOffset = (pc.lhsBatchCount == 1 ? 0 : batch) * pc.rows * pc.reduction;
    const uint rhsOffset = (pc.rhsBatchCount == 1 ? 0 : batch) * pc.reduction * pc.columns;

    // Each invocation stages eight values of each tile per step: the left tile
    // as 64 rows x 32 steps, the right tile as 32 steps x 128 columns.
    const uint lhsLoadRow = thread / 4;            // 0..63
    const uint lhsLoadStep = (thread % 4) * 8;     // 0, 8, 16, 24
    const uint rhsLoadStep = thread / 32;          // 0..7, +8, +16, +24
    const uint rhsLoadColumn = (thread % 32) * 4;  // 0..124

    vec4 accumulator[4][2];
    for (uint i = 0; i < 4; ++i) { accumulator[i][0] = vec4(0.0); accumulator[i][1] = vec4(0.0); }

    for (uint k0 = 0; k0 < pc.reduction; k0 += STEP) {
        const uint row = rowBase + lhsLoadRow;
        for (uint i = 0; i < 8; ++i) {
            const uint k = k0 + lhsLoadStep + i;
            lhsTile[lhsLoadStep + i][lhsLoadRow / 4][lhsLoadRow % 4] =
                row < pc.rows && k < pc.reduction ? lhs[lhsOffset + row * pc.reduction + k] : 0.0;
        }
        for (uint part = 0; part < 4; ++part) {
            const uint k = k0 + rhsLoadStep + part * 8;
            vec4 r;
            for (uint i = 0; i < 4; ++i) {
                const uint column = columnBase + rhsLoadColumn + i;
                r[i] = k < pc.reduction && column < pc.columns ? rhs[rhsOffset + k * pc.columns + column] : 0.0;
            }
            rhsTile[rhsLoadStep + part * 8][rhsLoadColumn / 4] = r;
        }
        barrier();
        for (uint s = 0; s < STEP; ++s) {
            const vec4 a = lhsTile[s][ty];
            const vec4 b0 = rhsTile[s][tx];
            const vec4 b1 = rhsTile[s][16 + tx];
            for (uint i = 0; i < 4; ++i) {
                accumulator[i][0] += a[i] * b0;
                accumulator[i][1] += a[i] * b1;
            }
        }
        barrier();
    }

    if (batch >= pc.batchCount) return;
    for (uint i = 0; i < 4; ++i) {
        const uint row = rowBase + ty * 4 + i;
        if (row >= pc.rows) continue;
        for (uint part = 0; part < 2; ++part) {
            for (uint j = 0; j < 4; ++j) {
                const uint column = columnBase + part * 64 + tx * 4 + j;
                if (column < pc.columns) {
                    const uint index = (batch * pc.rows + row) * pc.columns + column;
                    outputData[index] = epilogue(accumulator[i][part][j], index);
                }
            }
        }
    }
}
