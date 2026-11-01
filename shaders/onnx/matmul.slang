// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// Batched MatMul, output [batch, rows, columns] = lhs [batch, rows, reduction] x rhs [batch, reduction, columns].
// A workgroup computes a 64x128 output tile; each invocation keeps a 4x8 block
// of it in registers (rows 4 ty .. 4 ty + 3, columns 4 tx .. and 64 + 4 tx ..),
// so neighbouring invocations read neighbouring float4s of the shared tiles and
// every staged value feeds several FMAs. Tiles are staged 32 reduction steps at
// a time, the left one transposed so that both are read as float4 rows.
//
// Shaders call matmul() from a 16x16 workgroup with an epilogue that turns each
// product element into the stored value.
module matmul;

public struct MatMulPushConstants {
    public uint batchCount;
    public uint rows;
    public uint columns;
    public uint reduction;
    public uint lhsBatchCount;
    public uint rhsBatchCount;
};

// Turns the product element at output `index` into the value to store.
public interface IMatMulEpilogue {
    static float apply(float value, uint index);
}

// Stores the product unchanged.
public struct IdentityEpilogue : IMatMulEpilogue {
    public static float apply(float value, uint index) { return value; }
}

static const uint TILE_ROWS = 64;
static const uint TILE_COLUMNS = 128;
static const uint STEP = 32;

// The left tile, transposed: [reduction step][row / 4].
groupshared float4 lhsTile[STEP][TILE_ROWS / 4];
// The right tile: [reduction step][column / 4].
groupshared float4 rhsTile[STEP][TILE_COLUMNS / 4];

public void matmul<Epilogue : IMatMulEpilogue>(uint3 localId, uint3 groupId, MatMulPushConstants pc,
                                               StructuredBuffer<float> lhs, StructuredBuffer<float> rhs,
                                               RWStructuredBuffer<float> outputData) {
    const uint tx = localId.x;
    const uint ty = localId.y;
    const uint threadIndex = ty * 16 + tx;
    const uint rowBase = groupId.y * TILE_ROWS;
    const uint columnBase = groupId.x * TILE_COLUMNS;
    const uint batch = groupId.z;
    const uint lhsOffset = (pc.lhsBatchCount == 1 ? 0 : batch) * pc.rows * pc.reduction;
    const uint rhsOffset = (pc.rhsBatchCount == 1 ? 0 : batch) * pc.reduction * pc.columns;

    // Each invocation stages eight values of each tile per step: the left tile
    // as 64 rows x 32 steps, the right tile as 32 steps x 128 columns.
    const uint lhsLoadRow = threadIndex / 4;           // 0..63
    const uint lhsLoadStep = (threadIndex % 4) * 8;    // 0, 8, 16, 24
    const uint rhsLoadStep = threadIndex / 32;         // 0..7, +8, +16, +24
    const uint rhsLoadColumn = (threadIndex % 32) * 4; // 0..124

    float4 accumulator[4][2];
    for (uint i = 0; i < 4; ++i) {
        accumulator[i][0] = float4(0.0);
        accumulator[i][1] = float4(0.0);
    }

    for (uint k0 = 0; k0 < pc.reduction; k0 += STEP) {
        const uint row = rowBase + lhsLoadRow;
        for (uint i = 0; i < 8; ++i) {
            const uint k = k0 + lhsLoadStep + i;
            lhsTile[lhsLoadStep + i][lhsLoadRow / 4][lhsLoadRow % 4] =
                row < pc.rows && k < pc.reduction ? lhs[lhsOffset + row * pc.reduction + k] : 0.0;
        }
        for (uint part = 0; part < 4; ++part) {
            const uint k = k0 + rhsLoadStep + part * 8;
            float4 r;
            for (uint i = 0; i < 4; ++i) {
                const uint column = columnBase + rhsLoadColumn + i;
                r[i] = k < pc.reduction && column < pc.columns ? rhs[rhsOffset + k * pc.columns + column] : 0.0;
            }
            rhsTile[rhsLoadStep + part * 8][rhsLoadColumn / 4] = r;
        }
        GroupMemoryBarrierWithGroupSync();
        for (uint s = 0; s < STEP; ++s) {
            const float4 a = lhsTile[s][ty];
            const float4 b0 = rhsTile[s][tx];
            const float4 b1 = rhsTile[s][16 + tx];
            for (uint i = 0; i < 4; ++i) {
                accumulator[i][0] += a[i] * b0;
                accumulator[i][1] += a[i] * b1;
            }
        }
        GroupMemoryBarrierWithGroupSync();
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
                    outputData[index] = Epilogue.apply(accumulator[i][part][j], index);
                }
            }
        }
    }
}
