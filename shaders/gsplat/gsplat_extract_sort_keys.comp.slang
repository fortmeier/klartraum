// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

import scalar_buffer;

// Binned SoA inputs — only binMask and z are needed for key extraction
[[vk::binding(0, 0)]] ScalarBuffer<uint> binMask;
[[vk::binding(1, 0)]] ScalarBuffer<float> zBuf;
// One uint: the number of binned elements.
[[vk::binding(2, 0)]] ScalarBuffer<uint> totalCount;

[[vk::binding(3, 0)]] RWScalarBuffer<uint> values;
[[vk::binding(4, 0)]] RWScalarBuffer<uint> indexes;

[shader("compute")]
[numthreads(128, 1, 1)]
void main(uint3 globalId: SV_DispatchThreadID) {
    uint idx = globalId.x;
    if (idx >= totalCount[0]) return;

    // Sort key: (binIndex[3:0] << 28) | (asuint(z) >> 4)
    uint binIdx = uint(firstbithigh(binMask[idx]));
    uint zBits = asuint(zBuf[idx]) >> 4;
    values[idx] = (binIdx << 28u) | zBits;
    indexes[idx] = idx;
}
