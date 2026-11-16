// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

import matmul;

// The Q3_K weights as 16-byte vectors.
[[vk::binding(0, 0)]] StructuredBuffer<uint4> W4;

struct Decoder : IWeightDecoder {
    static void decode16(uint rowStart, uint element, out float4 w[4]) { decodeQ3_K(W4, rowStart, element, w); }
};

[shader("compute")]
[numthreads(THREADS, 1, 1)]
void main(uint3 groupId: SV_GroupID, uint local: SV_GroupIndex) {
    matmul<Decoder>(groupId, local);
}
