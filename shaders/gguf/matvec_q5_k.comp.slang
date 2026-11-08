// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

import matvec;

// The Q5_K weights as 16-byte vectors.
[[vk::binding(0, 0)]] StructuredBuffer<uint4> W4;

struct Decoder : IWeightDecoder {
    static void decode16(uint rowStart, uint element, out float4 w[4]) { decodeQ5_K(W4, rowStart, element, w); }
};

[shader("compute")]
[numthreads(32 * SUBGROUPS, 1, 1)]
void main(uint3 groupId: SV_GroupID) {
    matvec<Decoder>(groupId);
}
