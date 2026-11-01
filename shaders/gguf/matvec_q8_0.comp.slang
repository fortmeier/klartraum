// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

import matvec;

// The Q8_0 weights as words.
[[vk::binding(0, 0)]] StructuredBuffer<uint> W;

struct Decoder : IWeightDecoder {
    static void decode16(uint rowStart, uint element, out float4 w[4]) { decodeQ8_0(W, rowStart, element, w); }
};

[shader("compute")]
[numthreads(32 * SUBGROUPS, 1, 1)]
void main(uint3 groupId: SV_GroupID) {
    matvec<Decoder>(groupId);
}
