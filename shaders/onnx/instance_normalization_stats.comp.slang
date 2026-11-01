// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// Pass 1 of chunked instance normalization (see instance_normalization_chunks.slang).
import instance_normalization_chunks;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint localIndex: SV_GroupIndex, uint3 groupId: SV_GroupID) {
    computeChunkStatistics(localIndex, groupId);
}
