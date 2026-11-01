// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// Adds the split-K partial sums of a 3D convolution and its bias (see conv3d_tiled.slang).
import conv3d_tiled;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint localIndex: SV_GroupIndex, uint3 groupId: SV_GroupID) {
    conv3dReduce(localIndex, groupId);
}
