// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// 3D convolution with at most eight output channels (see conv3dSmall() in conv3d_tiled.slang).
import conv3d_tiled;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 globalId: SV_DispatchThreadID) {
    conv3dSmall(globalId.x);
}
