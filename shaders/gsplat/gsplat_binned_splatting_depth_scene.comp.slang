// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

import gsplat_binned_splatting;

struct Variant : ISplatVariant {
    static const bool outputDepth = true;
    static const bool depthOnly = false;
    static const bool sceneDepth = true;
};

[shader("compute")]
[numthreads(TILE_X, TILE_Y, 1)]
void main(uint3 globalId: SV_DispatchThreadID, uint localIndex: SV_GroupIndex) {
    binnedSplatting<Variant>(globalId, localIndex);
}
