// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

import gsplat_raster_fragment;

// Test scene depth before a fragment can update the surface-depth SSBO.
// The containing pass keeps native depth writes disabled for Gaussians.
[shader("fragment")]
[earlydepthstencil]
float4 main(SplatVertex input) : SV_Target0 {
    return rasterFragment<false>(input);
}
