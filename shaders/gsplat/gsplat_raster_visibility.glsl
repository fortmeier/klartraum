// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// Center visibility shared by raster culling and its depth-only compute projection.
bool gaussianRasterCenterVisible(vec3 ndc, float frustumDilation) {
    float clip = 1.0 + frustumDilation;
    return !(abs(ndc.x) > clip || abs(ndc.y) > clip || ndc.z < -frustumDilation || ndc.z > 1.0);
}
