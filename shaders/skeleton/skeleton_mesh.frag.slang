// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// Shades the skeleton's surface by how much it faces the camera and drops it
// where the scene's depth lies in front (see SkeletonMeshRenderer::setOcclusionDepth()).
import skeleton_mesh_common;

// NDC depth of the scene per pixel (+inf: no surface).
[[vk::binding(4, 1)]] StructuredBuffer<float> occlusionDepth;

static const float AMBIENT = 0.35;

bool occluded(SkeletonMeshVertex input) {
    int2 pixel = int2(input.position.xy);
    float surface = occlusionDepth[uint(pixel.y) * uint(pc.resolution.x) + uint(pixel.x)];
    if (isinf(surface)) {
        return false;
    }
    if (input.projection.z != 0.0) {
        float surfaceDistance = input.projection.x / (surface + input.projection.y);
        return -input.viewPosition.z > surfaceDistance * (1.0 + pc.occlusionBias);
    }
    return input.clipDepth > surface + pc.occlusionBias * abs(surface);
}

[shader("fragment")]
float4 main(SkeletonMeshVertex input) : SV_Target0 {
    if (pc.useOcclusion != 0u && occluded(input)) {
        discard;
    }
    float3 towardsCamera = input.projection.z != 0.0 ? normalize(-input.viewPosition) : float3(0.0, 0.0, 1.0);
    float facing = abs(dot(normalize(input.viewNormal), towardsCamera));
    return float4(input.colour.rgb * (AMBIENT + (1.0 - AMBIENT) * facing), input.colour.a);
}
