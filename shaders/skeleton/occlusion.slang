// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// Per-pixel occlusion of skeleton shapes by a scene depth buffer, shared by
// openpose.frag and character_mask.frag. `bone` holds the shape's bone in
// framebuffer coordinates (both ends at the joint for a disc), `boneDepth` a
// depth at each end that is linear in screen space (1 / distance with a
// perspective projection, NDC depth otherwise), and `projection` the constants
// P[2][3], P[2][2] (row, column) and whether the projection is perspective.
module occlusion;

// NDC depth of the scene per pixel (+inf: no surface).
[[vk::binding(1, 1)]] StructuredBuffer<float> occlusionDepth;

// Whether the scene surface at the fragment `fragCoord` lies in front of the
// skeleton. The skeleton's depth here is that of the nearest point of its bone,
// interpolated along the bone in screen space: 1 / distance (perspective) or
// NDC depth is linear there. The fragment is behind if it is farther than the
// surface by more than `bias` of the surface's distance.
public bool occluded(float2 fragCoord, float4 bone, float2 boneDepth, float3 projection, float bias, float2 resolution) {
    int2 pixel = int2(fragCoord);
    float surface = occlusionDepth[uint(pixel.y) * uint(resolution.x) + uint(pixel.x)];
    if (isinf(surface)) {
        return false;
    }
    float2 first = bone.xy, along = bone.zw - bone.xy;
    float lengthSquared = dot(along, along);
    float t = lengthSquared > 0.0 ? clamp(dot(fragCoord - first, along) / lengthSquared, 0.0, 1.0) : 0.0;
    float depth = lerp(boneDepth.x, boneDepth.y, t);
    if (projection.z != 0.0) {
        float distance = 1.0 / depth;
        float surfaceDistance = projection.x / (surface + projection.y);
        return distance > surfaceDistance * (1.0 + bias);
    }
    return depth > surface + bias * abs(surface);
}
