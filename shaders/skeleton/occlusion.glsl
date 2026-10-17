// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// Per-pixel occlusion of skeleton shapes by a scene depth buffer, shared by
// openpose.frag and character_mask.frag. `bone` holds the shape's bone in
// framebuffer coordinates (both ends at the joint for a disc), `boneDepth` a
// depth at each end that is linear in screen space (1 / distance with a
// perspective projection, NDC depth otherwise), and `projection` the constants
// P[3][2], P[2][2] and whether the projection is perspective.

// NDC depth of the scene per pixel (+inf: no surface).
layout(std430, set = 1, binding = 1) readonly buffer OcclusionDepth {
    float occlusionDepth[];
};

// Whether the scene surface at this pixel lies in front of the skeleton. The
// skeleton's depth here is that of the nearest point of its bone, interpolated
// along the bone in screen space: 1 / distance (perspective) or NDC depth is
// linear there. The fragment is behind if it is farther than the surface by more
// than `bias` of the surface's distance.
bool occluded(vec4 bone, vec2 boneDepth, vec3 projection, float bias, vec2 resolution) {
    ivec2 pixel = ivec2(gl_FragCoord.xy);
    float surface = occlusionDepth[uint(pixel.y) * uint(resolution.x) + uint(pixel.x)];
    if (isinf(surface)) {
        return false;
    }
    vec2 first = bone.xy, along = bone.zw - bone.xy;
    float lengthSquared = dot(along, along);
    float t = lengthSquared > 0.0 ? clamp(dot(gl_FragCoord.xy - first, along) / lengthSquared, 0.0, 1.0) : 0.0;
    float depth = mix(boneDepth.x, boneDepth.y, t);
    if (projection.z != 0.0) {
        float distance = 1.0 / depth;
        float surfaceDistance = projection.x / (surface + projection.y);
        return distance > surfaceDistance * (1.0 + bias);
    }
    return depth > surface + bias * abs(surface);
}
