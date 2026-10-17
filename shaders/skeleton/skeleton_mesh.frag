// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#version 450

// Shades the skeleton's surface by how much it faces the camera and drops it
// where the scene's depth lies in front (see SkeletonMeshRenderer::setOcclusionDepth()).

#include "skeleton_mesh_common.glsl"

layout(location = 0) in vec3 viewPosition;
layout(location = 1) in vec3 viewNormal;
layout(location = 2) in float clipDepth;
layout(location = 3) flat in vec4 colour;
layout(location = 4) flat in vec3 projection;

layout(location = 0) out vec4 outColor;

// NDC depth of the scene per pixel (+inf: no surface).
layout(std430, set = 1, binding = 4) readonly buffer OcclusionDepth { float occlusionDepth[]; };

const float AMBIENT = 0.35;

bool occluded() {
    ivec2 pixel = ivec2(gl_FragCoord.xy);
    float surface = occlusionDepth[uint(pixel.y) * uint(pc.resolution.x) + uint(pixel.x)];
    if (isinf(surface)) {
        return false;
    }
    if (projection.z != 0.0) {
        float surfaceDistance = projection.x / (surface + projection.y);
        return -viewPosition.z > surfaceDistance * (1.0 + pc.occlusionBias);
    }
    return clipDepth > surface + pc.occlusionBias * abs(surface);
}

void main() {
    if (pc.useOcclusion != 0u && occluded()) {
        discard;
    }
    vec3 towardsCamera = projection.z != 0.0 ? normalize(-viewPosition) : vec3(0.0, 0.0, 1.0);
    float facing = abs(dot(normalize(viewNormal), towardsCamera));
    outColor = vec4(colour.rgb * (AMBIENT + (1.0 - AMBIENT) * facing), colour.a);
}
