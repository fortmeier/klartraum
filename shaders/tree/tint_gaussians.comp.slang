// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

import glsl;
import scalar_buffer;

// Tints the colours of captured Gaussians for the current light
// (TreeRenderSettings::capturedTint), the shadows of the light volume
// (treeCapturedShadeAt()) and snow (treeCapturedColor()): the displayed colour
// 0.5 + C0 * sh0 is changed and turned back into the degree 0 coefficient.

#define TREE_FN
#include "klartraum/tree/tree_shared.hpp"
#define TREE_LIGHT_SET 0
#define TREE_LIGHT_BINDING 4
#include "tree_light.slang"

[[vk::binding(0, 0)]] ScalarBuffer<vec4> original;
[[vk::binding(1, 0)]] StructuredBuffer<TreeRenderSettings> settings;
[[vk::binding(2, 0)]] RWScalarBuffer<vec4> tinted;
[[vk::binding(3, 0)]] ScalarBuffer<vec3> positions;
[[vk::binding(5, 0)]] ScalarBuffer<vec4> rotations;
[[vk::binding(6, 0)]] ScalarBuffer<vec3> scales;

struct Push {
    uint count;
};
[[vk::push_constant]] ConstantBuffer<Push> pc;

static const float SH_C0 = 0.28209479f;

[shader("compute")]
[numthreads(256, 1, 1)]
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= pc.count) {
        return;
    }
    vec4 c = original[i];
    TreeRenderSettings s = settings[0];
    // Without the camera here, which side of a Gaussian is seen is unknown.
    float shade = treeCapturedShadeAt(s, positions[i], vec3(0.0f));
    vec3 normal = treeGaussianNormal(rotations[i], scales[i]);
    vec3 displayed = treeCapturedColor(vec3(0.5f) + SH_C0 * c.rgb, normal, positions[i], shade, s);
    tinted[i] = vec4((displayed - vec3(0.5f)) / SH_C0, c.a);
}
