// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

import glsl;
import scalar_buffer;

// First pass of TreeMossLod: makes the pool of level-of-detail Gaussians
// transparent, so slots no patch takes this frame stay invisible.

[[vk::binding(0, 0)]] RWScalarBuffer<vec4> colAlpha;

struct Push {
    uint offset;
    uint capacity;
};
[[vk::push_constant]] ConstantBuffer<Push> pc;

[shader("compute")]
[numthreads(256, 1, 1)]
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i < pc.capacity) {
        colAlpha[pc.offset + i] = vec4(0.0f);
    }
}
