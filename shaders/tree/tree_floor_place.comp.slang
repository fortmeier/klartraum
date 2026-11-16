// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

import glsl;

// The copies of moss patches on the forest floor near the camera
// (TreeForestFloor): one invocation per cell of the grid of counts.x x
// counts.x cells around the camera (treeFloorCell(), the same grid as
// treeFloorAround()), writing its copy into the cell's slot with how much
// of the sun the terrain takes away; scale 0 where there is none, and
// everywhere while the camera is too high above the ground.

#define TREE_CAMERA_SET 0
#define TREE_CAMERA_BINDING 1
#include "tree_render_common.slang"

[[vk::binding(0, 0)]] StructuredBuffer<TreeFloorSetup> setupBuffer;
[[vk::binding(2, 0)]] StructuredBuffer<TreeRenderSettings> settings;
[[vk::binding(3, 0)]] RWStructuredBuffer<TreeLodInstance> instances;
// [0] x 1 while the floor can be in reach of the camera (TreeForestFloor::setView()).
[[vk::binding(4, 0)]] StructuredBuffer<vec4> frame;

struct Push {
    uint pad;
};
[[vk::push_constant]] ConstantBuffer<Push> pc;

[shader("compute")]
[numthreads(16, 16, 1)]
void main() {
    uint n = setupBuffer[0].counts.x;
    uvec2 ij = gl_GlobalInvocationID.xy;
    if (ij.x >= n || ij.y >= n) {
        return;
    }
    TreeLodInstance inst;
    inst.posYaw = vec4(0.0f);
    inst.plane = vec4(0.0f);
    inst.ids = uvec4(0u);
    if (frame[0].x != 0.0f) {
        vec3 eye = camera.cameraWorldPos.xyz;
        int mid = int(n / 2u);
        int cx = int(floor(eye.x / setupBuffer[0].grid.x)) - mid + 1 + int(ij.x);
        int cz = int(floor(eye.z / setupBuffer[0].grid.x)) - mid + 1 + int(ij.y);
        inst = treeFloorCell(setupBuffer[0], cx, cz, eye);
        if (inst.plane.z > 0.0f) {
            float sun = treeTerrainSunVisibility(setupBuffer[0].terrain, inst.posYaw.xyz, settings[0].sun.xyz);
            inst.ids.z = floatBitsToUint(1.0f - sun);
        }
    }
    instances[ij.y * n + ij.x] = inst;
}
