// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

import glsl;

// Counts the visible trees of TreeProxies by their size on screen, times
// treeCopyPriority() of the forest at their foot, into the bins of
// treeSizeBin() (TreeGaussianForest), which are zeroed before the pass.

#define TREE_CAMERA_SET 0
#define TREE_CAMERA_BINDING 3
#include "tree_render_common.slang"

[[vk::binding(0, 0)]] StructuredBuffer<TreeSite> visibleList;
[[vk::binding(1, 0)]] StructuredBuffer<uint> counter;
[[vk::binding(2, 0)]] StructuredBuffer<TreeRenderSettings> settings;
[[vk::binding(5, 0)]] StructuredBuffer<TreeTileConfig> configBuffer;
// [0..3] the cut-off's state (tree_library_cut.comp), then the bins.
[[vk::binding(4, 0)]] RWStructuredBuffer<uint> state;

struct Push {
    uint capacity; // of the visible list
};
[[vk::push_constant]] ConstantBuffer<Push> pc;

[shader("compute")]
[numthreads(256, 1, 1)]
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= min(counter[0], pc.capacity)) {
        return;
    }
    TreeSite site = visibleList[i];
    float h = site.posHeight.w;
    float pixels = treePixelDiameter(camera.cameraWorldPos.xyz, site.posHeight.xyz + vec3(0.0f, 0.5f * h, 0.0f),
                                     0.5f * h, treeFocalPixels(settings[0]));
    pixels *= treeCopyPriority(treeLandCover(configBuffer[0].terrain, site.posHeight.x, site.posHeight.z).vegetation.x);
    atomicAdd(state[4u + treeSizeBin(pixels)], 1u);
}
