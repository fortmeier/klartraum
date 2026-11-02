// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

import glsl;

// Turns the visible trees of TreeProxies into copies of the Gaussian tree
// library (TreeGaussianForest), one per entry of the visible list: per tree
// the patch of its type and of the season sample nearest to the current
// season, its foot, turn and scale, and how far it has faded out. Trees whose
// size on screen times treeCopyPriority() lies below the cut-off
// (tree_size_cut.comp) fade out above it (treeCopyFade());
// they and the slots beyond the visible trees get scale 0 and are skipped.

#define TREE_CAMERA_SET 0
#define TREE_CAMERA_BINDING 5
#include "tree_render_common.slang"

[[vk::binding(0, 0)]] StructuredBuffer<TreeSite> visibleList;
[[vk::binding(1, 0)]] StructuredBuffer<uint> counter;
[[vk::binding(2, 0)]] StructuredBuffer<TreeRenderSettings> settings;
// The season samples (up to 4); the types' heights at scale 1 (x, y); x the number of season samples, y of types,
// z of copies at most; x the variants per type, y the scene's seed (treeSiteSeed()).
struct Library {
    vec4 seasons;
    vec4 heights;
    uvec4 counts;
    uvec4 variants;
};
[[vk::binding(3, 0)]] StructuredBuffer<Library> library;
[[vk::binding(4, 0)]] RWStructuredBuffer<TreeLodInstance> instances;
// [0] the cut-off in log2 pixels (float bits); see tree_size_cut.comp.
[[vk::binding(6, 0)]] StructuredBuffer<uint> state;

[[vk::binding(7, 0)]] StructuredBuffer<TreeTileConfig> configBuffer;

struct Push {
    uint capacity; // of the visible list
};
[[vk::push_constant]] ConstantBuffer<Push> pc;

[shader("compute")]
[numthreads(64, 1, 1)]
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= pc.capacity) {
        return;
    }
    TreeLodInstance inst;
    inst.posYaw = vec4(0.0f);
    inst.plane = vec4(0.0f);
    inst.ids = uvec4(0u);
    TreeSite site = visibleList[min(i, pc.capacity - 1u)];
    float h = site.posHeight.w;
    float pixels = treePixelDiameter(camera.cameraWorldPos.xyz, site.posHeight.xyz + vec3(0.0f, 0.5f * h, 0.0f),
                                     0.5f * h, treeFocalPixels(settings[0]));
    pixels *= treeCopyPriority(treeLandCover(configBuffer[0].terrain, site.posHeight.x, site.posHeight.z).vegetation.x);
    float shown = site.kind.w * treeCopyFade(pixels, uintBitsToFloat(state[0]));
    if (i < min(counter[0], pc.capacity) && shown > 0.0f) {
        uint type = min(site.kind.x == TREE_SITE_CONIFER ? 1u : 0u, library[0].counts.y - 1u);
        float season = treeSeasonWrap(settings[0].season.x);
        uint nearest = 0u;
        float best = 4.0f;
        for (uint k = 0u; k < library[0].counts.x; ++k) {
            float d = treeSeasonWrap(season - library[0].seasons[k]);
            d = min(d, 4.0f - d);
            if (d < best) {
                best = d;
                nearest = k;
            }
        }
        inst.posYaw = vec4(site.posHeight.xyz, site.kind.y);
        inst.plane = vec4(0.0f, 0.0f, site.posHeight.w / library[0].heights[type], 0.0f);
        uint variant = treeSiteSeed(library[0].variants.y, site.kind.y, library[0].variants.x) - 1u;
        inst.ids = uvec4((type * library[0].variants.x + variant) * library[0].counts.x + nearest, floatBitsToUint(1.0f - shown),
                         floatBitsToUint(1.0f - site.kind.z), 1u);
    }
    instances[i] = inst;
}
