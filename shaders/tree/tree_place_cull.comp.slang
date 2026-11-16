// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

import glsl;

// Culls the trees of the placement's pool for this frame (TreeProxies): one
// workgroup per slot. A tile outside the view is skipped at once by its
// bounds; of the others, each tree whose box is in the view and whose crown
// spans at least pc.minPixels pixels is appended to the visible instances
// with how far it has faded in (kind.w, where the forest canopy fades out)
// and how much sun the terrain lets through to its crown (kind.z),
// unless it is one of the near trees generated in full (within the near
// radius). Trees beyond the capacity are counted. With pc.bySize, every tree
// in the view is counted by its size times treeCopyPriority() into the bins
// of `sizes`, and only those at or above the last frame's cut-off
// (tree_size_cut.comp) are appended.

#define TREE_CAMERA_SET 0
#define TREE_CAMERA_BINDING 4
#include "tree_render_common.slang"

[[vk::binding(0, 0)]] StructuredBuffer<TreeTileConfig> configBuffer;
// Per slot: xy the tile's indices, w 1 when the slot holds a tile.
[[vk::binding(1, 0)]] StructuredBuffer<ivec4> slots;
[[vk::binding(2, 0)]] StructuredBuffer<TreeSite> sites;
[[vk::binding(3, 0)]] StructuredBuffer<vec4> bounds;
[[vk::binding(5, 0)]] StructuredBuffer<TreeRenderSettings> settings;
[[vk::binding(6, 0)]] RWStructuredBuffer<TreeSite> visibleList;
// [0] trees visible (also beyond the capacity), [1] trees left out for lack of room; zeroed before the pass.
[[vk::binding(7, 0)]] RWStructuredBuffer<uint> counter;
// x the near radius: trees whose foot is at most this far from the camera are drawn in full elsewhere.
[[vk::binding(8, 0)]] StructuredBuffer<vec4> near;

// [0] the cut-off in log2 pixels (float bits), then the bins from [4] on; see tree_size_cut.comp.
[[vk::binding(9, 0)]] RWStructuredBuffer<uint> sizes;

struct Push {
    uint capacity;
    float minPixels;
    float fadeAbove; // the trees fade in up to this size (kind.w of the visible sites), 0: no fading
    uint bySize;     // 1: the list holds the largest trees, down to the cut-off in `sizes`
};
[[vk::push_constant]] ConstantBuffer<Push> pc;

[shader("compute")]
[numthreads(256, 1, 1)]
void main() {
    uint slot = gl_WorkGroupID.x;
    ivec4 tile = slots[slot];
    vec4 b = bounds[slot];
    if (tile.w == 0 || b.z == 0.0f) {
        return;
    }
    uint cells = configBuffer[0].tiles.x;
    float size = float(cells) * configBuffer[0].placement.grid.x;
    mat4 viewProj = camera.proj * camera.view;
    vec3 lo = vec3(float(tile.x) * size, b.x, float(tile.y) * size);
    vec3 hi = vec3(lo.x + size, b.y, lo.z + size);
    if (treeBoxOutsideFrustum(viewProj, lo, hi)) {
        return;
    }
    vec3 eye = camera.cameraWorldPos.xyz;
    float focal = treeFocalPixels(settings[0]);
    uint base = slot * cells * cells;
    for (uint c = gl_LocalInvocationID.x; c < cells * cells; c += 256u) {
        TreeSite site = sites[base + c];
        float h = site.posHeight.w;
        if (h <= 0.0f) {
            continue;
        }
        vec3 foot = site.posHeight.xyz;
        if (distance(foot, eye) <= near[0].x) {
            continue;
        }
        vec4 shape = treeProxyShape(site.kind.x);
        vec3 r = vec3(shape.z * h, 0.0f, shape.z * h);
        if (treeBoxOutsideFrustum(viewProj, foot - r, foot + r + vec3(0.0f, h, 0.0f))) {
            continue;
        }
        float pixels = treePixelDiameter(eye, foot + vec3(0.0f, 0.5f * h, 0.0f), 0.5f * h, focal);
        // The band of the settings when they give one (it follows the number of visible trees).
        float minPixels = settings[0].canopyBand.x > 0.0f && pc.fadeAbove > 0.0f ? settings[0].canopyBand.x : pc.minPixels;
        float fadeAbove = settings[0].canopyBand.x > 0.0f && pc.fadeAbove > 0.0f ? settings[0].canopyBand.y : pc.fadeAbove;
        if (pixels < minPixels) {
            continue;
        }
        if (pc.bySize != 0u) {
            // The site's forest weight, before kind.z carries the sun's visibility.
            float rank = pixels * treeCopyPriority(site.kind.z);
            atomicAdd(sizes[4u + treeSizeBin(rank)], 1u);
            if (log2(rank) < uintBitsToFloat(sizes[0])) {
                continue;
            }
        }
        site.kind.z = treeTerrainSunVisibility(configBuffer[0].terrain, foot + vec3(0.0f, 0.6f * h, 0.0f), settings[0].sun.xyz);
        site.kind.w = fadeAbove > minPixels ? treeProxyFade(vec4(minPixels, fadeAbove, 0.0f, 0.0f), pixels)
                                                  : 1.0f;
        uint k = atomicAdd(counter[0], 1u);
        if (k < pc.capacity) {
            visibleList[k] = site;
        } else {
            atomicAdd(counter[1], 1u);
        }
    }
}
