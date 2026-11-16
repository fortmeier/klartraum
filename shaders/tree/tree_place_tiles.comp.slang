// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

import glsl;

// Generates the tiles of the tree placement that TreeTileCache asks for this
// frame (TreePlacementTiles): one workgroup per tile writes treeSiteAt() of
// each of its cells into the tile's slot of the pool, rows along x, and the
// tile's number of sites and the bounds of their heights (from the lowest
// foot to the highest top).

#define TREE_FN
#include "klartraum/tree/tree_shared.hpp"

[[vk::binding(0, 0)]] StructuredBuffer<TreeTileConfig> configBuffer;
// [0].x the number of tiles to generate; then per tile x its slot, yz its indices, w unused.
[[vk::binding(1, 0)]] StructuredBuffer<ivec4> generate;
// Per slot cells^2 sites.
[[vk::binding(2, 0)]] RWStructuredBuffer<TreeSite> sites;
// Per slot: x the lowest foot, y the highest top, z the number of trees and shrubs (float), w unused.
[[vk::binding(3, 0)]] RWStructuredBuffer<vec4> bounds;

groupshared int lowest;
groupshared int highest;
groupshared uint count;

// Heights as ints that order like the floats (all heights are positive).
int key(float h) {
    return floatBitsToInt(max(h, 0.0f));
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main() {
    uint w = gl_WorkGroupID.x;
    if (int(w) >= generate[0].x) {
        return;
    }
    ivec4 tile = generate[w + 1u];
    uint cells = configBuffer[0].tiles.x;
    if (gl_LocalInvocationID.x == 0u) {
        lowest = key(1e30f);
        highest = 0;
        count = 0u;
    }
    barrier();
    uint base = uint(tile.x) * cells * cells;
    for (uint c = gl_LocalInvocationID.x; c < cells * cells; c += 256u) {
        int cx = tile.y * int(cells) + int(c % cells);
        int cz = tile.z * int(cells) + int(c / cells);
        TreeSite site = treeSiteAt(configBuffer[0].terrain, configBuffer[0].placement, cx, cz);
        sites[base + c] = site;
        if (site.posHeight.w > 0.0f) {
            atomicMin(lowest, key(site.posHeight.y));
            atomicMax(highest, key(site.posHeight.y + site.posHeight.w));
            atomicAdd(count, 1u);
        }
    }
    barrier();
    if (gl_LocalInvocationID.x == 0u) {
        bounds[tile.x] = vec4(intBitsToFloat(lowest), intBitsToFloat(highest), float(count), 0.0f);
    }
}
