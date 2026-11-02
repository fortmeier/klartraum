// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

import glsl;

// The terrain's node selection (TreeTerrain, CDLOD): one workgroup walks the
// quadtree from the root, a level per step. The nodes of the current level
// are in one queue; each thread takes some of them and culls, draws or splits
// them (treeTerrainNodeAction()), putting the children of split nodes into
// the other queue for the next, finer level. The selected nodes and the
// indirect draw of all of them are the result; selectTerrainNodesReference()
// does the same on the CPU. It also writes the dispatch of the vertex pass.

#define TREE_CAMERA_SET 0
#define TREE_CAMERA_BINDING 5
#include "tree_render_common.slang"

#define THREADS 256u

// Per selected node: xy its lower corner (x, z), z its size, w its level.
[[vk::binding(0, 0)]] RWStructuredBuffer<vec4> nodes;
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> drawArgs; // vertexCount, instanceCount, firstVertex, firstInstance
// Two queues of node indices (ix, iz), queue q at [q * capacity, (q + 1) * capacity).
[[vk::binding(2, 0)]] RWStructuredBuffer<uvec2> queues;
[[vk::binding(3, 0)]] StructuredBuffer<TreeTerrainConfig> configBuffer;
[[vk::binding(4, 0)]] StructuredBuffer<vec2> bounds;
// The dispatch of tree_terrain_vertices.comp: a workgroup per selected node.
[[vk::binding(7, 0)]] RWStructuredBuffer<uint> dispatchArgs;
// TreeTerrainStat; zeroed before the pass.
[[vk::binding(6, 0)]] RWStructuredBuffer<uint> stats;

groupshared uint queueCount[2];

void draw(uint ix, uint iz, uint level) {
    float size = treeTerrainNodeSize(configBuffer[0].lod, level);
    uint k = atomicAdd(stats[0], 1u);
    if (k < configBuffer[0].lod.levels.z) {
        nodes[k] = vec4(configBuffer[0].lod.root.x + float(ix) * size, configBuffer[0].lod.root.y + float(iz) * size, size, float(level));
    } else {
        atomicAdd(stats[3], 1u);
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main() {
    uint t = gl_LocalInvocationID.x;
    uint capacity = configBuffer[0].lod.levels.w;
    mat4 viewProj = camera.proj * camera.view;
    vec3 eye = camera.cameraWorldPos.xyz;
    float zoom = treeTerrainZoom(camera.proj);
    if (t == 0u) {
        queues[0] = uvec2(0u, 0u);
        queueCount[0] = 1u;
        queueCount[1] = 0u;
    }
    memoryBarrierBuffer();
    memoryBarrierShared();
    barrier();

    uint current = 0u;
    for (int level = int(configBuffer[0].lod.levels.x) - 1; level >= 0; --level) {
        uint count = min(queueCount[current], capacity);
        uint next = 1u - current;
        for (uint i = t; i < count; i += THREADS) {
            uvec2 node = queues[current * capacity + i];
            vec2 heights = bounds[treeTerrainBoundsIndex(configBuffer[0].lod, node.x, node.y, uint(level))];
            uint action = treeTerrainNodeAction(configBuffer[0], viewProj, eye, zoom, node.x, node.y, uint(level), heights);
            if (action == TREE_TERRAIN_CULL) {
                atomicAdd(stats[1], 1u);
            } else if (action == TREE_TERRAIN_DRAW) {
                draw(node.x, node.y, uint(level));
            } else {
                uint q = atomicAdd(queueCount[next], 4u);
                if (q + 4u <= capacity) {
                    for (uint k = 0u; k < 4u; ++k) {
                        queues[next * capacity + q + k] = uvec2(2u * node.x + k % 2u, 2u * node.y + k / 2u);
                    }
                } else {
                    atomicAdd(stats[2], 1u);
                    draw(node.x, node.y, uint(level));
                }
            }
        }
        memoryBarrierBuffer();
        memoryBarrierShared();
        barrier();
        if (t == 0u) {
            queueCount[current] = 0u;
        }
        memoryBarrierShared();
        barrier();
        current = next;
    }

    if (t == 0u) {
        drawArgs[0] = 6u * uint(configBuffer[0].lod.ranges.z) * uint(configBuffer[0].lod.ranges.z);
        drawArgs[1] = min(stats[0], configBuffer[0].lod.levels.z);
        drawArgs[2] = 0u;
        drawArgs[3] = 0u;
        dispatchArgs[0] = min(stats[0], configBuffer[0].lod.levels.z);
        dispatchArgs[1] = 1u;
        dispatchArgs[2] = 1u;
    }
}
