// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

import glsl;

// Bindings and record queues of the tree generation passes (TreeGenerator),
// and the GLSL side of the hooks tree_generate.hpp needs.

#define TREE_FN
#define TREE_CTX_PARAM
#define TREE_CTX_ARG
#include "klartraum/tree/tree_shared.hpp"

[[vk::binding(0, 0)]] StructuredBuffer<TreeTypeParams> treeParams;
[[vk::binding(1, 0)]] StructuredBuffer<TreeInstance> treeInstances;
[[vk::binding(2, 0)]] StructuredBuffer<TreeFrame> treeFrames;
// The stem queues of all levels, level k at stemOffset[k].
[[vk::binding(3, 0)]] RWStructuredBuffer<StemRecord> treeStems;
[[vk::binding(4, 0)]] RWStructuredBuffer<SegmentRecord> treeSegments;
[[vk::binding(5, 0)]] RWStructuredBuffer<LeafRecord> treeLeaves;
// [0..3] stems per level, [4] segments, [5] leaves: the number of records
// written or attempted, which may exceed the capacity; [6] overflow bits.
[[vk::binding(6, 0)]] RWStructuredBuffer<uint> treeCounters;
// VkDispatchIndirectCommand for the next level pass.
[[vk::binding(7, 0)]] RWStructuredBuffer<uint> treeArgs;

struct TreeGenPush {
    uint level;
    uint instanceCount;
    uint segmentCapacity;
    uint leafCapacity;
    uvec4 stemOffset;
    uvec4 stemCapacity;
};
[[vk::push_constant]] ConstantBuffer<TreeGenPush> pc;

static const uint TREE_COUNTER_SEGMENTS = 4u;
static const uint TREE_COUNTER_LEAVES = 5u;
static const uint TREE_COUNTER_OVERFLOW = 6u;

#define TREE_PARAMS(type) treeParams[type]
#define TREE_INSTANCE(index) treeInstances[index]
#define TREE_FRAME treeFrames[0]

void treeEmitStem(StemRecord s) {
    uint level = uint(treeLevel(s.ids));
    uint index = atomicAdd(treeCounters[level], 1u);
    if (index < pc.stemCapacity[level]) {
        treeStems[pc.stemOffset[level] + index] = s;
    } else {
        atomicOr(treeCounters[TREE_COUNTER_OVERFLOW], 1u << level);
    }
}

void treeEmitSegment(SegmentRecord d) {
    uint index = atomicAdd(treeCounters[TREE_COUNTER_SEGMENTS], 1u);
    if (index < pc.segmentCapacity) {
        treeSegments[index] = d;
    } else {
        atomicOr(treeCounters[TREE_COUNTER_OVERFLOW], 16u);
    }
}

void treeEmitLeaf(LeafRecord l) {
    uint index = atomicAdd(treeCounters[TREE_COUNTER_LEAVES], 1u);
    if (index < pc.leafCapacity) {
        treeLeaves[index] = l;
    } else {
        atomicOr(treeCounters[TREE_COUNTER_OVERFLOW], 32u);
    }
}

#include "klartraum/tree/tree_generate.hpp"
