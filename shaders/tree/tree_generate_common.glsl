// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// Bindings and record queues of the tree generation passes (TreeGenerator),
// and the GLSL side of the hooks tree_generate.hpp needs.

#extension GL_GOOGLE_include_directive : require

#define TREE_FN
#define TREE_CTX_PARAM
#define TREE_CTX_ARG
#include "klartraum/tree/tree_shared.hpp"

layout(std430, binding = 0) readonly buffer TreeParamsBuffer {
    TreeTypeParams treeParams[];
};
layout(std430, binding = 1) readonly buffer TreeInstancesBuffer {
    TreeInstance treeInstances[];
};
layout(std430, binding = 2) readonly buffer TreeFrameBuffer {
    TreeFrame treeFrames[];
};
// The stem queues of all levels, level k at stemOffset[k].
layout(std430, binding = 3) buffer TreeStemsBuffer {
    StemRecord treeStems[];
};
layout(std430, binding = 4) buffer TreeSegmentsBuffer {
    SegmentRecord treeSegments[];
};
layout(std430, binding = 5) buffer TreeLeavesBuffer {
    LeafRecord treeLeaves[];
};
// [0..3] stems per level, [4] segments, [5] leaves: the number of records
// written or attempted, which may exceed the capacity; [6] overflow bits.
layout(std430, binding = 6) buffer TreeCountersBuffer {
    uint treeCounters[];
};
// VkDispatchIndirectCommand for the next level pass.
layout(std430, binding = 7) buffer TreeArgsBuffer {
    uint treeArgs[];
};

layout(push_constant) uniform TreeGenPush {
    uint level;
    uint instanceCount;
    uint segmentCapacity;
    uint leafCapacity;
    uvec4 stemOffset;
    uvec4 stemCapacity;
}
pc;

const uint TREE_COUNTER_SEGMENTS = 4u;
const uint TREE_COUNTER_LEAVES = 5u;
const uint TREE_COUNTER_OVERFLOW = 6u;

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
