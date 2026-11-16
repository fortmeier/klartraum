// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

import glsl;

// A cut-off by size (TreeSizeCut: TreeGaussianForest's copies, TreeProxies'
// list) after a frame's histogram: the lower edge of the smallest size bin
// from which on the larger trees fit the capacity (treeSizeCutTarget() on the
// CPU). It rises at once when it must, and
// falls by at most a quarter of the fade width per frame, so trees come back
// gradually.
//
// state: [0] the cut-off in log2 pixels (float bits), [1] this frame's target
// (float bits), [2] the trees at or above the target's bin, [3] 1 once the
// cut-off holds a value; then the bins (tree_library_histogram.comp).

#define TREE_FN
#include "klartraum/tree/tree_shared.hpp"

[[vk::binding(0, 0)]] RWStructuredBuffer<uint> state;

// With pc.leaveOut: [2] the share of the capacity to leave out (float bits), as tree_lod_fit.comp keeps it.
[[vk::binding(1, 0)]] StructuredBuffer<uint> fit;

struct Push {
    uint capacity;
    uint leaveOut; // 1: the capacity times 1 - fit[2]
};
[[vk::push_constant]] ConstantBuffer<Push> pc;

[shader("compute")]
[numthreads(1, 1, 1)]
void main() {
    uint capacity = pc.leaveOut != 0u ? uint(float(pc.capacity) * (1.0f - uintBitsToFloat(fit[2]))) : pc.capacity;
    uint sum = 0u;
    uint bin = TREE_SIZE_BINS;
    while (bin > 0u && sum + state[4u + bin - 1u] <= capacity) {
        bin -= 1u;
        sum += state[4u + bin];
    }
    float target = treeSizeBinEdge(bin);
    float cut = uintBitsToFloat(state[0]);
    if (state[3] == 0u || target >= cut) {
        cut = target;
    } else {
        cut = max(target, cut - 0.25f * TREE_COPY_FADE_WIDTH);
    }
    state[0] = floatBitsToUint(cut);
    state[1] = floatBitsToUint(target);
    state[2] = sum;
    state[3] = 1u;
}
