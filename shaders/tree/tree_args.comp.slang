// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

import glsl;

// Dispatch arguments of the level pass pc.level: one workgroup per stem in its queue.

#include "tree_generate_common.slang"

[shader("compute")]
[numthreads(1, 1, 1)]
void main() {
    treeArgs[0] = min(treeCounters[pc.level], pc.stemCapacity[pc.level]);
    treeArgs[1] = 1u;
    treeArgs[2] = 1u;
}
