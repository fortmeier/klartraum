// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

import glsl;

// One thread per tree instance: writes the trunk's stem record to the level 0 queue.

#include "tree_generate_common.slang"

[shader("compute")]
[numthreads(64, 1, 1)]
void main() {
    uint index = gl_GlobalInvocationID.x;
    // Instances of scale 0 have no tree (for example unused slots of a level of detail).
    if (index >= pc.instanceCount || treeInstances[index].posScale.w <= 0.0f) {
        return;
    }
    treeEmitStem(treeRootStem(treeInstances[index], index));
}
