// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

import glsl;

// The Level node of Kuth et al.: one workgroup of TREE_WAVE_SIZE threads per
// stem of level pc.level writes the stem's segments and its children (stems of
// the next level, or leaves).

#include "tree_generate_common.slang"

[shader("compute")]
[numthreads(32, 1, 1)]
void main() {
    StemRecord s = treeStems[pc.stemOffset[pc.level] + gl_WorkGroupID.x];
    treeGenerateStem(s, gl_LocalInvocationID.x);
}
