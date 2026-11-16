// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

import glsl;

// Second pass of TreeMossLod's fitting to the pool: picks the factor on the
// level boundaries for this frame from the sums of tree_lod_fit_count.comp.
// It coarsens at once to the finest step that fits, and refines by one step
// per frame only while the finer step leaves a tenth of the pool free (or
// needs no more than the current one), so the levels neither overflow the
// pool nor flip back and forth.

#define TREE_FN
#include "klartraum/tree/tree_shared.hpp"

// [0] the step in use, [1] its factor (float bits), [2] the share of the copies to leave out (float bits), then
// from [4] the sums per step.
[[vk::binding(0, 0)]] RWStructuredBuffer<uint> fit;

struct Push {
    uint capacity;
};
[[vk::push_constant]] ConstantBuffer<Push> pc;

[shader("compute")]
[numthreads(1, 1, 1)]
void main() {
    uint fitStep = min(fit[0], TREE_LOD_FIT_STEPS - 1u);
    uint fits = TREE_LOD_FIT_STEPS - 1u;
    for (uint k = TREE_LOD_FIT_STEPS; k > 0u; --k) {
        if (fit[4u + k - 1u] <= pc.capacity) {
            fits = k - 1u;
        }
    }
    if (fits > fitStep) {
        fitStep = fits;
    } else if (fitStep > 0u && (fit[4u + fitStep - 1u] <= pc.capacity - pc.capacity / 10u ||
                             fit[4u + fitStep - 1u] <= fit[4u + fitStep])) {
        fitStep -= 1u;
    }
    fit[0] = fitStep;
    fit[1] = floatBitsToUint(treeLodFitFactor(fitStep));
    // When even the coarsest step does not fit, fewer copies must come: the share to leave out, for the copies'
    // cut-off of the next frame (tree_size_cut.comp); it shrinks back gradually once there is room.
    float leaveOut = uintBitsToFloat(fit[2]);
    uint coarsest = fit[4u + TREE_LOD_FIT_STEPS - 1u];
    if (coarsest > pc.capacity) {
        leaveOut = 1.0f - (1.0f - leaveOut) * 0.95f * float(pc.capacity) / float(coarsest);
    } else if (coarsest < pc.capacity - pc.capacity / 7u) {
        leaveOut = max(leaveOut - 0.01f, 0.0f);
    }
    fit[2] = floatBitsToUint(leaveOut);
}
