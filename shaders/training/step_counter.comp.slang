// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// Counts the submissions of a training graph: adds one to counter[0].

struct PushConstants {
    uint unused; // gives the dispatch the trailing barrier of push-constant dispatches
};
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

[[vk::binding(0, 0)]] RWStructuredBuffer<uint> counter;

[shader("compute")]
[numthreads(1, 1, 1)]
void main() {
    counter[0] += 1;
}
