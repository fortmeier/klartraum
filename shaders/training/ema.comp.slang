// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// The exponential moving average of parameters: ema += rate (parameters - ema), with rate 1 (a copy) for the first
// warmupSteps steps of the step counter.

struct PushConstants {
    uint count;
    float rate;
    uint warmupSteps;
};
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

[[vk::binding(0, 0)]] RWStructuredBuffer<float> ema;
[[vk::binding(1, 0)]] StructuredBuffer<float> parameters;
[[vk::binding(2, 0)]] StructuredBuffer<uint> step;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 id: SV_DispatchThreadID) {
    if (id.x >= pc.count) return;
    const float rate = step[0] <= pc.warmupSteps ? 1.0 : pc.rate;
    ema[id.x] += rate * (parameters[id.x] - ema[id.x]);
}
