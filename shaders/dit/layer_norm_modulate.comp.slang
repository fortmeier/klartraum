// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// LayerNorm without affine parameters of every token's row of `width` values, then modulated by the shift and scale
// of the modulation vector: out = norm(x) (1 + scale) + shift (diffusion transformers' adaptive LayerNorm). One
// workgroup of 256 invocations per token.
import subgroup;

struct PushConstants {
    uint width;
    uint shiftOffset; // where the shift starts in the modulation vector
    uint scaleOffset;
    float epsilon;
};
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

[[vk::binding(0, 0)]] StructuredBuffer<float> X;          // [tokens][width]
[[vk::binding(1, 0)]] StructuredBuffer<float> modulation; // shift and scale among its values
[[vk::binding(2, 0)]] RWStructuredBuffer<float> Y;        // [tokens][width]

groupshared float partial[8];

// The sum of `value` over the workgroup's 256 invocations.
float groupSum(float value, uint local) {
    const float wave = WaveActiveSum(value);
    if (WaveIsFirstLane()) partial[local / WaveGetLaneCount()] = wave;
    GroupMemoryBarrierWithGroupSync();
    float total = 0.0;
    for (uint i = 0u; i < 256u / WaveGetLaneCount(); ++i) total += partial[i];
    GroupMemoryBarrierWithGroupSync();
    return total;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 groupId: SV_GroupID, uint local: SV_GroupIndex) {
    const uint base = groupId.x * pc.width;
    float sum = 0.0;
    for (uint i = local; i < pc.width; i += 256u) sum += X[base + i];
    const float mean = groupSum(sum, local) / float(pc.width);
    float squares = 0.0;
    for (uint i = local; i < pc.width; i += 256u) {
        const float d = X[base + i] - mean;
        squares += d * d;
    }
    const float inverse = rsqrt(groupSum(squares, local) / float(pc.width) + pc.epsilon);
    for (uint i = local; i < pc.width; i += 256u)
        Y[base + i] = (X[base + i] - mean) * inverse * (1.0 + modulation[pc.scaleOffset + i]) +
                      modulation[pc.shiftOffset + i];
}
