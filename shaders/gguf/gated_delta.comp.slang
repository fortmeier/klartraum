// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// Gated delta rule of one linear-attention value head per workgroup, for the
// params[0] active tokens in order, followed by the gated RMS normalization:
//   q, k  = L2-normalized query/key of key head (head % keyHeads); q /= sqrt(keyDim)
//   decay = exp(softplus(alpha + dtBias) * a),  beta = sigmoid(b)
//   S     = decay * S
//   S    += k (beta * (v - S^T k))^T
//   o     = S^T q
//   out   = o / rms(o) * normWeight * silu(z)
// S (keyDim x valueDim per head, value index fastest) persists across
// submissions. Invocation j owns value column j (keyDim, valueDim <= 128).
import subgroup;

struct PushConstants {
    uint keyHeads;
    uint valueHeads;
    uint keyDim;
    uint valueDim;
    uint channels; // convolution channels per token: 2 * keyHeads * keyDim + valueHeads * valueDim
    float epsilon;
};
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

[[vk::binding(0, 0)]] StructuredBuffer<float> C;     // [T][channels]: q | k | v
[[vk::binding(1, 0)]] StructuredBuffer<float> Z;     // [T][valueHeads * valueDim]
[[vk::binding(2, 0)]] StructuredBuffer<float> alpha; // [T][valueHeads]
[[vk::binding(3, 0)]] StructuredBuffer<float> beta;  // [T][valueHeads]
[[vk::binding(4, 0)]] StructuredBuffer<float> A;     // [valueHeads]
[[vk::binding(5, 0)]] StructuredBuffer<float> dtBias;
[[vk::binding(6, 0)]] StructuredBuffer<float> normWeight; // [valueDim]
[[vk::binding(7, 0)]] StructuredBuffer<uint> params;
[[vk::binding(8, 0)]] RWStructuredBuffer<float> S;   // [valueHeads][keyDim][valueDim]
[[vk::binding(9, 0)]] RWStructuredBuffer<float> O;   // [T][valueHeads * valueDim]

// The normalized query of the current token.
groupshared float queries[128];
// The normalized key of the current token.
groupshared float keys[128];
// Per subgroup, the three partial sums of workgroupSum().
groupshared float partial[3][4];

// Workgroup-wide sums of three values (all invocations must call it).
float3 workgroupSum(float3 value) {
    const float3 subgroupTotal = WaveActiveSum(value);
    if (WaveIsFirstLane()) {
        partial[0][subgroupId()] = subgroupTotal.x;
        partial[1][subgroupId()] = subgroupTotal.y;
        partial[2][subgroupId()] = subgroupTotal.z;
    }
    GroupMemoryBarrierWithGroupSync();
    float3 total = float3(0.0);
    for (uint s = 0u; s < subgroupCount(); ++s) total += float3(partial[0][s], partial[1][s], partial[2][s]);
    GroupMemoryBarrierWithGroupSync();
    return total;
}

[shader("compute")]
[numthreads(128, 1, 1)]
void main(uint3 localId: SV_GroupThreadID, uint3 groupId: SV_GroupID) {
    const uint head = groupId.x;
    const uint keyHead = head % pc.keyHeads;
    const uint j = localId.x;
    const bool keyLane = j < pc.keyDim, valueLane = j < pc.valueDim;
    const uint stateBase = head * pc.keyDim * pc.valueDim;
    const float a = A[head], bias = dtBias[head];
    const float queryScale = rsqrt(float(pc.keyDim));

    const uint tokens = params[0];
    for (uint t = 0u; t < tokens; ++t) {
        const uint row = t * pc.channels;
        const float q = keyLane ? C[row + keyHead * pc.keyDim + j] : 0.0;
        const float k = keyLane ? C[row + (pc.keyHeads + keyHead) * pc.keyDim + j] : 0.0;
        const float3 norms = workgroupSum(float3(q * q, k * k, 0.0));
        queries[j] = q * rsqrt(norms.x + 1e-6) * queryScale;
        keys[j] = k * rsqrt(norms.y + 1e-6);
        GroupMemoryBarrierWithGroupSync();

        const float x = alpha[t * pc.valueHeads + head] + bias;
        const float softplus = x > 20.0 ? x : log(1.0 + exp(x));
        const float decay = exp(softplus * a);
        const float b = 1.0 / (1.0 + exp(-beta[t * pc.valueHeads + head]));

        float output_ = 0.0;
        if (valueLane) {
            const float v = C[row + 2u * pc.keyHeads * pc.keyDim + head * pc.valueDim + j];
            float recalled = 0.0;
            for (uint i = 0u; i < pc.keyDim; ++i) recalled += S[stateBase + i * pc.valueDim + j] * keys[i];
            const float delta = b * (v - decay * recalled);
            for (uint i = 0u; i < pc.keyDim; ++i) {
                const uint index = stateBase + i * pc.valueDim + j;
                const float updated = decay * S[index] + keys[i] * delta;
                S[index] = updated;
                output_ += updated * queries[i];
            }
        }
        const float squares = workgroupSum(float3(output_ * output_, 0.0, 0.0)).x;
        if (valueLane) {
            const float z = Z[t * pc.valueHeads * pc.valueDim + head * pc.valueDim + j];
            const float normalizedOutput = output_ * rsqrt(squares / float(pc.valueDim) + pc.epsilon);
            O[t * pc.valueHeads * pc.valueDim + head * pc.valueDim + j] =
                normalizedOutput * normWeight[j] * (z / (1.0 + exp(-z)));
        }
        GroupMemoryBarrierWithGroupSync(); // queries/keys are rewritten by the next token
    }
}
