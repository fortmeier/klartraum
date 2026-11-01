// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

[[vk::binding(0, 0)]] StructuredBuffer<float> A;
[[vk::binding(2, 0)]] RWStructuredBuffer<float> R;

struct UniformBufferObject {
    float multiplier;
};
[[vk::binding(0, 1)]] ConstantBuffer<UniformBufferObject> ubo;

[shader("compute")]
[numthreads(1, 1, 1)]
void main(uint3 globalId: SV_DispatchThreadID) {
    uint index = globalId.x;
    R[index] = A[index] * ubo.multiplier;
}
