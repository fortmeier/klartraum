// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

[[vk::binding(0, 0)]] StructuredBuffer<float4x4> A;
[[vk::binding(1, 0)]] StructuredBuffer<float4> B;
[[vk::binding(2, 0)]] RWStructuredBuffer<float4> R;

[shader("compute")]
[numthreads(1, 1, 1)]
void main(uint3 globalId: SV_DispatchThreadID) {
    uint index = globalId.x;
    R[index] = mul(A[index], B[index]);
}
