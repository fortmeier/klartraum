// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

[[vk::binding(0, 0)]] StructuredBuffer<float> A;
[[vk::binding(1, 0)]] RWStructuredBuffer<float> B;
[[vk::binding(2, 0)]] RWStructuredBuffer<float> R;

[shader("compute")]
[numthreads(1, 1, 1)]
void main(uint3 globalId: SV_DispatchThreadID) {
    uint index = globalId.x;
    R[index] = A[index] * B[index];
}
