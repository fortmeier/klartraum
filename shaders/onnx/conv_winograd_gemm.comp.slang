// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// Pass 2 of the Winograd convolution: the 16 batched products (see conv_winograd.slang).
import conv_winograd;
import matmul;

[[vk::binding(4, 0)]] StructuredBuffer<float> u;
[[vk::binding(5, 0)]] StructuredBuffer<float> v;
[[vk::binding(6, 0)]] RWStructuredBuffer<float> m;

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 localId: SV_GroupThreadID, uint3 groupId: SV_GroupID) {
    MatMulPushConstants product;
    product.batchCount = 16;
    product.rows = pc.outputChannels;
    product.columns = tileCount();
    product.reduction = pc.inputChannels;
    product.lhsBatchCount = 16;
    product.rhsBatchCount = 16;
    matmul<IdentityEpilogue>(localId, groupId, product, u, v, m);
}
