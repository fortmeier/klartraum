// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// Writes transformed copies of Gaussians: p' = s R p + t, orientations turned
// by R, sizes scaled by s, SH coefficients rotated per band. The transform
// comes from make_transform.comp.slang.
import gaussian_common;

struct PushConstants {
    uint count;
};
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

[[vk::binding(0, 0)]] ScalarBuffer<float3> pos;
[[vk::binding(1, 0)]] ScalarBuffer<float4> rot;
[[vk::binding(2, 0)]] ScalarBuffer<float3> scale;
[[vk::binding(3, 0)]] ScalarBuffer<float4> colAlpha;
[[vk::binding(4, 0)]] ScalarBuffer<float> shR;
[[vk::binding(5, 0)]] ScalarBuffer<float> shG;
[[vk::binding(6, 0)]] ScalarBuffer<float> shB;
[[vk::binding(7, 0)]] ScalarBuffer<float> t;
[[vk::binding(8, 0)]] RWScalarBuffer<float3> outPos;
[[vk::binding(9, 0)]] RWScalarBuffer<float4> outRot;
[[vk::binding(10, 0)]] RWScalarBuffer<float3> outScale;
[[vk::binding(11, 0)]] RWScalarBuffer<float4> outColAlpha;
[[vk::binding(12, 0)]] RWScalarBuffer<float> outShR;
[[vk::binding(13, 0)]] RWScalarBuffer<float> outShG;
[[vk::binding(14, 0)]] RWScalarBuffer<float> outShB;

// Rotates one channel's 15 coefficients of Gaussian i (coefficient-major).
void rotateSh(uint i, ScalarBuffer<float> src, RWScalarBuffer<float> dst) {
    float c[15];
    for (int k = 0; k < 15; ++k) {
        c[k] = src[k * pc.count + i];
    }
    for (int row = 0; row < 3; ++row) {
        float v = 0.0;
        for (int col = 0; col < 3; ++col) {
            v += t[kShBand1 + row * 3 + col] * c[col];
        }
        dst[row * pc.count + i] = v;
    }
    for (int row = 0; row < 5; ++row) {
        float v = 0.0;
        for (int col = 0; col < 5; ++col) {
            v += t[kShBand2 + row * 5 + col] * c[3 + col];
        }
        dst[(3 + row) * pc.count + i] = v;
    }
    for (int row = 0; row < 7; ++row) {
        float v = 0.0;
        for (int col = 0; col < 7; ++col) {
            v += t[kShBand3 + row * 7 + col] * c[8 + col];
        }
        dst[(8 + row) * pc.count + i] = v;
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 globalId: SV_DispatchThreadID) {
    uint i = globalId.x;
    if (i >= pc.count) {
        return;
    }
    float3 translation = float3(t[kTranslation], t[kTranslation + 1], t[kTranslation + 2]);
    float s = t[kScale];
    float4 q = float4(t[kRotation], t[kRotation + 1], t[kRotation + 2], t[kRotation + 3]);

    outPos[i] = s * mul(quatToMat3(q), pos[i]) + translation;
    outRot[i] = normalize(quatMul(q, rot[i]));
    outScale[i] = s * scale[i];
    outColAlpha[i] = colAlpha[i];
    rotateSh(i, shR, outShR);
    rotateSh(i, shG, outShG);
    rotateSh(i, shB, outShB);
}
