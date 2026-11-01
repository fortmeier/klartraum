// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// Draws a skeleton as Gaussians (SkeletonGaussians): per bone, samplesPerBone
// thin Gaussians spaced evenly along it with their long (y) axis along the
// bone, then one round Gaussian per listed joint. Joints are moved into the
// scene by toScene; sizes are in scene units.
import scalar_buffer;

struct PushConstants {
    uint boneCount;
    uint samplesPerBone;
    uint jointCount;
    uint total;
};
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

struct Params {
    float4x4 toScene;
    float4 colourOpacity;
    float4 shape; // x: bone radius, y: joint radius relative to it
};

[[vk::binding(0, 0)]] ScalarBuffer<float4> joints;
// Bone pairs (2 per bone), then the joints that get a sphere.
[[vk::binding(1, 0)]] ScalarBuffer<uint> indices;
// One Params.
[[vk::binding(2, 0)]] ScalarBuffer<Params> params;
[[vk::binding(3, 0)]] RWScalarBuffer<float3> outPos;
[[vk::binding(4, 0)]] RWScalarBuffer<float4> outRot;
[[vk::binding(5, 0)]] RWScalarBuffer<float3> outScale;
[[vk::binding(6, 0)]] RWScalarBuffer<float4> outColAlpha;
[[vk::binding(7, 0)]] RWScalarBuffer<float> outShR;
[[vk::binding(8, 0)]] RWScalarBuffer<float> outShG;
[[vk::binding(9, 0)]] RWScalarBuffer<float> outShB;

static const float kC0 = 0.282094792;

float3 toScene(uint joint) {
    return mul(params[0].toScene, float4(joints[joint].xyz, 1.0)).xyz;
}

// The rotation (x, y, z, w) that turns +y onto the unit vector d.
float4 rotationFromUp(float3 d) {
    float c = d.y; // dot((0, 1, 0), d)
    if (c < -0.99999) {
        return float4(1.0, 0.0, 0.0, 0.0); // half a turn about x
    }
    return normalize(float4(d.z, 0.0, -d.x, 1.0 + c)); // cross((0, 1, 0), d), 1 + dot
}

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 globalId: SV_DispatchThreadID) {
    uint i = globalId.x;
    if (i >= pc.total) {
        return;
    }
    float radius = params[0].shape.x;
    uint boneSamples = pc.boneCount * pc.samplesPerBone;
    if (i < boneSamples) {
        uint bone = i / pc.samplesPerBone;
        uint k = i % pc.samplesPerBone;
        float3 a = toScene(indices[2u * bone]);
        float3 b = toScene(indices[2u * bone + 1u]);
        float3 along = b - a;
        float len = length(along);
        outPos[i] = lerp(a, b, (float(k) + 0.5) / float(pc.samplesPerBone));
        outRot[i] = len > 1e-9 ? rotationFromUp(along / len) : float4(0.0, 0.0, 0.0, 1.0);
        outScale[i] = float3(radius, max(0.6 * len / float(pc.samplesPerBone), radius), radius);
    } else {
        uint joint = indices[2u * pc.boneCount + (i - boneSamples)];
        outPos[i] = toScene(joint);
        outRot[i] = float4(0.0, 0.0, 0.0, 1.0);
        outScale[i] = float3(radius * params[0].shape.y);
    }
    outColAlpha[i] = float4((params[0].colourOpacity.rgb - 0.5) / kC0, params[0].colourOpacity.a);
    for (uint c = 0u; c < 15u; ++c) {
        outShR[c * pc.total + i] = 0.0;
        outShG[c * pc.total + i] = 0.0;
        outShB[c * pc.total + i] = 0.0;
    }
}
