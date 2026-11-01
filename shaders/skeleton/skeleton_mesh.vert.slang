// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// Places the unit shapes of SkeletonMeshRenderer: with drawSpheres == 0 one
// cylinder per bone (instance = bone) between its joints, else one sphere per
// joint with a sphere (instance = that joint).
import scalar_buffer;
import skeleton_mesh_common;

struct UniformBufferObject {
    float4x4 model;
    float4x4 view;
    float4x4 proj;
    float4 cameraWorldPos;
};
[[vk::binding(0, 0)]] ConstantBuffer<UniformBufferObject> ubo;

// SkeletonGaussianParams.
struct Params {
    float4x4 toScene;
    float4 colourOpacity;
    float4 shape; // x: bone radius (scene units), y: joint radius relative to it
};

[[vk::binding(0, 1)]] ScalarBuffer<float4> joints;
// Bone pairs (2 per bone), then the joints that get a sphere.
[[vk::binding(1, 1)]] ScalarBuffer<uint> indices;
// One Params.
[[vk::binding(2, 1)]] ScalarBuffer<Params> params;
// The unit cylinder (radius 1, y from 0 to 1), then the unit sphere, as triangle lists.
[[vk::binding(3, 1)]] ScalarBuffer<float4> mesh;

float3 toScene(float4 joint) { return mul(params[0].toScene, float4(joint.xyz, 1.0)).xyz; }

[shader("vertex")]
SkeletonMeshVertex main(uint vertexIndex: SV_VulkanVertexID, uint instanceIndex: SV_VulkanInstanceID) {
    SkeletonMeshVertex output = {};
    // A shape that is not drawn moves off screen.
    const float4 hidden = float4(-2.0, -2.0, 0.0, 1.0);
    float3 unit = mesh[vertexIndex].xyz;
    uint instance = instanceIndex;
    float radius = params[0].shape.x;
    float3 position, normal;
    if (pc.drawSpheres == 0u) {
        float4 a = joints[indices[2u * instance]], b = joints[indices[2u * instance + 1u]];
        if (a.w <= 0.0 || b.w <= 0.0) {
            output.position = hidden;
            return output;
        }
        float3 first = toScene(a), axis = toScene(b) - first;
        float len = length(axis);
        float3 direction = len > 1e-9 ? axis / len : float3(0.0, 1.0, 0.0);
        float3 helper = abs(direction.y) < 0.9 ? float3(0.0, 1.0, 0.0) : float3(1.0, 0.0, 0.0);
        float3 u = normalize(cross(direction, helper));
        float3 v = cross(direction, u);
        normal = unit.x * u + unit.z * v;
        position = first + unit.y * axis + radius * normal;
    } else {
        float4 joint = joints[indices[2u * pc.boneCount + instance]];
        if (joint.w <= 0.0) {
            output.position = hidden;
            return output;
        }
        normal = unit;
        position = toScene(joint) + radius * params[0].shape.y * unit;
    }
    if (params[0].colourOpacity.a <= 0.0) {
        output.position = hidden;
        return output;
    }

    // The model matrix scales uniformly, so it keeps normals perpendicular.
    float4x4 modelView = mul(ubo.view, ubo.model);
    float4 view = mul(modelView, float4(position, 1.0));
    output.viewPosition = view.xyz;
    output.viewNormal = mul((float3x3)modelView, normal);
    output.colour = params[0].colourOpacity;
    // P[3][2] (row 3, column 2) is -1 for a perspective projection and 0 otherwise.
    bool perspective = ubo.proj[3][2] != 0.0;
    output.projection = float3(ubo.proj[2][3], ubo.proj[2][2], perspective ? 1.0 : 0.0);
    float4 clip = mul(ubo.proj, view);
    output.clipDepth = clip.z;
    // The depth buffer gets (z + w) / 2: the camera's projection may map depth to [-1, 1] (OpenGL) or [0, 1]; both
    // land in Vulkan's [0, 1] in the same order.
    output.position = float4(clip.xy, 0.5 * (clip.z + clip.w), clip.w);
    return output;
}
