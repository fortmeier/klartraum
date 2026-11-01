// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// The close-up window around the character and its cameras
// (CloseUpWindow::update() and cropProjection() on the GPU). One invocation.
// The window is square in pixels; its size follows the body smoothly from
// frame to frame, kept in `state`, but never falls below 90 % of the body's.

struct Camera {
    float4x4 model;
    float4x4 view;
    float4x4 proj;
    float4 cameraWorldPos;
};

struct PushConstants {
    float4x4 characterToScene;
    float aspect; // width / height of the full view
    uint jointCount;
};
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

[[vk::binding(0, 0)]] StructuredBuffer<float4> joints;
[[vk::binding(1, 0)]] ConstantBuffer<Camera> full;
// centre.xy, halfExtent.xy, halfY, initialized
[[vk::binding(2, 0)]] RWStructuredBuffer<float> state;
// the scene's camera, then the skeleton's
[[vk::binding(3, 0)]] RWStructuredBuffer<Camera> cameras;

[shader("compute")]
[numthreads(1, 1, 1)]
void main() {
    float2 lower = float2(1e9), upper = float2(-1e9);
    float4x4 toClip = mul(mul(full.proj, full.view), pc.characterToScene);
    for (uint i = 0; i < pc.jointCount; ++i) {
        float4 clip = mul(toClip, float4(joints[i].xyz, 1.0));
        float2 ndc = clip.xy / clip.w;
        lower = min(lower, ndc);
        upper = max(upper, ndc);
    }
    float target = max(0.08, 0.8 * max(upper.y - lower.y, (upper.x - lower.x) * pc.aspect));
    float2 centre = 0.5 * (lower + upper);
    float halfY = state[5] != 0.0 ? state[4] + 0.35 * (target - state[4]) : target;
    halfY = max(halfY, 0.9 * target);
    float2 halfExtent = float2(halfY / pc.aspect, halfY);
    state[0] = centre.x;
    state[1] = centre.y;
    state[2] = halfExtent.x;
    state[3] = halfExtent.y;
    state[4] = halfY;
    state[5] = 1.0;

    // Scales the window to the full NDC range and moves its centre to the origin
    // (indices are [row][column]).
    float4x4 crop = float4x4(1.0, 0.0, 0.0, 0.0,
                             0.0, 1.0, 0.0, 0.0,
                             0.0, 0.0, 1.0, 0.0,
                             0.0, 0.0, 0.0, 1.0);
    crop[0][0] = 1.0 / halfExtent.x;
    crop[1][1] = 1.0 / halfExtent.y;
    crop[0][3] = -centre.x / halfExtent.x;
    crop[1][3] = -centre.y / halfExtent.y;
    Camera scene = full;
    scene.proj = mul(crop, full.proj);
    cameras[0] = scene;
    Camera skeleton = scene;
    skeleton.model = pc.characterToScene;
    cameras[1] = skeleton;
}
