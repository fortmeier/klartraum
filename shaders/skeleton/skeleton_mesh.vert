// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#version 450

// Places the unit shapes of SkeletonMeshRenderer: with drawSpheres == 0 one
// cylinder per bone (instance = bone) between its joints, else one sphere per
// joint with a sphere (instance = that joint).

#extension GL_EXT_scalar_block_layout : require

#include "skeleton_mesh_common.glsl"

layout(set = 0, binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
    vec4 cameraWorldPos;
} ubo;

// SkeletonGaussianParams.
struct Params {
    mat4 toScene;
    vec4 colourOpacity;
    vec4 shape; // x: bone radius (scene units), y: joint radius relative to it
};

layout(scalar, set = 1, binding = 0) readonly buffer Joints { vec4 joints[]; };
// Bone pairs (2 per bone), then the joints that get a sphere.
layout(scalar, set = 1, binding = 1) readonly buffer Indices { uint indices[]; };
layout(scalar, set = 1, binding = 2) readonly buffer ParamBuffer { Params params; };
// The unit cylinder (radius 1, y from 0 to 1), then the unit sphere, as triangle lists.
layout(scalar, set = 1, binding = 3) readonly buffer Mesh { vec4 mesh[]; };

layout(location = 0) out vec3 viewPosition;
layout(location = 1) out vec3 viewNormal;
layout(location = 2) out float clipDepth;       // clip z; the NDC depth with an orthographic projection
layout(location = 3) flat out vec4 colour;      // rgb, opacity
layout(location = 4) flat out vec3 projection;  // P[3][2], P[2][2], 1 if perspective

vec3 toScene(vec4 joint) { return (params.toScene * vec4(joint.xyz, 1.0)).xyz; }

void hide() { gl_Position = vec4(-2.0, -2.0, 0.0, 1.0); }

void main() {
    vec3 unit = mesh[gl_VertexIndex].xyz;
    uint instance = uint(gl_InstanceIndex);
    float radius = params.shape.x;
    vec3 position, normal;
    if (pc.drawSpheres == 0u) {
        vec4 a = joints[indices[2u * instance]], b = joints[indices[2u * instance + 1u]];
        if (a.w <= 0.0 || b.w <= 0.0) {
            hide();
            return;
        }
        vec3 first = toScene(a), axis = toScene(b) - first;
        float len = length(axis);
        vec3 direction = len > 1e-9 ? axis / len : vec3(0.0, 1.0, 0.0);
        vec3 helper = abs(direction.y) < 0.9 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
        vec3 u = normalize(cross(direction, helper));
        vec3 v = cross(direction, u);
        normal = unit.x * u + unit.z * v;
        position = first + unit.y * axis + radius * normal;
    } else {
        vec4 joint = joints[indices[2u * pc.boneCount + instance]];
        if (joint.w <= 0.0) {
            hide();
            return;
        }
        normal = unit;
        position = toScene(joint) + radius * params.shape.y * unit;
    }
    if (params.colourOpacity.a <= 0.0) {
        hide();
        return;
    }

    // The model matrix scales uniformly, so it keeps normals perpendicular.
    mat4 modelView = ubo.view * ubo.model;
    vec4 view = modelView * vec4(position, 1.0);
    viewPosition = view.xyz;
    viewNormal = mat3(modelView) * normal;
    colour = params.colourOpacity;
    bool perspective = ubo.proj[2][3] != 0.0;
    projection = vec3(ubo.proj[3][2], ubo.proj[2][2], perspective ? 1.0 : 0.0);
    vec4 clip = ubo.proj * view;
    clipDepth = clip.z;
    // The depth buffer gets (z + w) / 2: the camera's projection may map depth to [-1, 1] (OpenGL) or [0, 1]; both
    // land in Vulkan's [0, 1] in the same order.
    gl_Position = vec4(clip.xy, 0.5 * (clip.z + clip.w), clip.w);
}
