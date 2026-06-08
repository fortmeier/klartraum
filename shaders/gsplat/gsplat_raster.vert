#version 450
#extension GL_EXT_scalar_block_layout : enable

// set 0: camera UBO, as in the existing draw components
layout(set = 0, binding = 0) uniform UBO {
    mat4 model;
    mat4 view;
    mat4 proj;
    vec4 cameraWorldPos;
} ubo;

// set 1: per-splat SoA storage buffers + sorted-index buffer, in the order
// supplied to GaussianSplatRasterizer's constructor (binding == array index).
// v1 bring-up reads position + premultiplied color directly (point-cloud
// quads, no covariance/SH yet — guide §8 stage (a)); the EWA covariance and
// spherical-harmonics evaluation are added once the compute culling/sort
// stages produce the matching per-splat 2D attributes (guide §5C Stage C).
layout(scalar, set = 1, binding = 0) readonly buffer Positions   { vec3 positions[];   };
layout(scalar, set = 1, binding = 1) readonly buffer ColorsAlpha { vec4 colorsAlpha[]; };
layout(scalar, set = 1, binding = 2) readonly buffer SortedIndices { uint sortedIndices[]; };

layout(push_constant) uniform PushConstants {
    vec2  resolution;
    vec2  focal;
    float splatScale;
    int   shDegree;
} pc;

layout(location = 0) out vec4 outColor;
layout(location = 1) out vec2 outQuadUV;

// {(-1,-1), (1,-1), (-1,1), (1,1)} — a triangle-strip quad from gl_VertexIndex,
// drawn with no vertex buffer (guide §3.4).
vec2 quadCorner(uint vertexIndex) {
    const vec2 corners[4] = vec2[](vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(-1.0, 1.0), vec2(1.0, 1.0));
    return corners[vertexIndex];
}

void main() {
    uint splatId = sortedIndices[gl_InstanceIndex];
    vec3 position = positions[splatId];
    vec4 colorAlpha = colorsAlpha[splatId];

    vec4 viewPos = ubo.view * ubo.model * vec4(position, 1.0);
    vec4 clip = ubo.proj * viewPos;

    vec2 corner = quadCorner(gl_VertexIndex);
    vec2 ndcOffset = corner * pc.splatScale / pc.resolution;

    gl_Position = vec4(clip.xy / clip.w + ndcOffset, clip.z / clip.w, 1.0);

    // premultiplied: rgb already scaled by alpha so the hardware blender
    // composites back-to-front correctly (guide §3.4 "premultiplied over")
    float alpha = colorAlpha.a;
    outColor = vec4(colorAlpha.rgb * alpha, alpha);
    outQuadUV = corner;
}
